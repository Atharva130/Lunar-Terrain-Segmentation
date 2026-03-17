"""
model.py - FIXED v3
Uses SMP Unet directly but shares the encoder properly
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class MultiTaskUNet(nn.Module):
    """
    Multi-Task U-Net for Lunar Surface Segmentation

    Shared MobileNetV2 encoder (pretrained ImageNet)
    Rock Decoder    -> 4-class (sky, rock, boulder, ground)
    Crater Decoder  -> 1-class binary (crater / background)
    """

    def __init__(self):
        super().__init__()

        # Two complete SMP UNets
        self.rock_unet = smp.Unet(
            encoder_name    = 'mobilenet_v2',
            encoder_weights = 'imagenet',
            in_channels     = 3,
            classes         = 4,
        )
        self.crater_unet = smp.Unet(
            encoder_name    = 'mobilenet_v2',
            encoder_weights = 'imagenet',
            in_channels     = 3,
            classes         = 1,
        )

        # Share encoder: crater_unet uses rock_unet's encoder
        # This is the KEY architectural contribution
        self.crater_unet.encoder = self.rock_unet.encoder

        enc   = sum(p.numel() for p in self.rock_unet.encoder.parameters())
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"MultiTaskUNet initialized")
        print(f"  Encoder     : mobilenet_v2 (pretrained=imagenet)")
        print(f"  Encoder params  : {enc:,}")
        print(f"  Total params    : {total:,}")
        print(f"  Rock head   : 4 classes (sky/rock/boulder/ground)")
        print(f"  Crater head : 1 class  (binary)")

    def forward(self, x, task='both'):
        """
        x    : (B, 3, H, W)
        task : 'rock' | 'crater' | 'both'
        """
        out = {}

        if task in ('rock', 'both'):
            out['rock'] = self.rock_unet(x)       # (B, 4, H, W)

        if task in ('crater', 'both'):
            out['crater'] = self.crater_unet(x)   # (B, 1, H, W)

        return out


class MultiTaskLoss(nn.Module):
    """
    Masked loss - only relevant head computed per batch.
    Keio batch   -> rock loss only
    Crater batch -> crater loss only
    Shared encoder gets gradients from BOTH tasks.
    """

    def __init__(self):
        super().__init__()

        # Rock: Dice + weighted CrossEntropy (handles class imbalance)
        # sky=4% upweighted, rock=4%, boulder=2% upweighted, ground=74% downweighted
        weights = torch.tensor([0.5, 3.0, 4.0, 0.5])
        self.rock_ce   = nn.CrossEntropyLoss(weight=weights)
        self.rock_dice = smp.losses.DiceLoss(mode='multiclass', from_logits=True)

        # Crater: Dice + BCE (binary)
        self.crater_bce  = nn.BCEWithLogitsLoss()
        self.crater_dice = smp.losses.DiceLoss(mode='binary', from_logits=True)

    def rock_loss(self, pred, target):
        dice = self.rock_dice(pred, target)
        ce   = self.rock_ce(pred, target.long())
        return 0.5 * dice + 0.5 * ce

    def crater_loss(self, pred, target):
        t = (target.float() / 255.0)
        t = (t > 0.5).float()
        bce  = self.crater_bce(pred, t)
        dice = self.crater_dice(pred, t)
        return 0.5 * bce + 0.5 * dice

    def forward(self, outputs, targets, task):
        if task == 'rock':
            loss = self.rock_loss(outputs['rock'], targets['rock'])
            return loss, {'rock_loss': loss.item(), 'crater_loss': 0.0}
        elif task == 'crater':
            loss = self.crater_loss(outputs['crater'], targets['crater'])
            return loss, {'rock_loss': 0.0, 'crater_loss': loss.item()}
        else:
            r = self.rock_loss(outputs['rock'],     targets['rock'])
            c = self.crater_loss(outputs['crater'], targets['crater'])
            return 0.5*r + 0.5*c, {'rock_loss': r.item(), 'crater_loss': c.item()}


# Sanity test
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nTesting MultiTaskUNet on {device.upper()}\n")

    model     = MultiTaskUNet().to(device)
    criterion = MultiTaskLoss().to(device)

    dummy = torch.randn(2, 3, 256, 256).to(device)

    print("\nForward pass (batch=2, 256x256):")
    with torch.no_grad():
        out_rock   = model(dummy, task='rock')
        out_crater = model(dummy, task='crater')
        out_both   = model(dummy, task='both')

    print(f"  Rock   output : {out_rock['rock'].shape}")
    print(f"  Crater output : {out_crater['crater'].shape}")
    print(f"  Both - rock   : {out_both['rock'].shape}")
    print(f"  Both - crater : {out_both['crater'].shape}")

    rock_target   = torch.randint(0, 4, (2, 256, 256)).to(device)
    crater_target = (torch.rand(2, 1, 256, 256) * 255).to(device)

    loss_r, _ = criterion(out_both, {'rock': rock_target, 'crater': crater_target}, 'rock')
    loss_c, _ = criterion(out_both, {'rock': rock_target, 'crater': crater_target}, 'crater')

    print(f"\nLoss test:")
    print(f"  Rock   loss : {loss_r.item():.4f}")
    print(f"  Crater loss : {loss_c.item():.4f}")
    print(f"\n  ✅ Model test PASSED - Ready for training!")
    print(f"  Run: python train.py")
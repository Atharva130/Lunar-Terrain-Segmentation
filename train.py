"""
train.py
Full training pipeline for Multi-Task Lunar U-Net

Run: python train.py

Trains on:
  - Keio dataset  → rock/terrain segmentation
  - Crater dataset → crater detection
Using alternating batch strategy with masked loss
"""

import os
import time
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import MultiTaskUNet, MultiTaskLoss

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
CFG = {
    # Paths
    'keio_dir'   : r"D:\Lunar\processed\keio",
    'crater_dir' : r"D:\Lunar\processed\craters",
    'out_dir'    : r"D:\Lunar\checkpoints",
    'log_dir'    : r"D:\Lunar\logs",

    # Training
    'img_size'   : 256,
    'batch_size' : 16,       # increased for speed
    'epochs'     : 30,
    'lr'         : 1e-4,
    'weight_decay': 1e-5,
    'seed'       : 42,

    # Mixed precision (faster + less memory on RTX 4050)
    'amp'        : True,

    # Save best model
    'save_every' : 10,       # save checkpoint every N epochs
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ══════════════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════════════

def get_transforms(split, img_size):
    if split == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.Normalize(mean=(0.485,0.456,0.406),
                        std =(0.229,0.224,0.225)),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485,0.456,0.406),
                        std =(0.229,0.224,0.225)),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})


class KeioDataset(Dataset):
    """Keio rock/terrain segmentation dataset"""

    def __init__(self, root, split='train', transform=None):
        self.img_dir = os.path.join(root, split, 'images')
        self.msk_dir = os.path.join(root, split, 'masks')
        self.transform = transform

        self.images = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith('.png')
        ])
        print(f"  KeioDataset [{split}]: {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]
        stem  = fname.rsplit('.', 1)[0]

        img  = cv2.imread(os.path.join(self.img_dir, fname))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(
            os.path.join(self.msk_dir, stem + '_mask.png'),
            cv2.IMREAD_GRAYSCALE
        )

        if self.transform:
            aug  = self.transform(image=img, mask=mask)
            img  = aug['image']
            mask = aug['mask'].long()

        return img, mask


class CraterDataset(Dataset):
    """Crater segmentation dataset (SAM-generated masks)"""

    def __init__(self, root, split='train', transform=None):
        # craters split structure: train/val/test
        split_map = {'train':'train', 'val':'val', 'test':'test'}
        sp = split_map.get(split, split)

        self.img_dir = os.path.join(root, sp, 'images')
        self.msk_dir = os.path.join(root, sp, 'masks')
        self.transform = transform

        if not os.path.exists(self.img_dir):
            self.images = []
            print(f"  CraterDataset [{split}]: 0 images (path not found)")
            return

        self.images = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith('.png')
        ])
        print(f"  CraterDataset [{split}]: {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]
        stem  = fname.rsplit('.', 1)[0]

        img  = cv2.imread(os.path.join(self.img_dir, fname))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.msk_dir, stem + '_mask.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        if self.transform:
            aug  = self.transform(image=img, mask=mask)
            img  = aug['image']
            mask = aug['mask']

        # Crater mask shape: (H, W) → (1, H, W) float
        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0).float()
        else:
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return img, mask


# ══════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════

def iou_score(pred, target, num_classes):
    """Mean IoU for multi-class segmentation"""
    pred   = pred.argmax(dim=1).cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    ious = []
    for c in range(num_classes):
        inter = ((pred == c) & (target == c)).sum()
        union = ((pred == c) | (target == c)).sum()
        if union == 0:
            continue
        ious.append(inter / union)
    return np.mean(ious) if ious else 0.0

def binary_iou(pred_logits, target):
    """IoU for binary crater segmentation"""
    pred   = (torch.sigmoid(pred_logits) > 0.5).cpu().numpy().flatten()
    target = (target > 127).cpu().numpy().flatten()
    inter  = (pred & target).sum()
    union  = (pred | target).sum()
    return inter / union if union > 0 else 0.0

# ══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════

def train_one_epoch(model, rock_loader, crater_loader,
                    optimizer, criterion, scaler, epoch):
    model.train()

    rock_iter   = iter(rock_loader)
    crater_iter = iter(crater_loader)

    total_loss  = 0.0
    rock_loss_sum   = 0.0
    crater_loss_sum = 0.0
    steps = 0

    # Alternate between rock and crater batches
    max_steps = min(len(rock_loader), 200) * 2  # cap at 200 steps per epoch

    pbar = tqdm(range(max_steps), desc=f"Epoch {epoch:03d}", ncols=80)

    for step in pbar:
        # Alternate task each step
        use_rock = (step % 2 == 0)

        optimizer.zero_grad()

        if use_rock:
            try:
                imgs, masks = next(rock_iter)
            except StopIteration:
                rock_iter = iter(rock_loader)
                imgs, masks = next(rock_iter)

            imgs  = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            with torch.cuda.amp.autocast(enabled=CFG['amp']):  # type: ignore
                outputs = model(imgs, task='rock')
                loss, loss_dict = criterion(
                    outputs, {'rock': masks, 'crater': None}, 'rock')

            rock_loss_sum += loss_dict['rock_loss']

        else:
            try:
                imgs, masks = next(crater_iter)
            except StopIteration:
                crater_iter = iter(crater_loader)
                imgs, masks = next(crater_iter)

            imgs  = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            with torch.cuda.amp.autocast(enabled=CFG['amp']):  # type: ignore
                outputs = model(imgs, task='crater')
                loss, loss_dict = criterion(
                    outputs, {'rock': None, 'crater': masks}, 'crater')

            crater_loss_sum += loss_dict['crater_loss']

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        steps += 1

        pbar.set_postfix({
            'loss': f"{total_loss/steps:.4f}",
            'rock': f"{rock_loss_sum/max(1,step//2+1):.3f}",
            'crat': f"{crater_loss_sum/max(1,step//2+1):.3f}",
        })

    return {
        'total'  : total_loss / steps,
        'rock'   : rock_loss_sum / (steps // 2 + 1),
        'crater' : crater_loss_sum / (steps // 2 + 1),
    }


@torch.no_grad()
def validate(model, rock_val_loader, crater_val_loader, criterion):
    model.eval()

    rock_iou_sum   = 0.0
    crater_iou_sum = 0.0
    rock_loss_sum  = 0.0
    crat_loss_sum  = 0.0

    # Validate rock head
    for imgs, masks in rock_val_loader:
        imgs  = imgs.to(DEVICE)
        masks = masks.to(DEVICE)
        outputs = model(imgs, task='rock')
        loss, _ = criterion(outputs, {'rock': masks, 'crater': None}, 'rock')
        rock_loss_sum  += loss.item()
        rock_iou_sum   += iou_score(outputs['rock'], masks, 4)

    # Validate crater head
    for imgs, masks in crater_val_loader:
        imgs  = imgs.to(DEVICE)
        masks = masks.to(DEVICE)
        outputs = model(imgs, task='crater')
        loss, _ = criterion(outputs, {'rock': None, 'crater': masks}, 'crater')
        crat_loss_sum  += loss.item()
        crater_iou_sum += binary_iou(outputs['crater'], masks)

    n_rock   = len(rock_val_loader)
    n_crater = len(crater_val_loader)

    return {
        'rock_loss'  : rock_loss_sum  / n_rock,
        'crater_loss': crat_loss_sum  / n_crater,
        'rock_iou'   : rock_iou_sum   / n_rock,
        'crater_iou' : crater_iou_sum / n_crater,
    }


def save_training_plot(history, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('Training Progress', color='white', fontsize=13)

    epochs = range(1, len(history['train_loss']) + 1)
    plots = [
        ('Total Loss',   'train_loss',   None,         '#3b82f6'),
        ('Rock IoU',     'rock_iou',     None,         '#10b981'),
        ('Crater IoU',   'crater_iou',   None,         '#f59e0b'),
    ]

    for ax, (title, key, _, color) in zip(axes, plots):
        ax.set_facecolor('#111827')
        ax.plot(epochs, history[key], color=color, lw=2, label='Val')
        ax.set_title(title, color='white', fontsize=10)
        ax.tick_params(colors='gray')
        ax.spines[:].set_color('#374151')
        ax.legend(facecolor='#1f2937', labelcolor='white', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_plot.png'),
                dpi=120, facecolor='#0d1117', bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':

    set_seed(CFG['seed'])
    os.makedirs(CFG['out_dir'], exist_ok=True)
    os.makedirs(CFG['log_dir'], exist_ok=True)

    print("=" * 58)
    print("  MULTI-TASK LUNAR U-NET — TRAINING")
    print("=" * 58)
    print(f"  Device     : {DEVICE.upper()}")
    print(f"  Batch size : {CFG['batch_size']}")
    print(f"  Epochs     : {CFG['epochs']}")
    print(f"  Image size : {CFG['img_size']}x{CFG['img_size']}")
    print(f"  AMP        : {CFG['amp']}")
    print()

    # ── Transforms ────────────────────────────────────────────
    train_tf = get_transforms('train', CFG['img_size'])
    val_tf   = get_transforms('val',   CFG['img_size'])

    # ── Datasets ──────────────────────────────────────────────
    print("Loading datasets...")
    rock_train = KeioDataset(CFG['keio_dir'],   'train', train_tf)
    rock_val   = KeioDataset(CFG['keio_dir'],   'val',   val_tf)
    crat_train = CraterDataset(CFG['crater_dir'],'train',train_tf)
    crat_val   = CraterDataset(CFG['crater_dir'],'val',  val_tf)

    # ── DataLoaders ───────────────────────────────────────────
    rock_train_dl = DataLoader(rock_train, CFG['batch_size'],
                               shuffle=True,  num_workers=2, pin_memory=True)
    rock_val_dl   = DataLoader(rock_val,   CFG['batch_size'],
                               shuffle=False, num_workers=2, pin_memory=True)
    crat_train_dl = DataLoader(crat_train, CFG['batch_size'],
                               shuffle=True,  num_workers=2, pin_memory=True)
    crat_val_dl   = DataLoader(crat_val,   CFG['batch_size'],
                               shuffle=False, num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────
    print("\nInitializing model...")
    model     = MultiTaskUNet().to(DEVICE)
    criterion = MultiTaskLoss().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG['epochs'], eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler(enabled=CFG['amp'])  # type: ignore

    # ── Training ──────────────────────────────────────────────
    print(f"\nStarting training for {CFG['epochs']} epochs...\n")

    history = {
        'train_loss': [], 'rock_iou': [], 'crater_iou': [],
        'rock_loss': [], 'crater_loss': [],
    }
    best_rock_iou   = 0.0
    best_crater_iou = 0.0

    for epoch in range(1, CFG['epochs'] + 1):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, rock_train_dl, crat_train_dl,
            optimizer, criterion, scaler, epoch)

        # Validate
        val_metrics = validate(
            model, rock_val_dl, crat_val_dl, criterion)

        scheduler.step()
        elapsed = time.time() - t0

        # Log
        history['train_loss'].append(train_metrics['total'])
        history['rock_iou'].append(val_metrics['rock_iou'])
        history['crater_iou'].append(val_metrics['crater_iou'])

        print(f"\nEpoch {epoch:03d}/{CFG['epochs']} "
              f"({elapsed:.0f}s) | "
              f"Loss: {train_metrics['total']:.4f} | "
              f"Rock IoU: {val_metrics['rock_iou']:.4f} | "
              f"Crater IoU: {val_metrics['crater_iou']:.4f}")

        # Save best models
        if val_metrics['rock_iou'] > best_rock_iou:
            best_rock_iou = val_metrics['rock_iou']
            torch.save(model.state_dict(),
                       os.path.join(CFG['out_dir'], 'best_rock.pth'))
            print(f"  ✅ New best Rock IoU: {best_rock_iou:.4f} → saved")

        if val_metrics['crater_iou'] > best_crater_iou:
            best_crater_iou = val_metrics['crater_iou']
            torch.save(model.state_dict(),
                       os.path.join(CFG['out_dir'], 'best_crater.pth'))
            print(f"  ✅ New best Crater IoU: {best_crater_iou:.4f} → saved")

        # Periodic checkpoint
        if epoch % CFG['save_every'] == 0:
            torch.save({
                'epoch'     : epoch,
                'model'     : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'history'   : history,
            }, os.path.join(CFG['out_dir'], f'checkpoint_ep{epoch:03d}.pth'))

        # Save training plot
        if epoch % 5 == 0:
            save_training_plot(history, CFG['log_dir'])

    # ── Final save ────────────────────────────────────────────
    torch.save(model.state_dict(),
               os.path.join(CFG['out_dir'], 'final_model.pth'))

    print("\n" + "=" * 58)
    print("  TRAINING COMPLETE")
    print("=" * 58)
    print(f"  Best Rock IoU   : {best_rock_iou:.4f}")
    print(f"  Best Crater IoU : {best_crater_iou:.4f}")
    print(f"  Models saved to : {CFG['out_dir']}")
    print("\n  Next: python evaluate.py")
    print("=" * 58)
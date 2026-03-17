"""
inference.py - Fixed version
Usage:
  python inference.py --image "D:\Lunar\test.png"   ← single image
  python inference.py --folder "D:\Lunar\myfolder"  ← whole folder
  python inference.py                                ← runs on test sets
"""

import os
import sys
import cv2
import torch
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from model import MultiTaskUNet

# ── CONFIG ────────────────────────────────────────────────────
ROCK_CKPT   = r"D:\Lunar\checkpoints\best_rock.pth"
OUT_DIR     = r"D:\Lunar\inference_results"
IMG_SIZE    = 256
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
KEIO_TEST   = r"D:\Lunar\processed\keio\test\images"
CRATER_TEST = r"D:\Lunar\processed\craters\test\images"

# Color map
ROCK_COLORS = {
    0: ([135, 206, 235], 'Sky'),
    1: ([34,  197,  94], 'Rock'),
    2: ([59,  130, 246], 'Boulder'),
    3: ([71,   85, 105], 'Ground'),
}
CRATER_COLOR = ([255, 140, 0], 'Crater')

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

os.makedirs(OUT_DIR, exist_ok=True)

# ── HELPERS ───────────────────────────────────────────────────

def clean_filename(fname):
    """Remove any characters that cause OSError on Windows"""
    import re
    name = os.path.splitext(os.path.basename(fname))[0]
    name = re.sub(r'[\\/:*?"<>|]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name

def preprocess(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

def predict(model, tensor):
    model.eval()
    with torch.no_grad():
        out    = model(tensor.to(DEVICE), task='both')
        rock   = out['rock'].argmax(dim=1).squeeze().cpu().numpy()
        crater = (torch.sigmoid(out['crater']) > 0.5).squeeze().cpu().numpy()
    return rock, crater

def save_4panel(orig_bgr, rock_pred, crater_pred, out_path, title):
    h, w     = orig_bgr.shape[:2]
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    # Rock only mask
    rock_only = np.zeros((rock_pred.shape[0], rock_pred.shape[1], 3), dtype=np.uint8)
    for cls_id, (rgb, _) in ROCK_COLORS.items():
        rock_only[rock_pred == cls_id] = rgb
    rock_only = cv2.resize(rock_only, (w, h), interpolation=cv2.INTER_NEAREST)

    # Crater only mask
    crat_only = np.zeros((h, w, 3), dtype=np.uint8)
    crat_r    = cv2.resize(crater_pred.astype(np.uint8), (w, h),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    crat_only[crat_r] = CRATER_COLOR[0]

    # Combined overlay
    full_mask = np.zeros((rock_pred.shape[0], rock_pred.shape[1], 3), dtype=np.uint8)
    for cls_id, (rgb, _) in ROCK_COLORS.items():
        full_mask[rock_pred == cls_id] = rgb
    full_mask[crater_pred.astype(bool)] = CRATER_COLOR[0]
    full_mask = cv2.resize(full_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay   = cv2.addWeighted(orig_rgb, 0.4, full_mask, 0.6, 0)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle(f'Lunar Surface Segmentation — {title}',
                 fontsize=12, fontweight='bold')

    for ax, (img, panel_title, tc) in zip(axes, [
        (orig_rgb,  'Original Image',         '#374151'),
        (rock_only, 'Rock Segmentation',      '#10b981'),
        (crat_only, 'Crater Detection (SAM)', '#f59e0b'),
        (overlay,   'Combined Output',        '#3b82f6'),
    ]):
        ax.imshow(img)
        ax.set_title(panel_title, color=tc, fontsize=11,
                     fontweight='bold', pad=6)
        ax.axis('off')

    # Legend
    legend_items = [
        mpatches.Patch(color=np.array(rgb)/255, label=name)
        for _, (rgb, name) in ROCK_COLORS.items()
    ]
    legend_items.append(
        mpatches.Patch(color=np.array(CRATER_COLOR[0])/255,
                       label=CRATER_COLOR[1]))
    fig.legend(handles=legend_items, loc='lower center', ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_path, dpi=160, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved: {out_path}")


def run_single(img_path, model):
    """Predict a single image file"""
    print(f"\n  Predicting: {img_path}")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"  ERROR: Could not read image at {img_path}")
        return

    rock_pred, crat_pred = predict(model, preprocess(img_bgr))

    clean_name = clean_filename(img_path)
    out_path   = os.path.join(OUT_DIR, f'{clean_name}_result.png')
    save_4panel(img_bgr, rock_pred, crat_pred, out_path, clean_name)


def run_on_folder(img_dir, model, label, max_images=6):
    """Run inference on all images in a folder"""
    if not os.path.exists(img_dir):
        print(f"  Skipping {label} — folder not found: {img_dir}")
        return

    files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])[:max_images]

    print(f"\n  {label}: {len(files)} images")
    for fname in files:
        img_path = os.path.join(img_dir, fname)
        img_bgr  = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  Skipping unreadable: {fname}")
            continue

        rock_pred, crat_pred = predict(model, preprocess(img_bgr))

        # Clean filename to avoid Windows OSError
        clean_name = clean_filename(fname)
        out_path   = os.path.join(OUT_DIR, f'{label}_{clean_name}_result.png')
        save_4panel(img_bgr, rock_pred, crat_pred, out_path, fname)


# ── MAIN ──────────────────────────────────────────────────────
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Lunar Segmentation Inference')
    parser.add_argument('--image',  type=str, default=None,
                        help='Path to a single image file')
    parser.add_argument('--folder', type=str, default=None,
                        help='Path to a folder of images')
    args = parser.parse_args()

    print("=" * 55)
    print("  LUNAR SEGMENTATION INFERENCE")
    print("=" * 55)
    print(f"  Device : {DEVICE.upper()}")
    print(f"  Output : {OUT_DIR}")

    # Load model
    print("\nLoading model...")
    model = MultiTaskUNet().to(DEVICE)
    if os.path.exists(ROCK_CKPT):
        model.load_state_dict(
            torch.load(ROCK_CKPT, map_location=DEVICE, weights_only=True))
        print(f"  Loaded: {ROCK_CKPT}")
    else:
        print(f"  Checkpoint not found: {ROCK_CKPT}")
        sys.exit(1)

    # ── DECIDE WHAT TO RUN ────────────────────────────────────
    if args.image:
        # Single image mode
        run_single(args.image, model)

    elif args.folder:
        # Folder mode
        run_on_folder(args.folder, model,
                      label=os.path.basename(args.folder),
                      max_images=999)

    else:
        # Default: run on both test sets
        run_on_folder(KEIO_TEST,   model, 'keio',   max_images=6)
        run_on_folder(CRATER_TEST, model, 'crater', max_images=6)

    print(f"\n  Done! Results in: {OUT_DIR}")
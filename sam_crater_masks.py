"""
sam_crater_masks.py
Run: python sam_crater_masks.py

Uses Segment Anything Model (SAM) to convert
YOLO bounding boxes → precise pixel-level crater masks

BEFORE RUNNING:
  Download SAM checkpoint (~375MB):
  URL: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
  Save to: D:\Lunar\sam_checkpoint\sam_vit_b_01ec64.pth
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────
SAM_CHECKPOINT = r"D:\Lunar\sam_checkpoint\sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"          # vit_b fits in 6GB VRAM

SPLITS = {
    'train': (
        r"D:\Lunar\datasets\craters\craters\train\images",
        r"D:\Lunar\datasets\craters\craters\train\labels",
    ),
    'val': (
        r"D:\Lunar\datasets\craters\craters\valid\images",
        r"D:\Lunar\datasets\craters\craters\valid\labels",
    ),
    'test': (
        r"D:\Lunar\datasets\craters\craters\test\images",
        r"D:\Lunar\datasets\craters\craters\test\labels",
    ),
}

OUT_DIR   = r"D:\Lunar\processed\craters"
IMG_SIZE  = 256
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

def check_sam_checkpoint():
    """Verify SAM checkpoint exists, give download instructions if not"""
    if os.path.exists(SAM_CHECKPOINT):
        size_mb = os.path.getsize(SAM_CHECKPOINT) / 1e6
        print(f"  ✅ SAM checkpoint found ({size_mb:.0f} MB)")
        return True

    print(f"  ❌ SAM checkpoint NOT found at:")
    print(f"     {SAM_CHECKPOINT}")
    print()
    print("  Download it now:")
    print("  Option A — PowerShell:")
    print("    New-Item -ItemType Directory -Force D:\\Lunar\\sam_checkpoint")
    print("    Invoke-WebRequest -Uri https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -OutFile D:\\Lunar\\sam_checkpoint\\sam_vit_b_01ec64.pth")
    print()
    print("  Option B — Browser:")
    print("    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
    print("    Save to: D:\\Lunar\\sam_checkpoint\\")
    return False

def load_sam():
    """Load SAM model onto GPU"""
    from segment_anything import sam_model_registry, SamPredictor
    print(f"  Loading SAM ({SAM_MODEL_TYPE}) on {DEVICE.upper()}...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    print(f"  ✅ SAM loaded on {DEVICE.upper()}")
    return predictor

def yolo_to_pixel_box(xc, yc, bw, bh, img_w, img_h):
    """Convert YOLO normalized coords → pixel [x1,y1,x2,y2]"""
    x1 = int((xc - bw/2) * img_w)
    y1 = int((yc - bh/2) * img_h)
    x2 = int((xc + bw/2) * img_w)
    y2 = int((yc + bh/2) * img_h)
    # Clamp to image bounds
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img_w-1, x2); y2 = min(img_h-1, y2)
    return [x1, y1, x2, y2]

def read_yolo_labels(label_path):
    """Read YOLO .txt file → list of (class, xc, yc, w, h)"""
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                boxes.append((cls, xc, yc, bw, bh))
    return boxes

def generate_crater_mask_sam(predictor, image_rgb, boxes_pixel):
    """
    Use SAM to generate masks for all craters in one image
    Returns combined binary mask (255=crater, 0=background)
    """
    h, w = image_rgb.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    if not boxes_pixel:
        return combined_mask

    # Set image in SAM (done once per image)
    predictor.set_image(image_rgb)

    for box in boxes_pixel:
        box_np = np.array(box)  # [x1, y1, x2, y2]

        # Skip tiny boxes (noise)
        if (box[2]-box[0]) < 5 or (box[3]-box[1]) < 5:
            continue

        # SAM prediction with bounding box prompt
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_np[None, :],   # SAM expects (1,4)
            multimask_output=False  # single best mask
        )

        # Take best mask (index 0 when multimask=False)
        mask = masks[0].astype(np.uint8) * 255
        combined_mask = np.maximum(combined_mask, mask)

    return combined_mask

def process_split(predictor, split_name, img_dir, lbl_dir, out_base):
    """Process one split (train/val/test)"""
    out_img = os.path.join(out_base, split_name, 'images')
    out_msk = os.path.join(out_base, split_name, 'masks')
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_msk, exist_ok=True)

    img_files = [f for f in os.listdir(img_dir)
                 if f.lower().endswith(('.jpg','.jpeg','.png'))]

    stats = {'processed': 0, 'skipped': 0, 'total_craters': 0}

    for fname in tqdm(img_files, desc=f"  {split_name:5s}", ncols=70):
        stem     = fname.rsplit('.', 1)[0]
        img_path = os.path.join(img_dir, fname)
        lbl_path = os.path.join(lbl_dir, stem + '.txt')

        # Load image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            stats['skipped'] += 1
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w    = img_rgb.shape[:2]

        # Read YOLO boxes
        boxes_norm = read_yolo_labels(lbl_path)
        boxes_px   = [yolo_to_pixel_box(xc,yc,bw,bh,w,h)
                      for _,xc,yc,bw,bh in boxes_norm]
        stats['total_craters'] += len(boxes_px)

        # Generate SAM masks
        crater_mask = generate_crater_mask_sam(predictor, img_rgb, boxes_px)

        # Resize both to target size
        img_out  = cv2.resize(img_bgr,     (IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        mask_out = cv2.resize(crater_mask, (IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        # Save
        cv2.imwrite(os.path.join(out_img, stem+'.png'), img_out)
        cv2.imwrite(os.path.join(out_msk, stem+'_mask.png'), mask_out)
        stats['processed'] += 1

    return stats

def save_sam_verification(out_base, n=6):
    """Save visual verification of SAM outputs"""
    img_dir = os.path.join(out_base, 'train', 'images')
    msk_dir = os.path.join(out_base, 'train', 'masks')

    imgs = [f for f in os.listdir(img_dir) if f.endswith('.png')][:n]

    fig, axes = plt.subplots(2, n, figsize=(n*3, 6))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('SAM Crater Mask Verification\nTop: Images  |  Bottom: SAM Masks',
                 color='white', fontsize=11)

    for i, fname in enumerate(imgs):
        img  = cv2.imread(os.path.join(img_dir, fname))
        mask = cv2.imread(os.path.join(msk_dir,
                fname.replace('.png','_mask.png')),
                cv2.IMREAD_GRAYSCALE)

        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Overlay mask on image
            overlay = img_rgb.copy()
            if mask is not None:
                overlay[mask > 127] = [255, 140, 0]
            axes[0][i].imshow(img_rgb)
            axes[0][i].set_title(f'Image {i+1}', color='#f59e0b', fontsize=8)

        if mask is not None:
            axes[1][i].imshow(mask, cmap='hot')
            axes[1][i].set_title(f'Crater Mask {i+1}', color='#f59e0b', fontsize=8)

    for ax in axes.flat:
        ax.axis('off')
        ax.set_facecolor('#0d1117')

    plt.tight_layout()
    out_path = r"D:\Lunar\sam_verification.png"
    plt.savefig(out_path, dpi=120, facecolor='#0d1117', bbox_inches='tight')
    print(f"\n  SAM verification saved → {out_path}")
    plt.close()

# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 58)
    print("  SAM CRATER MASK GENERATOR")
    print("=" * 58)
    print(f"\n  Device: {DEVICE.upper()}")
    print(f"  Output: {OUT_DIR}\n")

    # Step 1: Check checkpoint
    print("Step 1: Checking SAM checkpoint...")
    if not check_sam_checkpoint():
        print("\n  Run the download command above, then re-run this script.")
        exit(1)

    # Step 2: Load SAM
    print("\nStep 2: Loading SAM model...")
    predictor = load_sam()

    # Step 3: Process all splits
    print("\nStep 3: Generating crater masks with SAM...")
    print(f"  (This will take ~20-40 min for all images on RTX 4050)\n")

    total_stats = {'processed': 0, 'skipped': 0, 'total_craters': 0}

    for split_name, (img_dir, lbl_dir) in SPLITS.items():
        if not os.path.exists(img_dir):
            print(f"  ⚠️  {split_name} images not found, skipping")
            continue
        stats = process_split(predictor, split_name, img_dir, lbl_dir, OUT_DIR)
        print(f"    Processed: {stats['processed']} | "
              f"Skipped: {stats['skipped']} | "
              f"Craters: {stats['total_craters']}")
        for k in total_stats:
            total_stats[k] += stats[k]

    # Step 4: Summary
    print("\n" + "=" * 58)
    print("  SAM GENERATION COMPLETE")
    print("=" * 58)
    print(f"  Total processed  : {total_stats['processed']}")
    print(f"  Total skipped    : {total_stats['skipped']}")
    print(f"  Total craters    : {total_stats['total_craters']}")
    print(f"  Output at        : {OUT_DIR}")

    # Step 5: Visual verification
    print("\nStep 4: Saving verification samples...")
    try:
        save_sam_verification(OUT_DIR)
    except Exception as e:
        print(f"  (Skipped: {e})")

    print("\n  ✅ All crater masks generated!")
    print("  Next: python train.py")
    print("=" * 58)
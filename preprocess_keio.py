"""
preprocess_keio.py
Run: python preprocess_keio.py

Converts Keio color masks → class ID masks
Resizes to 256x256
Splits into train/val/test
Saves to D:\Lunar\processed\keio\
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────
KEIO_RENDER = r"D:\Lunar\datasets\keio\images\render"
KEIO_GROUND = r"D:\Lunar\datasets\keio\images\ground"
OUT_DIR     = r"D:\Lunar\processed\keio"
IMG_SIZE    = 256          # resize to 256×256
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
SEED        = 42

# ── CLASS MAPPING ─────────────────────────────────────────────
# Color in ground mask  →  Class ID  →  Meaning
# Red   (255, 0,   0)   →     0      →  Sky
# Green (0,   255, 0)   →     1      →  Rock (small)
# Blue  (0,   0,   255) →     2      →  Boulder (large)
# Black (0,   0,   0)   →     3      →  Ground / Plains

CLASS_NAMES = {0: 'Sky', 1: 'Rock', 2: 'Boulder', 3: 'Ground'}
CLASS_COLORS_VIZ = {
    0: (135, 206, 235),   # light blue  → sky
    1: (34,  197, 94),    # green       → rock
    2: (59,  130, 246),   # blue        → boulder
    3: (71,  85,  105),   # slate       → ground
}

def color_mask_to_class_ids(mask_bgr):
    """
    Convert BGR color mask → single-channel class ID mask
    Returns uint8 array with values 0,1,2,3
    """
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    r = mask_rgb[:, :, 0].astype(np.int16)
    g = mask_rgb[:, :, 1].astype(np.int16)
    b = mask_rgb[:, :, 2].astype(np.int16)

    out = np.full(mask_rgb.shape[:2], 3, dtype=np.uint8)  # default: ground

    # Red → sky (class 0)
    red_mask   = (r > 150) & (g < 80) & (b < 80)
    # Green → rock (class 1)
    green_mask = (g > 150) & (r < 80) & (b < 80)
    # Blue → boulder (class 2)
    blue_mask  = (b > 150) & (r < 80) & (g < 80)
    # Black stays ground (class 3)

    out[red_mask]   = 0
    out[green_mask] = 1
    out[blue_mask]  = 2

    return out

def class_ids_to_rgb(class_mask):
    """Convert class ID mask → RGB for visualization"""
    h, w = class_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS_VIZ.items():
        rgb[class_mask == cls_id] = color
    return rgb

def get_matching_pairs(render_dir, ground_dir):
    """Find render images that have matching ground masks"""
    render_files = {f for f in os.listdir(render_dir)
                    if f.lower().endswith(('.png', '.jpg'))}
    ground_files = {f for f in os.listdir(ground_dir)
                    if f.lower().endswith(('.png', '.jpg'))}

    # Match by filename (render0001.png ↔ ground0001.png)
    matched = []
    for rf in render_files:
        stem = rf.rsplit('.', 1)[0]   # e.g. "render0001"
        # ground file might be "ground0001.png"
        gf_candidate = stem.replace('render', 'ground') + '.' + rf.rsplit('.',1)[1]
        if gf_candidate in ground_files:
            matched.append((rf, gf_candidate))

    return matched

def process_and_save(pairs, render_dir, ground_dir, split_name, out_base):
    """Process image-mask pairs and save to output directory"""
    img_out = os.path.join(out_base, split_name, 'images')
    msk_out = os.path.join(out_base, split_name, 'masks')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(msk_out, exist_ok=True)

    skipped = 0
    class_pixel_counts = np.zeros(4, dtype=np.int64)

    for render_f, ground_f in tqdm(pairs, desc=f"  {split_name:5s}", ncols=70):
        # Load
        img  = cv2.imread(os.path.join(render_dir, render_f))
        mask = cv2.imread(os.path.join(ground_dir, ground_f))

        if img is None or mask is None:
            skipped += 1
            continue

        # Resize
        img  = cv2.resize(img,  (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        # Convert mask colors → class IDs
        class_mask = color_mask_to_class_ids(mask)

        # Track class distribution
        for c in range(4):
            class_pixel_counts[c] += (class_mask == c).sum()

        # Save
        stem = render_f.rsplit('.', 1)[0]
        cv2.imwrite(os.path.join(img_out, stem + '.png'), img)
        cv2.imwrite(os.path.join(msk_out, stem + '_mask.png'), class_mask)

    return skipped, class_pixel_counts

def save_verification_samples(out_base, n=6):
    """Save side-by-side verification of image + mask"""
    train_img = os.path.join(out_base, 'train', 'images')
    train_msk = os.path.join(out_base, 'train', 'masks')

    imgs = [f for f in os.listdir(train_img) if f.endswith('.png')][:n]

    fig, axes = plt.subplots(2, n, figsize=(n*3, 6))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('Keio Preprocessing Verification\nTop: Images  |  Bottom: Class Masks',
                 color='white', fontsize=11)

    for i, fname in enumerate(imgs):
        img  = cv2.imread(os.path.join(train_img, fname))
        mask = cv2.imread(os.path.join(train_msk,
                fname.replace('.png','_mask.png')), cv2.IMREAD_GRAYSCALE)

        if img is not None:
            axes[0][i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0][i].set_title(f'Image {i+1}', color='#10b981', fontsize=8)

        if mask is not None:
            mask_rgb = class_ids_to_rgb(mask)
            axes[1][i].imshow(mask_rgb)
            axes[1][i].set_title(f'Mask {i+1}', color='#3b82f6', fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend = [Patch(color=np.array(c)/255, label=CLASS_NAMES[i])
              for i, c in CLASS_COLORS_VIZ.items()]
    fig.legend(handles=legend, loc='lower center', ncol=4,
               facecolor='#111827', labelcolor='white', fontsize=9)

    for ax in axes.flat:
        ax.axis('off')
        ax.set_facecolor('#0d1117')

    plt.tight_layout()
    out_path = r"D:\Lunar\keio_preprocessing_check.png"
    plt.savefig(out_path, dpi=120, facecolor='#0d1117', bbox_inches='tight')
    print(f"\n  Verification image saved → {out_path}")
    plt.close()

# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 58)
    print("  KEIO DATASET PREPROCESSING")
    print("=" * 58)
    print(f"\n  Input  render : {KEIO_RENDER}")
    print(f"  Input  ground : {KEIO_GROUND}")
    print(f"  Output        : {OUT_DIR}")
    print(f"  Target size   : {IMG_SIZE}×{IMG_SIZE}")
    print(f"  Split         : {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}\n")

    # Step 1: Find matching pairs
    print("Step 1: Finding render ↔ ground pairs...")
    pairs = get_matching_pairs(KEIO_RENDER, KEIO_GROUND)
    print(f"  Found {len(pairs)} matching pairs")

    if len(pairs) == 0:
        print("\n  ❌ No matching pairs found!")
        print("  Check that render/ and ground/ filenames match")
        print("  e.g. render0001.png ↔ ground0001.png")
        exit(1)

    # Step 2: Shuffle and split
    print("\nStep 2: Splitting dataset...")
    random.seed(SEED)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:n_train+n_val]
    test_pairs  = pairs[n_train+n_val:]

    print(f"  Train : {len(train_pairs)}")
    print(f"  Val   : {len(val_pairs)}")
    print(f"  Test  : {len(test_pairs)}")

    # Step 3: Process and save
    print("\nStep 3: Processing images and masks...")

    all_counts = np.zeros(4, dtype=np.int64)
    for split_name, split_pairs in [
        ('train', train_pairs),
        ('val',   val_pairs),
        ('test',  test_pairs)
    ]:
        skipped, counts = process_and_save(
            split_pairs, KEIO_RENDER, KEIO_GROUND, split_name, OUT_DIR)
        all_counts += counts
        if skipped:
            print(f"    ⚠️  Skipped {skipped} corrupted files in {split_name}")

    # Step 4: Class distribution
    print("\nStep 4: Class distribution in training set:")
    total_px = all_counts.sum() or 1
    for c in range(4):
        pct = all_counts[c] / total_px * 100
        bar = '█' * int(pct / 2)
        print(f"  Class {c} ({CLASS_NAMES[c]:8s}): {pct:5.1f}%  {bar}")

    # Step 5: Verify
    print("\nStep 5: Verifying output...")
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(OUT_DIR, split, 'images')
        msk_dir = os.path.join(OUT_DIR, split, 'masks')
        n_imgs  = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
        n_msks  = len(os.listdir(msk_dir)) if os.path.exists(msk_dir) else 0
        ok = "✅" if n_imgs == n_msks and n_imgs > 0 else "❌"
        print(f"  {ok} {split:5s}: {n_imgs} images, {n_msks} masks")

    # Step 6: Save visual check
    print("\nStep 6: Saving verification samples...")
    try:
        save_verification_samples(OUT_DIR)
    except Exception as e:
        print(f"  (Skipped: {e})")

    print("\n" + "=" * 58)
    print("  ✅ KEIO PREPROCESSING COMPLETE")
    print(f"  Output at: {OUT_DIR}")
    print("  Next: python sam_crater_masks.py")
    print("=" * 58)
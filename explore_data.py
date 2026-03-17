"""
explore_data.py
Run: python explore_data.py
Verifies both datasets are ready before training
"""

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── PATHS ─────────────────────────────────────────────────────
KEIO_RENDER  = r"D:\Lunar\datasets\keio\images\render"
KEIO_GROUND  = r"D:\Lunar\datasets\keio\images\ground"
KEIO_CLEAN   = r"D:\Lunar\datasets\keio\images\clean"

CRATER_TRAIN_IMG = r"D:\Lunar\datasets\craters\craters\train\images"
CRATER_TRAIN_LBL = r"D:\Lunar\datasets\craters\craters\train\labels"
CRATER_VAL_IMG   = r"D:\Lunar\datasets\craters\craters\valid\images"
CRATER_TEST_IMG  = r"D:\Lunar\datasets\craters\craters\test\images"

# ── CLASS MAP for Keio ground masks ───────────────────────────
# Red=(255,0,0)   → Sky     → class 0
# Green=(0,255,0) → Rock    → class 1
# Blue=(0,0,255)  → Boulder → class 2
# Black=(0,0,0)   → Ground  → class 3

def count_images(folder, exts=('.jpg','.jpeg','.png')):
    if not os.path.exists(folder):
        return 0, []
    files = [f for f in os.listdir(folder)
             if f.lower().endswith(exts)]
    return len(files), files

def check_image_size(folder, sample=5):
    _, files = count_images(folder)
    if not files:
        return None
    sizes = []
    for f in files[:sample]:
        img = cv2.imread(os.path.join(folder, f))
        if img is not None:
            sizes.append(img.shape[:2])
    return sizes

def analyze_mask_colors(ground_folder, sample=10):
    """Check what colors are actually in the Keio ground masks"""
    _, files = count_images(ground_folder)
    color_counts = {'red':0, 'green':0, 'blue':0, 'black':0, 'other':0}

    for f in files[:sample]:
        img = cv2.imread(os.path.join(ground_folder, f))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]

        red_mask   = (r > 150) & (g < 80) & (b < 80)
        green_mask = (g > 150) & (r < 80) & (b < 80)
        blue_mask  = (b > 150) & (r < 80) & (g < 80)
        black_mask = (r < 30)  & (g < 30) & (b < 30)
        other_mask = ~(red_mask | green_mask | blue_mask | black_mask)

        color_counts['red']   += red_mask.sum()
        color_counts['green'] += green_mask.sum()
        color_counts['blue']  += blue_mask.sum()
        color_counts['black'] += black_mask.sum()
        color_counts['other'] += other_mask.sum()

    total = sum(color_counts.values()) or 1
    return {k: f"{v/total*100:.1f}%" for k, v in color_counts.items()}

def count_craters_in_labels(label_folder, sample=20):
    """Count average craters per image"""
    if not os.path.exists(label_folder):
        return 0
    files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]
    total = 0
    checked = 0
    for f in files[:sample]:
        with open(os.path.join(label_folder, f)) as fp:
            lines = [l.strip() for l in fp if l.strip()]
            total += len(lines)
            checked += 1
    return total / checked if checked else 0

def save_sample_grid():
    """Save a visual sample of both datasets"""
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('Dataset Samples', color='white', fontsize=14, fontweight='bold')

    # Row 1: Keio render + ground pairs
    _, render_files = count_images(KEIO_RENDER)
    _, ground_files = count_images(KEIO_GROUND)

    for i in range(3):
        if i < len(render_files):
            img = cv2.imread(os.path.join(KEIO_RENDER, render_files[i*100]))
            if img is not None:
                axes[0][i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axes[0][i].set_title(f'Keio Render {i+1}', color='#10b981', fontsize=8)
        if i < len(ground_files):
            msk = cv2.imread(os.path.join(KEIO_GROUND, ground_files[i*100]))
            if msk is not None:
                axes[0][i+3].imshow(cv2.cvtColor(msk, cv2.COLOR_BGR2RGB))
                axes[0][i+3].set_title(f'Keio Mask {i+1}', color='#10b981', fontsize=8)

    # Row 2: Crater images with bbox overlay
    _, crater_files = count_images(CRATER_TRAIN_IMG)
    for i in range(min(6, len(crater_files))):
        img_path = os.path.join(CRATER_TRAIN_IMG, crater_files[i*5])
        lbl_path = os.path.join(CRATER_TRAIN_LBL,
                    crater_files[i*5].rsplit('.',1)[0] + '.txt')
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        if os.path.exists(lbl_path):
            with open(lbl_path) as fp:
                for line in fp:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        xc,yc,bw,bh = [float(x) for x in parts[1:5]]
                        x1 = int((xc-bw/2)*w); y1 = int((yc-bh/2)*h)
                        x2 = int((xc+bw/2)*w); y2 = int((yc+bh/2)*h)
                        cv2.rectangle(img_rgb,(x1,y1),(x2,y2),(255,165,0),2)

        axes[1][i].imshow(img_rgb)
        axes[1][i].set_title(f'Crater {i+1}', color='#f59e0b', fontsize=8)

    for ax in axes.flat:
        ax.axis('off')
        ax.set_facecolor('#0d1117')

    plt.tight_layout()
    out = r"D:\Lunar\dataset_samples.png"
    plt.savefig(out, dpi=120, facecolor='#0d1117', bbox_inches='tight')
    print(f"\nSample grid saved → {out}")
    plt.close()

# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 58)
    print("  LUNAR PROJECT — DATASET EXPLORER")
    print("=" * 58)

    # ── KEIO ──────────────────────────────────────────────────
    print("\n[DATASET A] Keio Synthetic Lunar Dataset")
    print("-" * 40)

    n_render, _ = count_images(KEIO_RENDER)
    n_ground, _ = count_images(KEIO_GROUND)
    n_clean,  _ = count_images(KEIO_CLEAN)

    print(f"  render/ images  : {n_render}")
    print(f"  ground/ masks   : {n_ground}")
    print(f"  clean/  masks   : {n_clean}")

    sizes = check_image_size(KEIO_RENDER)
    if sizes:
        print(f"  Image size      : {sizes[0][1]}x{sizes[0][0]} (HxW)")

    print("\n  Analyzing mask colors (sample of 10)...")
    colors = analyze_mask_colors(KEIO_GROUND)
    print(f"  Red   (Sky)     : {colors['red']}")
    print(f"  Green (Rock)    : {colors['green']}")
    print(f"  Blue  (Boulder) : {colors['blue']}")
    print(f"  Black (Ground)  : {colors['black']}")
    print(f"  Other           : {colors['other']}")

    if n_render == n_ground:
        print(f"\n  ✅ render/ground counts MATCH ({n_render} pairs)")
    else:
        print(f"\n  ⚠️  MISMATCH: {n_render} renders vs {n_ground} masks")

    # ── CRATERS ───────────────────────────────────────────────
    print("\n[DATASET B] LincolnZH Crater Dataset")
    print("-" * 40)

    n_train, _ = count_images(CRATER_TRAIN_IMG)
    n_val,   _ = count_images(CRATER_VAL_IMG)
    n_test,  _ = count_images(CRATER_TEST_IMG)
    total_c = n_train + n_val + n_test

    print(f"  train images    : {n_train}")
    print(f"  valid images    : {n_val}")
    print(f"  test  images    : {n_test}")
    print(f"  TOTAL           : {total_c}")

    avg_craters = count_craters_in_labels(CRATER_TRAIN_LBL)
    print(f"  Avg craters/img : {avg_craters:.1f}")
    print(f"  Est. total bbox : ~{int(avg_craters * total_c)}")

    sizes_c = check_image_size(CRATER_TRAIN_IMG)
    if sizes_c:
        print(f"  Image size      : {sizes_c[0][1]}x{sizes_c[0][0]} (HxW)")

    # ── SUMMARY ───────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  SUMMARY")
    print("=" * 58)
    print(f"  Keio images     : {n_render:,}")
    print(f"  Crater images   : {total_c:,}")
    print(f"  Total dataset   : {n_render + total_c:,} images")
    print()

    all_ok = True
    checks = [
        (n_render > 5000,   f"Keio render images > 5000  ({n_render})"),
        (n_ground > 5000,   f"Keio ground masks  > 5000  ({n_ground})"),
        (n_render==n_ground,f"render/ground counts match"),
        (n_train  > 100,    f"Crater train images > 100  ({n_train})"),
    ]
    for ok, msg in checks:
        icon = "✅" if ok else "❌"
        print(f"  {icon}  {msg}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("  ✅ ALL CHECKS PASSED — Ready for preprocessing!")
        print("  Next: python preprocess_keio.py")
    else:
        print("  ⚠️  Fix issues above before proceeding")

    print("\nGenerating visual sample grid...")
    try:
        save_sample_grid()
    except Exception as e:
        print(f"  (Sample grid skipped: {e})")
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class LunarDataset(Dataset):
    def __init__(self, base_path, img_size=(256, 256)):
        self.img_size = img_size
        self.image_dir = os.path.join(base_path, 'keio', 'images', 'render')
        self.mask_dir = os.path.join(base_path, 'keio', 'images', 'ground')
        
        # 1. Load the IDs we need to exclude
        self.blacklist = self._get_blacklist(os.path.join(base_path, 'keio'))

        # 2. Collect all valid image filenames
        all_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        self.images = [f for f in all_files if f[6:10] not in self.blacklist]

        print(f"✅ Dataset initialized: {len(self.images)} valid images found.")

    def _get_blacklist(self, path):
        blacklist_files = [
            'mismatch_IDs.txt', 'cam_anomaly_IDs.txt',
            'shadow_IDs.txt', 'ground_facing_IDs.txt', 'top200_largerocks_IDs.txt'
        ]
        bad_ids = set()
        for filename in blacklist_files:
            file_path = os.path.join(path, filename)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    ids = [line.strip() for line in f.readlines() if line.strip()]
                    bad_ids.update(ids)
        return bad_ids

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load Image
        img_path = os.path.join(self.image_dir, self.images[index])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)

        # Load Mask
        mask_name = self.images[index].replace("render", "ground")
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        # Convert RGB to 3-class mask: 0=bg, 1=rocks, 2=crater (in future)
        label_mask = np.zeros(self.img_size, dtype=np.int64)

        red = (mask[:,:,0] > 128) & (mask[:,:,1] < 50) & (mask[:,:,2] < 50)
        green = (mask[:,:,1] > 128) & (mask[:,:,0] < 50) & (mask[:,:,2] < 50)
        blue = (mask[:,:,2] > 128) & (mask[:,:,0] < 50) & (mask[:,:,1] < 50)

        label_mask[red] = 1   # Sky
        label_mask[green] = 2 # Small rocks
        label_mask[blue] = 3  # Large rocks

        # Normalize image to [0,1] and convert to torch
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # (C, H, W)

        return torch.tensor(image), torch.tensor(label_mask)

# Test
if __name__ == "__main__":
    ds = LunarDataset(r"D:\Lunar")
    img, mask = ds[0]
    print(f"Sample image shape: {img.shape}")
    print(f"Sample mask shape : {mask.shape}")
    print(f"Unique labels in mask: {torch.unique(mask)}")
import matplotlib.pyplot as plt

def visualize(img, mask):
    img = img.permute(1, 2, 0).numpy()  # [H,W,3]
    mask_np = mask.numpy()

    color_map = {
        0: [0, 0, 0],       # Background: Black
        1: [255, 0, 0],     # Sky: Red
        2: [0, 255, 0],     # Small Rocks: Green
        3: [0, 0, 255],     # Large Rocks: Blue
    }

    overlay = np.zeros_like(img)
    for cls, color in color_map.items():
        overlay[mask_np == cls] = color

    overlay = overlay.astype(np.uint8)

    blended = (0.6 * img * 255 + 0.4 * overlay).astype(np.uint8)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(img); plt.title('Image'); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(blended); plt.title('Overlay'); plt.axis('off')
    plt.tight_layout(); plt.show()

# Call the function
visualize(img, mask)

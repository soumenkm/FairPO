from datasets import load_dataset
import torch, tqdm, os, sys, pickle, datetime, logging, random
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, List, Dict, Optional
from torchvision import transforms
torch.manual_seed(42)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChestXRayDataset(Dataset):
    def __init__(self, frac: float, is_train: bool, max_pixels: int = 89478485) -> None:
        super(ChestXRayDataset, self).__init__()
        self.is_train = is_train
        self.frac = frac
        self.max_pixels = max_pixels
        self.transform = self._get_transform()
        self.ds, self.label_names = self._get_dataset()
        
    def _get_transform(self) -> transforms:
        return transforms.Compose([
            transforms.Resize(256),              
            transforms.CenterCrop(224),          
            transforms.ToTensor(),               
            transforms.Normalize(                
                mean=[0.485],      
                std=[0.229]     
            )
        ]) # TODO: Currently it is ImageNet stat, add ChestXRay stat
        
    def _get_dataset(self) -> Tuple[Dataset, List[str]]:
        key = "train" if self.is_train else "test"
        ds = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", "image-classification")[key]
        length = int(len(ds) * self.frac)
        ds = ds.select(range(length)).shuffle(seed=42)
        size = int(len(ds))
        final_ds = []
        label_names = ds.features['labels'].feature.names
        
        with tqdm.tqdm(range(size), desc=f"Preparing {key} dataset...", colour="green", unit="examples") as pbar:
            for i in pbar:
                image = ds[i]["image"].convert()
                if image.size[0] * image.size[1] <= self.max_pixels:
                    image = self.transform(image)
                    label_indices = ds[i]["labels"]
                    multi_hot_label = torch.zeros(len(label_names), dtype=torch.int64)
                    multi_hot_label[label_indices] = 1
                    final_ds.append({"pixels": image, "labels": multi_hot_label})
                else:
                    print(f"Skipping image {i} due to large size.")
        return final_ds, label_names
    
    def __len__(self) -> int:
        return len(self.ds)
    
    def __getitem__(self, index: int) -> dict:
        return self.ds[index]

class COCODataset(Dataset):
    def __init__(self, root_dir: str, frac: float, is_train: bool) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.is_train = is_train
        self.frac = frac
        self.max_pixels = 89478485
        self.seed = 42
        self.transform = self._get_transform()
        self.split_name = "train2014" if self.is_train else "val2014"
        data = self._load_data()
        self.data, self.label_names = data["data"], data["label_names"]

        if not self.data:
            logging.warning(f"No data loaded for the {self.split_name} split. Check paths and dataset integrity.")


    def _get_transform(self) -> transforms.Compose:
        # Standard ViT preprocessing often uses ImageNet stats
        # Ensure input images are RGB
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), # Higher quality interpolation
            transforms.CenterCrop(224),
            transforms.ToTensor(), # Converts PIL image (H, W, C) [0, 255] to Tensor (C, H, W) [0.0, 1.0]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], # ImageNet mean for RGB
                std=[0.229, 0.224, 0.225]   # ImageNet std for RGB
            ) # TODO: Add COCO stats
        ])

    def _load_data(self) -> dict:
        split = self.split_name
        img_dir = os.path.join(self.root_dir, "images", split)
        label_dir = os.path.join(self.root_dir, "labels", split)
        names_file = os.path.join(self.root_dir, "coco.names")

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not os.path.isdir(label_dir):
             raise FileNotFoundError(f"Label directory not found: {label_dir}")
        if not os.path.isfile(names_file):
            raise FileNotFoundError(f"Class names file not found: {names_file}")

        # 1. Load class names
        with open(names_file, 'r') as f:
            label_names = [line.strip() for line in f.readlines() if line.strip()]
        num_classes = len(label_names)
        logging.info(f"Loaded {num_classes} class names from {names_file}")

        # 2. Find all image files
        all_image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        logging.info(f"Found {len(all_image_files)} potential image files in {img_dir}.")
        
        random.seed(self.seed)
        random.shuffle(all_image_files)
        num_samples_to_keep = int(len(all_image_files) * self.frac)
        all_image_files = all_image_files[:num_samples_to_keep]
        
        processed_data = []
        skipped_count = 0

        # 3. Process each image and its corresponding label file
        with tqdm.tqdm(all_image_files, desc=f"Preparing {split} dataset", colour="green", unit="images") as pbar:
            for img_filename in pbar:
                img_path = os.path.join(img_dir, img_filename)
                # Derive label filename from image filename (e.g., img.jpg -> img.txt)
                base_name, _ = os.path.splitext(img_filename)
                label_filename = base_name + ".txt"
                label_path = os.path.join(label_dir, label_filename)

                try:
                    # Load image using PIL
                    with Image.open(img_path) as img:
                        img = img.convert("RGB") # Ensure image is RGB

                        # Optional: Check image size before loading full data
                        if img.width * img.height > self.max_pixels:
                            skipped_count += 1
                            continue

                        # Find unique class IDs present in the label file
                        unique_class_ids = set()
                        if os.path.exists(label_path):
                            with open(label_path, 'r') as lf:
                                for line in lf:
                                    parts = line.strip().split()
                                    if len(parts) >= 1:
                                        try:
                                            class_id = int(parts[0])
                                            if 0 <= class_id < num_classes:
                                                unique_class_ids.add(class_id)
                                            else:
                                                logging.warning(f"Invalid class ID {class_id} in {label_filename}. Skipping ID.")
                                        except ValueError:
                                            logging.warning(f"Non-integer class ID found in first column of {label_filename}. Line: '{line.strip()}'")
                        else: # Image might legitimately have no objects/labels
                            logging.debug(f"Label file not found for {img_filename}, assuming no labeled objects.")

                        # Apply transformations
                        transformed_image = self.transform(img)

                        # Create multi-hot label vector
                        # Use float32, common for loss functions like BCEWithLogitsLoss
                        multi_hot_label = torch.zeros(num_classes, dtype=torch.float32)
                        if unique_class_ids:
                             # Convert set to list for indexing
                            indices = list(unique_class_ids)
                            multi_hot_label[indices] = 1.0

                        processed_data.append({"pixels": transformed_image, "labels": multi_hot_label})

                except FileNotFoundError:
                     logging.warning(f"Image file listed but not found: {img_path}. Skipping.")
                     skipped_count += 1
                except Exception as e:
                    logging.error(f"Error processing image {img_filename}: {e}. Skipping.")
                    skipped_count += 1

        if skipped_count > 0:
             logging.info(f"Skipped {skipped_count} images due to errors or filters.")

        logging.info(f"Successfully processed {len(processed_data)} samples for {split} split (using {self.frac*100:.1f}% of available data).")

        final_data = {"data": processed_data, "label_names": label_names}
        logging.info(f"Cached processed data to data/coco_{split}.pkl")
        return final_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.data[index]

    def get_label_names(self) -> List[str]:
        return self.label_names

if __name__ == '__main__':
    coco_root = "/raid/speech/soumen/.cache/kagglehub/datasets/jeffaudi/coco-2014-dataset-for-yolov3/versions/4/coco2014" # Adjust this path
    ds = COCODataset(root_dir=coco_root, frac=0.01, is_train=True)
    print(f"Number of samples in dataset: {len(ds)}")
    print(ds[0])



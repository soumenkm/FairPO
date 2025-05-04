from datasets import load_dataset
import torch, tqdm, os, sys, pickle, datetime, logging, random, csv, logging
import pandas as pd
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, List, Dict, Optional
from torchvision import transforms
torch.manual_seed(42)
from tqdm import tqdm

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
    def __init__(self, root_dir: str, frac: float, is_train: bool, privileged_indices_set: set) -> None:
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
        all_labels = list(range(len(self.label_names)))
        non_privileged_indices_set = set(all_labels) - privileged_indices_set
        self.privileged_indices = sorted(list(privileged_indices_set))
        self.non_privileged_indices = sorted(list(non_privileged_indices_set))
        
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
        return final_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.data[index]

    def get_label_names(self) -> List[str]:
        return self.label_names

class COCODatasetOnDemand(Dataset):
    def __init__(
        self,
        root_dir: str,
        split_name: str,
        frac: float = 1.0,
        privileged_indices_set: set = None,
        seed: int = 42,
        index_dir: Optional[str] = None, # Optional dir for index files
        force_regenerate: bool = False   # Flag to force index regeneration
    ) -> None:
        
        super().__init__()
        self.root_dir = Path(root_dir) # Use pathlib for easier path handling
        self.frac = max(0.0, min(1.0, frac))
        self.seed = seed
        self.transform = self._get_transform()
        if split_name == "train":
            self.split_name = "train2014" 
        elif split_name == "val":
            self.split_name = "val2014"
        elif split_name == "test":
            self.split_name = "test2014"
        else:
            raise ValueError(f"Invalid split name: {split_name}. Use 'train', 'val', or 'test'.")

        # Handle privileged indices
        if privileged_indices_set is None:
            privileged_indices_set = set()

        # Determine index directory
        if index_dir is None:
            self.index_dir = self.root_dir / ".index_cache"
        else:
            self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True) # Ensure index directory exists

        # Load class names first (needed for num_classes)
        self.label_names = self._load_label_names()
        self.num_classes = len(self.label_names)

        # Determine privileged/non-privileged based on loaded names
        all_labels_set = set(range(self.num_classes))
        
        # Ensure provided privileged indices are valid
        valid_privileged_set = {idx for idx in privileged_indices_set if 0 <= idx < self.num_classes}
        if len(valid_privileged_set) != len(privileged_indices_set):
            logging.warning("Some provided privileged indices were out of bounds and ignored.")

        non_privileged_indices_set = all_labels_set - valid_privileged_set
        self.privileged_indices = sorted(list(valid_privileged_set))
        self.non_privileged_indices = sorted(list(non_privileged_indices_set))
        logging.info(f"Privileged Indices: {self.privileged_indices}")
        logging.info(f"Non-Privileged Indices: {self.non_privileged_indices}")

        # Load or generate index
        self.index_file_path = self._get_index_filepath()
        self.index = self._load_or_generate_index(force_regenerate)

        if not self.index:
             logging.warning(f"No index data loaded for the {self.split_name} split. Check paths and dataset integrity.")

    def _get_transform(self) -> transforms.Compose:
        # (Same as before)
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_label_names(self) -> List[str]:
        """Loads class names from the coco.names file."""
        names_file = self.root_dir / "coco.names"
        if not names_file.is_file():
            raise FileNotFoundError(f"Class names file not found: {names_file}")
        try:
            with open(names_file, 'r') as f:
                label_names = [line.strip() for line in f.readlines() if line.strip()]
            if not label_names:
                raise ValueError(f"Class names file is empty: {names_file}")
            logging.info(f"Loaded {len(label_names)} class names from {names_file}")
            return label_names
        except Exception as e:
            logging.error(f"Error reading class names file {names_file}: {e}")
            raise

    def _get_index_filepath(self) -> Path:
        """Generates the path for the index CSV file."""
        frac_str = f"{self.frac:.4f}".replace('.', 'p')
        filename = f"coco_idx_{self.split_name}_frac{frac_str}_seed{self.seed}.csv"
        return self.index_dir / filename

    def _generate_index(self) -> List[Tuple[str, str]]:
        """Scans directories and creates the list of (image_path, label_path) pairs."""
        logging.info(f"Generating index file: {self.index_file_path}...")
        img_dir = self.root_dir / "images" / self.split_name
        label_dir = self.root_dir / "labels" / self.split_name

        if not img_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not label_dir.is_dir():
             # Allow label dir to be missing, maybe some images have no labels
             logging.warning(f"Label directory not found: {label_dir}. Images may not have corresponding labels.")

        image_files = [f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
        logging.info(f"Found {len(image_files)} potential image files in {img_dir}.")

        index_data = []
        missing_labels = 0
        with tqdm.tqdm(image_files, desc=f"Scanning for {self.split_name} files", colour="blue", unit="images") as pbar:
            for img_path in pbar:
                base_name = img_path.stem # Get filename without extension
                label_filename = base_name + ".txt"
                label_path = label_dir / label_filename

                # Store paths as strings for CSV compatibility
                img_path_str = str(img_path.resolve())
                label_path_str = str(label_path.resolve())

                if not label_path.exists():
                    missing_labels += 1
                    # Decide whether to include images with missing labels
                    # Option 1: Include them (label path will be checked in __getitem__)
                    index_data.append((img_path_str, label_path_str))
                    # Option 2: Skip them
                    # continue

                else:
                    index_data.append((img_path_str, label_path_str))

        if missing_labels > 0:
            logging.warning(f"{missing_labels} images did not have a corresponding label file in {label_dir}.")

        # Apply shuffling and fraction *before* saving
        random.seed(self.seed)
        random.shuffle(index_data)
        num_samples_to_keep = int(len(index_data) * self.frac)
        final_index_data = index_data[:num_samples_to_keep]

        logging.info(f"Selected {len(final_index_data)} samples based on frac={self.frac}, seed={self.seed}.")

        # Save to CSV
        try:
            with open(self.index_file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['image_path', 'label_path']) # Header
                writer.writerows(final_index_data)
            logging.info(f"Successfully saved index to {self.index_file_path}")
        except IOError as e:
            logging.error(f"Failed to write index file {self.index_file_path}: {e}")
            # Decide how to handle write failure - maybe return empty list?
            return []

        return final_index_data

    def _load_or_generate_index(self, force_regenerate: bool) -> List[Tuple[str, str]]:
        """Loads index from CSV or generates it if needed."""
        if not force_regenerate and self.index_file_path.exists():
            logging.info(f"Loading existing index file: {self.index_file_path}")
            index_data = []
            try:
                with open(self.index_file_path, 'r', newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    header = next(reader) # Skip header
                    if header != ['image_path', 'label_path']:
                        logging.warning(f"Index file {self.index_file_path} has unexpected header: {header}. Regenerating.")
                        return self._generate_index()
                    for row in reader:
                        if len(row) == 2:
                            index_data.append((row[0], row[1]))
                        else:
                            logging.warning(f"Skipping malformed row in {self.index_file_path}: {row}")
                logging.info(f"Loaded {len(index_data)} entries from index.")
                return index_data
            except (IOError, csv.Error, StopIteration) as e:
                 logging.warning(f"Failed to load or parse index file {self.index_file_path}: {e}. Regenerating.")
                 return self._generate_index()
            except Exception as e:
                 logging.error(f"Unexpected error loading index file {self.index_file_path}: {e}. Regenerating.")
                 return self._generate_index()

        else:
            if force_regenerate:
                logging.info("Forcing index regeneration.")
            else:
                logging.info("Index file not found.")
            return self._generate_index()

    def __len__(self) -> int:
        """Returns the number of samples described in the index."""
        return len(self.index)

    def __getitem__(self, index: int) -> Optional[Dict[str, torch.Tensor]]:
        """Loads and processes a single sample on demand."""
        if index >= len(self.index):
            raise IndexError(f"Index {index} out of bounds for dataset with length {len(self.index)}")

        img_path_str, label_path_str = self.index[index]

        try:
            # 1. Load Image
            img = Image.open(img_path_str).convert("RGB")

            # 2. Load Labels
            unique_class_ids = set()
            label_path = Path(label_path_str)
            if label_path.exists():
                with open(label_path, 'r') as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            try:
                                class_id = int(parts[0])
                                if 0 <= class_id < self.num_classes:
                                    unique_class_ids.add(class_id)
                                # else: Optional: Log invalid class IDs found during getitem
                            except ValueError:
                                # Optional: Log lines with non-integer class IDs
                                pass
            # else: Image legitimately might not have a label file

            # 3. Apply Transformations
            transformed_image = self.transform(img)

            # 4. Create multi-hot label vector
            multi_hot_label = torch.zeros(self.num_classes, dtype=torch.float32)
            if unique_class_ids:
                indices = list(unique_class_ids)
                multi_hot_label[indices] = 1.0

            return {"pixels": transformed_image, "labels": multi_hot_label}

        except FileNotFoundError:
             logging.error(f"ERROR in __getitem__: Image file not found at {img_path_str} (listed in index). Skipping sample {index}.")
             # Option 1: Return None (requires collate_fn to handle Nones)
             # return None
             # Option 2: Re-raise or raise custom error (might stop training)
             # raise FileNotFoundError(f"Image file missing: {img_path_str}")
             # Option 3: Return the *next* valid item (complex, breaks standard Dataset)
             # For now, we'll log error and let it potentially cause issues downstream if None isn't handled
             return None # Requires a collate_fn that filters Nones
        except (UnidentifiedImageError, IOError) as e:
             logging.error(f"ERROR in __getitem__: Failed to load/read image {img_path_str}: {e}. Skipping sample {index}.")
             return None # Requires a collate_fn that filters Nones
        except Exception as e:
            logging.error(f"ERROR in __getitem__: Unexpected error processing sample {index} ({img_path_str}): {e}")
            # Optionally re-raise the exception for debugging
            # raise e
            return None # Requires a collate_fn that filters Nones

    def get_label_names(self) -> List[str]:
        """Returns the list of class names."""
        return self.label_names

# --- Example Usage ---

# Define a collate function that handles None values returned by __getitem__
def collate_fn_skip_none(batch):
    """Collate function that filters out None items."""
    batch = [item for item in batch if item is not None]
    if not batch:
        # If all items in the batch were None, return None or an empty structure
        # depending on what the Trainer expects. Returning None might cause issues.
        # It might be better to ensure the dataset doesn't produce too many Nones.
        logging.warning("Entire batch was None after filtering.")
        # Return dummy batch structure to avoid crashing Trainer, although this isn't ideal
        return {'pixels': torch.empty(0), 'labels': torch.empty(0)}
        # return None # This might cause the 'NoneType' error again in Trainer

    # Use default collate on the filtered batch
    return torch.utils.data.dataloader.default_collate(batch)

def get_label_distribution(dataset: COCODatasetOnDemand, batch_size: int = 64, num_workers: int = 4):
    """
    Calculates the distribution of positive labels in a COCODatasetOnDemand.

    Args:
        dataset: An initialized instance of COCODatasetOnDemand.
        batch_size: Batch size for efficient loading.
        num_workers: Number of workers for the DataLoader.

    Returns:
        pandas.DataFrame: DataFrame with columns 'label_id', 'label_name', 'count'.
                          Returns None if dataset is empty or label names are missing.
    """
    label_names = dataset.get_label_names()
    if not label_names:
        logging.error("Dataset does not contain label names.")
        return None

    num_labels = len(label_names)
    if len(dataset) == 0:
        logging.warning("Dataset is empty. Returning empty distribution.")
        return pd.DataFrame({'label_id': [], 'label_name': [], 'count': []})

    # Initialize counts
    label_counts = torch.zeros(num_labels, dtype=torch.int64)

    # Use DataLoader for efficient iteration
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, # Order doesn't matter for counting
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_skip_none # Important to handle potential None items
    )

    logging.info(f"Calculating label distribution for {len(dataset)} samples...")
    for batch in tqdm(loader, desc="Counting Labels", unit="batch"):
        if batch is None or 'labels' not in batch or batch['labels'].numel() == 0:
            continue # Skip empty or invalid batches

        # batch['labels'] shape: (batch_size, num_labels)
        # Sum positive labels across the batch dimension for each label
        batch_label_sum = torch.sum(batch['labels'], dim=0) # Shape: (num_labels)
        label_counts += batch_label_sum.long() # Accumulate counts

    # Create DataFrame
    distribution_df = pd.DataFrame({
        'label_id': list(range(num_labels)),
        'label_name': label_names,
        'count': label_counts.numpy() # Convert tensor to numpy array
    })

    logging.info("Label distribution calculation complete.")
    return distribution_df

if __name__ == '__main__':
    coco_root = "/raid/speech/soumen/.cache/kagglehub/datasets/jeffaudi/coco-2014-dataset-for-yolov3/versions/4/coco2014" # Adjust this path
    # ds = COCODataset(root_dir=coco_root, frac=0.01, is_train=True, privileged_indices_set=set([]))
    ds = COCODatasetOnDemand(root_dir=coco_root, frac=1.0, split_name="train", privileged_indices_set=set([]))
    print(f"Number of samples in dataset: {len(ds)}")
    print(ds[0])
    df = get_label_distribution(ds, batch_size=64, num_workers=4)
    print(df)
    
    


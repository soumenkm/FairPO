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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChestXRayDataset(Dataset):
    """
    Dataset class for loading NIH Chest X-Ray images with multi-label classification.

    Args:
        frac (float): Fraction of the dataset to use.
        is_train (bool): If True, use training split; else use test split.
        max_pixels (int): Maximum allowed pixels for an image to be loaded.

    Attributes:
        is_train (bool): Training or test flag.
        frac (float): Fraction of dataset.
        max_pixels (int): Maximum image pixels allowed.
        transform (transforms.Compose): Image transform pipeline.
        ds (List[dict]): List of processed samples.
        label_names (List[str]): List of class label names.
    """
    def __init__(self, frac: float, is_train: bool, max_pixels: int = 89478485) -> None:
        super(ChestXRayDataset, self).__init__()
        self.is_train = is_train
        self.frac = frac
        self.max_pixels = max_pixels
        self.transform = self._get_transform()
        self.ds, self.label_names = self._get_dataset()
        
    def _get_transform(self) -> transforms.Compose:
        """
        Returns the image transformation pipeline for Chest X-Ray images.

        Returns:
            transforms.Compose: Image transformation sequence.
        """
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485],
                std=[0.229]
            )
        ])
        
    def _get_dataset(self) -> Tuple[List[Dict], List[str]]:
        """
        Loads and processes the NIH Chest X-Ray dataset.

        Returns:
            Tuple containing:
                - List of processed samples with image tensors and multi-hot labels.
                - List of label names.
        """
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
        return final_ds, label_names
    
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.ds)
    
    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a single sample at the specified index.

        Args:
            index (int): Index of the sample.

        Returns:
            dict: Sample containing 'pixels' tensor and 'labels' tensor.
        """
        return self.ds[index]

class COCODataset(Dataset):
    """
    Dataset class for loading the COCO 2014 dataset with multi-label classification.

    Args:
        root_dir (str): Root directory containing COCO images and labels.
        frac (float): Fraction of the dataset to use.
        is_train (bool): If True, use training split; else validation split.
        privileged_indices_set (set): Set of privileged class indices.

    Attributes:
        root_dir (str): Root directory path.
        is_train (bool): Training or validation flag.
        frac (float): Fraction of dataset.
        max_pixels (int): Maximum allowed image pixels.
        seed (int): Random seed for shuffling.
        transform (transforms.Compose): Image transformation pipeline.
        split_name (str): Dataset split name ('train2014' or 'val2014').
        data (List[dict]): List of processed samples.
        label_names (List[str]): List of class label names.
        privileged_indices (List[int]): Sorted privileged class indices.
        non_privileged_indices (List[int]): Sorted non-privileged class indices.
    """
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
        """
        Returns the image transformation pipeline for COCO images.

        Returns:
            transforms.Compose: Image transformation sequence.
        """
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_data(self) -> dict:
        """
        Loads and processes the COCO dataset images and labels.

        Returns:
            dict: Dictionary containing 'data' (list of samples) and 'label_names' (class names).
        """
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

        with open(names_file, 'r') as f:
            label_names = [line.strip() for line in f.readlines() if line.strip()]
        num_classes = len(label_names)
        logging.info(f"Loaded {num_classes} class names from {names_file}")

        all_image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        logging.info(f"Found {len(all_image_files)} potential image files in {img_dir}.")
        
        random.seed(self.seed)
        random.shuffle(all_image_files)
        num_samples_to_keep = int(len(all_image_files) * self.frac)
        all_image_files = all_image_files[:num_samples_to_keep]
        
        processed_data = []
        skipped_count = 0

        with tqdm.tqdm(all_image_files, desc=f"Preparing {split} dataset", colour="green", unit="images") as pbar:
            for img_filename in pbar:
                img_path = os.path.join(img_dir, img_filename)
                base_name, _ = os.path.splitext(img_filename)
                label_filename = base_name + ".txt"
                label_path = os.path.join(label_dir, label_filename)

                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")

                        if img.width * img.height > self.max_pixels:
                            skipped_count += 1
                            continue

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
                                        except ValueError:
                                            logging.warning(f"Non-integer class ID found in first column of {label_filename}. Line: '{line.strip()}'")
                        transformed_image = self.transform(img)
                        multi_hot_label = torch.zeros(num_classes, dtype=torch.float32)
                        if unique_class_ids:
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

        return {"data": processed_data, "label_names": label_names}

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a single sample at the specified index.

        Args:
            index (int): Index of the sample.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with 'pixels' tensor and 'labels' tensor.
        """
        return self.data[index]

    def get_label_names(self) -> List[str]:
        """
        Returns the list of class label names.

        Returns:
            List[str]: List of label names.
        """
        return self.label_names

class COCODatasetOnDemand(Dataset):
    """
    Dataset class for COCO 2014 dataset with on-demand loading using pre-generated index.

    Args:
        root_dir (str): Root directory containing COCO data.
        split_name (str): Dataset split name ("train", "val", or "test").
        frac (float): Fraction of dataset to load.
        privileged_indices_set (set, optional): Set of privileged class indices.
        seed (int): Random seed.
        index_dir (str, optional): Directory for index files.
        force_regenerate (bool): Force regeneration of index.

    Attributes:
        root_dir (Path): Root directory Path.
        frac (float): Fraction of dataset.
        seed (int): Random seed.
        transform (transforms.Compose): Image transformation pipeline.
        split_name (str): Dataset split name.
        index_dir (Path): Directory for index files.
        label_names (List[str]): List of class label names.
        num_classes (int): Number of classes.
        privileged_indices (List[int]): Sorted privileged class indices.
        non_privileged_indices (List[int]): Sorted non-privileged class indices.
        index_file_path (Path): Path to index CSV file.
        index (List[Tuple[str, str]]): List of (image_path, label_path) pairs.
    """
    def __init__(
        self,
        root_dir: str,
        split_name: str,
        frac: float = 1.0,
        privileged_indices_set: set = None,
        seed: int = 42,
        index_dir: Optional[str] = None,
        force_regenerate: bool = False
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
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
        if privileged_indices_set is None:
            privileged_indices_set = set()
        if index_dir is None:
            self.index_dir = self.root_dir / ".index_cache"
        else:
            self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.label_names = self._load_label_names()
        self.num_classes = len(self.label_names)
        all_labels_set = set(range(self.num_classes))
        valid_privileged_set = {idx for idx in privileged_indices_set if 0 <= idx < self.num_classes}
        if len(valid_privileged_set) != len(privileged_indices_set):
            logging.warning("Some provided privileged indices were out of bounds and ignored.")
        non_privileged_indices_set = all_labels_set - valid_privileged_set
        self.privileged_indices = sorted(list(valid_privileged_set))
        self.non_privileged_indices = sorted(list(non_privileged_indices_set))
        logging.info(f"Privileged Indices: {self.privileged_indices}")
        logging.info(f"Non-Privileged Indices: {self.non_privileged_indices}")
        self.index_file_path = self._get_index_filepath()
        self.index = self._load_or_generate_index(force_regenerate)
        if not self.index:
            logging.warning(f"No index data loaded for the {self.split_name} split. Check paths and dataset integrity.")

    def _get_transform(self) -> transforms.Compose:
        """
        Returns the image transformation pipeline.

        Returns:
            transforms.Compose: Image transformations.
        """
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
        """
        Loads class names from the coco.names file.

        Returns:
            List[str]: List of class label names.

        Raises:
            FileNotFoundError: If coco.names file does not exist.
        """
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
        """
        Generates the filepath for the index CSV file.

        Returns:
            Path: Path to index CSV file.
        """
        frac_str = f"{self.frac:.4f}".replace('.', 'p')
        filename = f"coco_idx_{self.split_name}_frac{frac_str}_seed{self.seed}.csv"
        return self.index_dir / filename

    def _generate_index(self) -> List[Tuple[str, str]]:
        """
        Generates the index CSV file containing image and label file paths.

        Returns:
            List of tuples: Each tuple is (image_path, label_path).
        """
        logging.info(f"Generating index file: {self.index_file_path}...")
        img_dir = self.root_dir / "images" / self.split_name
        label_dir = self.root_dir / "labels" / self.split_name
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not label_dir.is_dir():
            logging.warning(f"Label directory not found: {label_dir}. Images may not have corresponding labels.")

        image_files = [f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
        logging.info(f"Found {len(image_files)} potential image files in {img_dir}.")

        index_data = []
        missing_labels = 0
        with tqdm.tqdm(image_files, desc=f"Scanning for {self.split_name} files", colour="blue", unit="images") as pbar:
            for img_path in pbar:
                base_name = img_path.stem
                label_filename = base_name + ".txt"
                label_path = label_dir / label_filename
                img_path_str = str(img_path.resolve())
                label_path_str = str(label_path.resolve())

                if not label_path.exists():
                    missing_labels += 1
                    index_data.append((img_path_str, label_path_str))
                else:
                    index_data.append((img_path_str, label_path_str))

        if missing_labels > 0:
            logging.warning(f"{missing_labels} images did not have a corresponding label file in {label_dir}.")

        random.seed(self.seed)
        random.shuffle(index_data)
        num_samples_to_keep = int(len(index_data) * self.frac)
        final_index_data = index_data[:num_samples_to_keep]

        logging.info(f"Selected {len(final_index_data)} samples based on frac={self.frac}, seed={self.seed}.")

        try:
            with open(self.index_file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['image_path', 'label_path'])
                writer.writerows(final_index_data)
            logging.info(f"Successfully saved index to {self.index_file_path}")
        except IOError as e:
            logging.error(f"Failed to write index file {self.index_file_path}: {e}")
            return []

        return final_index_data

    def _load_or_generate_index(self, force_regenerate: bool) -> List[Tuple[str, str]]:
        """
        Loads the index from CSV or generates it if missing or forced.

        Args:
            force_regenerate (bool): Whether to force regeneration of index.

        Returns:
            List of tuples: (image_path, label_path).
        """
        if not force_regenerate and self.index_file_path.exists():
            logging.info(f"Loading existing index file: {self.index_file_path}")
            index_data = []
            try:
                with open(self.index_file_path, 'r', newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    header = next(reader)
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
        """
        Returns the number of indexed samples.

        Returns:
            int: Number of samples.
        """
        return len(self.index)

    def __getitem__(self, index: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Loads and returns a single sample (image and multi-hot labels) on demand.

        Args:
            index (int): Index of the sample.

        Returns:
            dict or None: Dictionary with 'pixels' tensor and 'labels' tensor, or None if error occurs.
        """
        if index >= len(self.index):
            raise IndexError(f"Index {index} out of bounds for dataset with length {len(self.index)}")

        img_path_str, label_path_str = self.index[index]

        try:
            img = Image.open(img_path_str).convert("RGB")
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
                            except ValueError:
                                pass
            transformed_image = self.transform(img)
            multi_hot_label = torch.zeros(self.num_classes, dtype=torch.float32)
            if unique_class_ids:
                indices = list(unique_class_ids)
                multi_hot_label[indices] = 1.0
            return {"pixels": transformed_image, "labels": multi_hot_label}
        except FileNotFoundError:
            logging.error(f"ERROR in __getitem__: Image file not found at {img_path_str} (listed in index). Skipping sample {index}.")
            return None
        except (UnidentifiedImageError, IOError) as e:
            logging.error(f"ERROR in __getitem__: Failed to load/read image {img_path_str}: {e}. Skipping sample {index}.")
            return None
        except Exception as e:
            logging.error(f"ERROR in __getitem__: Unexpected error processing sample {index} ({img_path_str}): {e}")
            return None

    def get_label_names(self) -> List[str]:
        """
        Returns the list of class label names.

        Returns:
            List[str]: List of label names.
        """
        return self.label_names

class NUSWIDEDatasetOnDemand(Dataset):
    """
    Dataset class for NUS-WIDE dataset with on-demand loading and indexing.

    Args:
        root_dir (str): Root directory of NUS-WIDE dataset.
        split_name (str): Dataset split ("train" or "test").
        frac (float): Fraction of dataset to use.
        privileged_indices_set (set, optional): Privileged class indices.
        seed (int): Random seed.
        index_dir (str, optional): Directory for index files.
        force_regenerate (bool): Force index regeneration.

    Attributes:
        root_dir (Path): Root directory path.
        frac (float): Fraction of dataset.
        seed (int): Random seed.
        transform (transforms.Compose): Image transformation pipeline.
        nus_split_name (str): Dataset split name normalized.
        label_names (List[str]): List of 81 concept names.
        num_classes (int): Number of classes.
        privileged_indices (List[int]): Privileged class indices.
        non_privileged_indices (List[int]): Non-privileged class indices.
        image_base_dir (Path): Base directory for images.
        index_file_path (Path): Path to index CSV file.
        index (List[Tuple[str, str]]): List of (image_rel_path, label_vector_str).
    """
    def __init__(
        self,
        root_dir: str,
        split_name: str,
        frac: float = 1.0,
        privileged_indices_set: Optional[set] = None,
        seed: int = 42,
        index_dir: Optional[str] = None,
        force_regenerate: bool = False
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.frac = max(0.0, min(1.0, frac))
        self.seed = seed
        self.transform = self._get_transform()
        self.nus_split_name = split_name.lower()
        if self.nus_split_name not in ["train", "test"]:
            raise ValueError(f"Invalid split name for NUS-WIDE: {split_name}. Use 'train' or 'test'.")
        if privileged_indices_set is None:
            privileged_indices_set = set()
        if index_dir is None:
            self.index_dir = self.root_dir / ".nuswide_index_cache"
        else:
            self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.label_names = self._load_label_names()
        self.num_classes = len(self.label_names)
        if self.num_classes != 81:
            logging.warning(f"Expected 81 classes for NUS-WIDE, but found {self.num_classes}. Check All_Tags.txt or equivalent.")
        all_labels_set = set(range(self.num_classes))
        valid_privileged_set = {idx for idx in privileged_indices_set if 0 <= idx < self.num_classes}
        if len(valid_privileged_set) != len(privileged_indices_set):
            logging.warning("Some provided NUS-WIDE privileged indices were out of bounds and ignored.")
        non_privileged_indices_set = all_labels_set - valid_privileged_set
        self.privileged_indices = sorted(list(valid_privileged_set))
        self.non_privileged_indices = sorted(list(non_privileged_indices_set))
        logging.info(f"NUS-WIDE Privileged Indices: {self.privileged_indices}")
        logging.info(f"NUS-WIDE Non-Privileged Indices: {self.non_privileged_indices}")
        self.image_base_dir = self.root_dir / "Flickr"
        self.index_file_path = self._get_index_filepath()
        self.index = self._load_or_generate_index(force_regenerate)
        if not self.index:
            logging.warning(f"No NUS-WIDE index data loaded for the {self.nus_split_name} split.")

    def _get_transform(self) -> transforms.Compose:
        """
        Returns the image transformation pipeline.

        Returns:
            transforms.Compose: Image transformations.
        """
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
        """
        Loads the 81 concept names for NUS-WIDE dataset.

        Returns:
            List[str]: List of concept names.

        Raises:
            FileNotFoundError: If concept file not found.
        """
        tags_file = self.root_dir / "NUS_WID_Tags" / "Concepts81.txt"
        if not tags_file.is_file():
            tags_file = self.root_dir / "NUS_WID_Tags" / "All_Tags.txt"
            if not tags_file.is_file():
                raise FileNotFoundError(f"NUS-WIDE class names file not found in {self.root_dir / 'NUS_WID_Tags'}")
        try:
            with open(tags_file, 'r') as f:
                label_names = [line.strip() for line in f.readlines() if line.strip()]
            if not label_names or len(label_names) != 81:
                logging.warning(f"NUS-WIDE class names file {tags_file} did not yield 81 names. Found {len(label_names)}.")
                if not label_names and len(label_names) < 81:
                    return [f"concept_{i}" for i in range(81)]
            logging.info(f"Loaded {len(label_names)} NUS-WIDE class names from {tags_file}")
            return label_names
        except Exception as e:
            logging.error(f"Error reading NUS-WIDE class names file {tags_file}: {e}")
            raise

    def _get_index_filepath(self) -> Path:
        """
        Returns the path for the index CSV file.

        Returns:
            Path: Index CSV file path.
        """
        frac_str = f"{self.frac:.4f}".replace('.', 'p')
        filename = f"nuswide_idx_{self.nus_split_name}_frac{frac_str}_seed{self.seed}.csv"
        return self.index_dir / filename

    def _generate_index(self) -> List[Tuple[str, str]]:
        """
        Generates the NUS-WIDE index by reading image lists and label vectors.

        Returns:
            List of tuples: (image_rel_path, label_vector_str)
        """
        logging.info(f"Generating NUS-WIDE index file: {self.index_file_path}...")
        image_list_filename = f"{self.nus_split_name.capitalize()}Imagelist.txt"
        image_list_file = self.root_dir / "ImageList" / image_list_filename
        label_filename = f"Labels_{self.nus_split_name.capitalize()}.txt"
        label_file = self.root_dir / "Groundtruth" / label_filename

        if not image_list_file.is_file():
            raise FileNotFoundError(f"NUS-WIDE Image list file not found: {image_list_file}")
        if not label_file.is_file():
            raise FileNotFoundError(f"NUS-WIDE Label file not found: {label_file}")

        try:
            with open(image_list_file, 'r') as f_img:
                image_rel_paths = [line.strip().replace('\\', '/') for line in f_img if line.strip()]
            with open(label_file, 'r') as f_lbl:
                labels_str = [line.strip() for line in f_lbl if line.strip()]

            if len(image_rel_paths) != len(labels_str):
                raise ValueError(f"Mismatch in number of images ({len(image_rel_paths)}) and labels ({len(labels_str)}) for NUS-WIDE {self.nus_split_name}")

            index_data_raw = []
            for i, rel_path in enumerate(tqdm(image_rel_paths, desc=f"Processing NUS-WIDE {self.nus_split_name} entries", unit="image")):
                full_img_path = self.image_base_dir / rel_path
                if not full_img_path.exists():
                    logging.warning(f"Image {full_img_path} listed in {image_list_filename} but not found. Skipping.")
                    continue
                label_vector_str = labels_str[i].split()
                if len(label_vector_str) != self.num_classes:
                    logging.warning(f"Label vector for {rel_path} has incorrect length ({len(label_vector_str)} vs {self.num_classes}). Skipping.")
                    continue
                try:
                    _ = torch.tensor([int(x) for x in label_vector_str], dtype=torch.float32)
                except ValueError:
                    logging.warning(f"Invalid value in label vector for {rel_path}. Skipping.")
                    continue
                index_data_raw.append((rel_path, " ".join(label_vector_str)))

        except Exception as e:
            logging.error(f"Error processing NUS-WIDE source files: {e}")
            return []

        random.seed(self.seed)
        random.shuffle(index_data_raw)
        num_samples_to_keep = int(len(index_data_raw) * self.frac)
        final_index_data = index_data_raw[:num_samples_to_keep]

        logging.info(f"Selected {len(final_index_data)} NUS-WIDE samples based on frac={self.frac}, seed={self.seed}.")

        try:
            with open(self.index_file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['image_rel_path', 'label_vector_str'])
                writer.writerows(final_index_data)
            logging.info(f"Successfully saved NUS-WIDE index to {self.index_file_path}")
        except IOError as e:
            logging.error(f"Failed to write NUS-WIDE index file {self.index_file_path}: {e}")
            return []
        return final_index_data

    def _load_or_generate_index(self, force_regenerate: bool) -> List[Tuple[str, str]]:
        """
        Loads or generates the NUS-WIDE index CSV file.

        Args:
            force_regenerate (bool): Whether to force regeneration.

        Returns:
            List of tuples: (image_rel_path, label_vector_str).
        """
        if not force_regenerate and self.index_file_path.exists():
            logging.info(f"Loading existing NUS-WIDE index file: {self.index_file_path}")
            index_data = []
            try:
                with open(self.index_file_path, 'r', newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    header = next(reader)
                    if header != ['image_rel_path', 'label_vector_str']:
                        logging.warning(f"NUS-WIDE Index file {self.index_file_path} has unexpected header. Regenerating.")
                        return self._generate_index()
                    for row in reader:
                        if len(row) == 2:
                            index_data.append((row[0], row[1]))
                        else:
                            logging.warning(f"Skipping malformed row in NUS-WIDE index {self.index_file_path}: {row}")
                logging.info(f"Loaded {len(index_data)} NUS-WIDE entries from index.")
                return index_data
            except Exception as e:
                logging.warning(f"Failed to load/parse NUS-WIDE index file {self.index_file_path}: {e}. Regenerating.")
                return self._generate_index()
        else:
            return self._generate_index()

    def __len__(self) -> int:
        """
        Returns the number of samples in the NUS-WIDE dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.index)

    def __getitem__(self, index: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Loads and returns a single NUS-WIDE sample (image and multi-hot label vector).

        Args:
            index (int): Sample index.

        Returns:
            dict or None: Dictionary with 'pixels' tensor and 'labels' tensor, or None if error occurs.
        """
        if index >= len(self.index):
            raise IndexError(f"NUS-WIDE Index {index} out of bounds for dataset with length {len(self.index)}")

        image_rel_path, label_vector_str = self.index[index]
        full_img_path = self.image_base_dir / image_rel_path

        try:
            img = Image.open(full_img_path).convert("RGB")
            transformed_image = self.transform(img)
            label_values = [int(x) for x in label_vector_str.split()]
            if len(label_values) != self.num_classes:
                logging.error(f"Label vector for {image_rel_path} in NUS-WIDE index has incorrect length. Expected {self.num_classes}, got {len(label_values)}.")
                return None
            multi_hot_label = torch.tensor(label_values, dtype=torch.float32)
            return {"pixels": transformed_image, "labels": multi_hot_label}
        except FileNotFoundError:
            logging.error(f"ERROR in NUS-WIDE __getitem__: Image file not found at {full_img_path} (listed in index). Skipping sample {index}.")
            return None
        except (UnidentifiedImageError, IOError) as e:
            logging.error(f"ERROR in NUS-WIDE __getitem__: Failed to load/read image {full_img_path}: {e}. Skipping sample {index}.")
            return None
        except ValueError as e:
            logging.error(f"ERROR in NUS-WIDE __getitem__: Failed to parse label vector for {full_img_path}: '{label_vector_str}'. Error: {e}. Skipping sample {index}.")
            return None
        except Exception as e:
            logging.error(f"ERROR in NUS-WIDE __getitem__: Unexpected error processing sample {index} ({full_img_path}): {e}")
            return None

    def get_label_names(self) -> List[str]:
        """
        Returns the list of NUS-WIDE class label names.

        Returns:
            List[str]: List of label names.
        """
        return self.label_names


def collate_fn_skip_none(batch):
    """
    Collate function that filters out None items from the batch.

    Args:
        batch (List): List of samples, possibly containing None.

    Returns:
        Batch with None items removed or dummy batch if all are None.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        logging.warning("Entire batch was None after filtering.")
        return {'pixels': torch.empty(0), 'labels': torch.empty(0)}
    return torch.utils.data.dataloader.default_collate(batch)


def get_label_distribution(dataset: COCODatasetOnDemand, batch_size: int = 64, num_workers: int = 4):
    """
    Calculates the distribution of positive labels in a COCODatasetOnDemand.

    Args:
        dataset (COCODatasetOnDemand): Dataset instance.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of DataLoader workers.

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

    label_counts = torch.zeros(num_labels, dtype=torch.int64)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_skip_none
    )

    logging.info(f"Calculating label distribution for {len(dataset)} samples...")
    for batch in tqdm(loader, desc="Counting Labels", unit="batch"):
        if batch is None or 'labels' not in batch or batch['labels'].numel() == 0:
            continue
        batch_label_sum = torch.sum(batch['labels'], dim=0)
        label_counts += batch_label_sum.long()

    distribution_df = pd.DataFrame({
        'label_id': list(range(num_labels)),
        'label_name': label_names,
        'count': label_counts.numpy()
    })

    logging.info("Label distribution calculation complete.")
    return distribution_df


if __name__ == '__main__':
    coco_root = "/raid/speech/user/.cache/kagglehub/datasets/jeffaudi/coco-2014-dataset-for-yolov3/versions/4/coco2014"
    ds = COCODatasetOnDemand(root_dir=coco_root, frac=1.0, split_name="train", privileged_indices_set=set([]))
    print(f"Number of samples in dataset: {len(ds)}")
    print(ds[0])
    df = get_label_distribution(ds, batch_size=64, num_workers=4)
    print(df)

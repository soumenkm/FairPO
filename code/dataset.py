from datasets import load_dataset
import torch, tqdm, os, sys, pickle, datetime
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, List, Union
from torchvision import transforms
torch.manual_seed(42)

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
                mean=[0.485, 0.456, 0.406],      
                std=[0.229, 0.224, 0.225]     
            )
        ])
        
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

if __name__ == "__main__":
    ds = ChestXRayDataset(frac=0.01, is_train=True)
    print(ds[0])
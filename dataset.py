from typing import *
import os

from torch.utils.data import Dataset
from PIL import Image

class AnimeDataset(Dataset):

    def __init__(self, root_dir: str, transform = None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.images = [file for file in os.listdir(root_dir) if file.endswith(".jpg")]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):

        img = Image.open(os.path.join(self.root_dir, self.images[index]))

        if self.transform is not None:
            img = self.transform(img)
        
        return img
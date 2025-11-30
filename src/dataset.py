from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch

from .preprocessing import Vocabulary
from pathlib import Path
import os

class FlickrDataset(Dataset):
    def __init__(self, images_dir, captions_file, vocab, transform=None, max_len=30):
        self.images_dir = Path(images_dir)
        self.vocab = vocab
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        ])
        self.max_len = max_len

        # load captions
        self.image_ids = []
        self.captions = []
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line=line.strip()
                if not line: continue
                parts = line.split('\t')
                if len(parts) != 2:
                    parts = [p.strip() for p in line.split(' ',1)]
                image_id = parts[0].split('#')[0]
                caption = parts[1]
                self.image_ids.append(image_id)
                self.captions.append(caption)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = self.images_dir / img_id
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        caption = self.captions[idx]
        numericalized = [self.vocab.stoi['<start>']] + self.vocab.numericalize(caption) + [self.vocab.stoi['<end>']]
        if len(numericalized) > self.max_len:
            numericalized = numericalized[:self.max_len]
        caption_tensor = torch.tensor(numericalized)

        return image, caption_tensor


def collate_fn(batch):
    images = [item[0].unsqueeze(0) for item in batch]
    images = torch.cat(images, dim=0)

    captions = [item[1] for item in batch]
    lengths = [len(cap) for cap in captions]
    max_len = max(lengths)
    padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i,cap in enumerate(captions):
        padded[i,:len(cap)] = cap

    return images, padded, lengths

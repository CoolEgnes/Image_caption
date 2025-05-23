"""
This file is used to load the data
"""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPProcessor


class VizwizDataset(Dataset):
    def __init__(self, image_dir, caption_file, processor, max_length=50):
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        self.captions = self._load_captions(caption_file)

    def _load_captions(self, caption_file):
        captions = []
        with open(caption_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                image_path = parts[0]
                caption_list = parts[1:]
                captions.append((image_path, caption_list))
        return captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_path, caption_list = self.captions[idx]
        image = Image.open(image_path).convert("RGB")

        # Process image and caption
        inputs = self.processor(images=image, return_tensors="pt", padding=True, truncation=True)

        # preprocess all 5 captions
        all_caption_ids = []
        for caption in caption_list:
            caption_ids = self.processor.tokenizer.encode(
                caption,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).squeeze(0)
            all_caption_ids.append(caption_ids)

        return inputs['pixel_values'].squeeze(0), torch.stack(all_caption_ids)  # (5, max_length)


def get_data_loaders(image_dir, caption_file, batch_size):
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        captions = torch.stack([item[1] for item in batch])  # (batch_size,5,seq_len)
        return images, captions
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dataset = VizwizDataset(image_dir, caption_file, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return dataloader


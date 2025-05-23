"""
This file is used to load the data
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict


class VizwizDataset(Dataset):
    def __init__(self, image_dir, caption_file, processor, max_length=50, max_captions=None):
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        self.max_captions = max_captions

        self.samples = defaultdict(list)
        with open(caption_file, 'r') as f:
            for line in f:
                image_path, caption = line.strip().split('\t')
                self.samples[image_path].append(caption)
        # Convert dictionary to list
        self.samples = [{"image_path": k, "captions": v} for k,v in self.samples.items()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        captions = sample["captions"]
        image = Image.open(image_path).convert("RGB")

        # Process image and caption
        inputs = self.processor(images=image, return_tensors='pt', padding=True, truncation=True)
        inputs['pixel_values'] = inputs['pixel_values'].squeeze(0)

        # Tokenize caption
        if self.max_captions is not None:
            captions = captions[:self.max_captions]
        caption_ids = self.processor.tokenizer.encode(
            captions, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt'
        ).squeeze(0)

        return inputs['pixel_values'], caption_ids['input_ids'], caption_ids['attention_mask']


def collate_fn(batch):
    """
    Custom collate function to process batch data
    :param batch: a batch of data returned by VizwizDataset
    :return:
    """
    images, input_ids, attention_masks = zip(*batch)
    # Stack the image tensors
    images = torch.stack(images, dim=0)

    # Stack input_ids and attention_masks
    input_ids = torch.stack(input_ids, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)

    return images, input_ids, attention_masks

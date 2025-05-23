"""
This file is the main file to run
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from load_data import VizwizDataset


if __name__ == "__main__":
    device = "cpu"

    # load the data
    image_dir = "dataset/train"
    caption_file = "dataset/annotations/train.txt"
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dataset = VizwizDataset(image_dir, caption_file, processor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    vocab_size = processor.tokenizer.vocab_size
    hidden_size = 512

    for batch_idx, (image, input_ids, attention_masks) in enumerate(dataloader):
        pass

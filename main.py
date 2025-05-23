"""
This is the main file to run
"""
import os.path

import torch
from transformers import CLIPModel, CLIPProcessor
from load_data import get_data_loaders
from LSTM import CLIPLSTMCaptioner
from train import train, evaluate
import torch.optim as optim
from caption_parser import extract_annotations_to_txt, save_to_txt
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


def main():
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    batch_size = 16

    # load the data
    image_dir = "dataset/train"
    caption_file = "dataset/annotations/train.txt"
    train_json = "dataset/annotations/train.json"
    if not os.path.exists(caption_file):
        train_annotations = extract_annotations_to_txt(train_json, image_dir)
        save_to_txt(train_annotations, caption_file)
    train_loader = get_data_loaders(image_dir, caption_file, batch_size)
    for batch_idx, (image, captions) in enumerate(train_loader):
        print(f"image.shape: {image.shape}")
        print(f"captions.shape: {captions.shape}")
        break
    # initialize the model
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    vocab_size = processor.tokenizer.vocab_size
    hidden_size = 512
    model = CLIPLSTMCaptioner(clip_model, hidden_size, vocab_size).to(device)

    # loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # train and evaluation
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device, batch_size)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")

        # evaluation
        # val_loss = evaluate(model, val_loader, criterion, device)
        # print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}")

    # save the model
    # torch.save(model.state_dict(), "clip_lstm_captioner.pth")


if __name__ == "__main__":
    main()


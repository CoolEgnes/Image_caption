"""
This file is used to define the training function
"""
import torch
from tqdm import tqdm


def train(model, dataloader, criterion, optimizer, device, batch_size):
    model.train()
    running_loss = 0.0

    # set the hyperparameter 'reduction' to none to get the loss for each token
    criterion_none = torch.nn.CrossEntropyLoss(reduction="none")

    for images, all_captions in tqdm(dataloader):  # all_captions: (batch,5,seq_len)
        images, all_captions = images.to(device), all_captions.to(device)

        # get the size of current batch
        current_batch_size = images.size(0)
        seq_len = all_captions.size(-1)  # original seq_len

        # flat the captions
        captions_flat = all_captions.view(-1, seq_len)  # (16*5,50) = (80,50)

        # Forward pass
        # outputs = model(images, all_captions[:, :, :-1])  # images: Tensor: (batch_size,3,224,224) captions: Tensor: (batch_size,seq_len)
        outputs = model(images, captions_flat[:, :-1])  # images: Tensor: (batch_size,3,224,224) captions: Tensor: (batch_size,seq_len)
        # outputs: Tensor: (batch_size,seq_len,vocab_size)

        targets = captions_flat[:, 1:].reshape(-1)

        # calculate the loss of each token
        loss_per_token = criterion_none(outputs.reshape(-1, outputs.size(-1)), targets)

        # group the loss by captions
        loss_per_caption = loss_per_token.view(current_batch_size * 5, seq_len-1).mean(dim=1)  # (batch*5,)

        # group the loss by images and calculate the average
        loss_per_image = loss_per_caption.view(current_batch_size, 5).mean(dim=1)  # (batch_size,)

        # calculate the final loss
        loss = loss_per_image.mean()

        # loss = 0
        # # calculate the average loss of each image
        # for i in range(5):
        #     start = i * batch_size
        #     end = (i + 1) * batch_size
        #     caption_output = outputs[start:end, :, :]
        #     caption_target = all_captions[:, i, 1:].reshape(-1)
        #     temp1 = caption_output.reshape(-1, caption_output.size(-1))
        #     temp2 = caption_target
        #     loss += criterion(caption_output.reshape(-1, caption_output.size(-1)), caption_target)
        # loss = loss / 5
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, captions in tqdm(dataloader):
            images, captions = images.to(device), captions.to(device)

            # Forward pass
            outputs = model(images, captions[:, :-1])
            loss = criterion(outputs.view(-1, model.fc.out_features), captions[:, 1:].reshape(-1))

            running_loss += loss.item()

    return running_loss / len(dataloader)


def generate_caption(clip_model, image, lstm_model, processor, max_length=50):
    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)

    


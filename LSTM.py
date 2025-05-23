"""
This file is used to define the structure of the LSTM model
"""

import torch.nn as nn
import torch


class CLIPLSTMCaptioner(nn.Module):
    def __init__(self, clip_model, hidden_size, vocab_size, num_layers=1):
        super(CLIPLSTMCaptioner, self).__init__()
        self.clip_model = clip_model
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, images, all_captions):
        """

        :param images: (batch_size, 3, 224, 224)
        :param all_captions: (batch_size, 5, seq_len)
        :return: None
        """
        batch_size = images.size(0)

        # Extract the image features (batch_size, hidden_size)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(pixel_values=images)

        # Extend the image feature (batch_size*5, hidden_size)
        image_features = (
            image_features.unsqueeze(1).repeat(1, 5, 1).view(batch_size * 5, -1)
        )
        # Process all the captions (batch_size*5, seq_len-1)
        captions = all_captions.view(batch_size * 5, -1)
        caption_embeddings = self.embedding(captions[:, :-1])

        # Concatenate image features and caption embeddings
        lstm_input = torch.cat(
            [
                image_features.unsqueeze(1),  # (batch*5,1,hidden)
                caption_embeddings,  # (batch*5,seq_len-1,hidden)
            ],
            dim=1,
        )

        # LSTM
        lstm_out, _ = self.lstm(lstm_input)

        # Predict next word
        outputs = self.fc(lstm_out)

        # Remove the first output (image feature part)
        # outputs = outputs[:, 1:, :]  # (batch_size, sequence_length, vocab_size)

        return outputs  # (batch*5, seq_len, vocab_size)

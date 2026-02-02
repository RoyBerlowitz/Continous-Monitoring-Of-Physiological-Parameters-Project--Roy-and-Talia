import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class embedder_cnn (nn.Module):
    # We define a convolutional neural network to obtain an embedding vector that holds learned connections
    def __init__(self, number_of_channels, number_of_classes, embedding_size, dropout=0.3):
        super(embedder_cnn, self).__init__()
        # we first define the blocks of the CNN, which are the part when the model study the underlying relation in the data.
        # we expand the number of channels while reduce the time axis to get important time encoded information
        self.features = nn.Sequential(
            # First Block
            nn.Conv1d(in_channels=number_of_channels, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=dropout),
            # Second Block
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=dropout),
            # Third Block
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(p=dropout)
        )
        # we add dropout to maximize model efficiency
        self.dropout = nn.Dropout(dropout)
        # we define the embedding layer that holds the features we will use for the model
        self.embedding_layer = nn.Sequential(
            nn.Linear(in_features=64, out_features=embedding_size),
        )
        # we define the final classifier
        self.classifier = nn.Linear(in_features=embedding_size, out_features=number_of_classes)

    def forward(self, x):
        # we define the forward operation of the model
        x = self.features(x)
        x = x.flatten(1)
        # we use dropout
        x = self.dropout(x)
        # we use Relu to get only the positive embedding value - as the negative here does not make any sense
        embedding  = F.relu(self.embedding_layer(x))
        output = self.classifier(embedding)
        return output

    def get_embedding (self, x):
        # this functions gives us the embedding, which will allow us add features for our model
        # we set the model in eval mode as we just use the training
        self.eval()
        # we do not allow back propagation
        with torch.no_grad():
            # we set the data to the device
            device = next(self.parameters()).device
            x = x.to(device)
            # we preform the forward
            x = self.features(x)
            x = x.flatten(1)
            # we get the embedding of the model
            embedding = F.relu(self.embedding_layer(x))
        return embedding.cpu().numpy()


def train_one_epoch(dataloader, model, optimizer, criterion, device):
    # this function define the training loop of the model
    model.train()
    running_loss = 0.0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)
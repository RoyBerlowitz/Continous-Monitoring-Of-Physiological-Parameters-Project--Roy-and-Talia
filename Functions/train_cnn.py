import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class embedder_cnn (nn.Module):
    def __init__(self, number_of_channels, number_of_classes, embedding_size):
        super(embedder_cnn, self).__init__()
        self.features = nn.Sequential(
            # בלוק 1: תפיסת שינויים מיידיים (Low Level)
            nn.Conv1d(in_channels=number_of_channels, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # חיתוך זמן לחצי

            # בלוק 2: תבניות בינוניות (Mid Level)
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # חיתוך זמן לרבע

            # בלוק 3: תבניות מורכבות (High Level)
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # כיווץ כל הזמן שנשאר לנקודה אחת
        )

        self.embedding_layer = nn.Sequential(
            nn.Linear(in_features=64, out_features=embedding_size),
        )

        self.classifier = nn.Linear(in_features=embedding_size, out_features=number_of_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        embedding  = F.relu(self.embedding_layer(x))
        output = self.classifier(embedding)
        return output

    def get_embedding (self, x):
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            x = x.to(device)
            x = self.features(x)
            x = x.flatten(1)
            embedding = F.relu(self.embedding_layer(x))
        return embedding.cpu().numpy()


def train_one_epoch(dataloader, model, optimizer, criterion, device):
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
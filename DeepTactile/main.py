# -*- coding: utf-8 -*`-
"""
Created on August 1, 2024

@author: Famging Guo

This is the code for Event-driven Tactile Sensing With Dense Spiking Graph Neural Networks.

"""

import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import DeepTactile
from torch import nn
import numpy as np

# Dataset class 
class TactileDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, train=True):
        self.data_path = os.path.join(data_path, 'train' if train else 'test')
        self.files = os.listdir(self.data_path)

    def __getitem__(self, index):
        file_name = self.files[index]
        label = int(file_name.split('_label_')[-1].split('.')[0])
        data = np.load(os.path.join(self.data_path, file_name))
        data = torch.from_numpy(data)
        label = torch.tensor(label, dtype=torch.long)
        return data, label

    def __len__(self):
        return len(self.files)

# Training function
def train(model, train_loader, criterion, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct, total = 0, 0

        for train_data, train_label in train_loader:
            train_data, train_label = train_data.to(device), train_label.to(device)
            optimizer.zero_grad()

            outputs = model(train_data)
            labels_one_hot = torch.zeros(train_label.size(0), model.num_classes).scatter_(
                1, train_label.view(-1, 1), 1
            )
            loss = criterion(outputs, labels_one_hot)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += train_label.size(0)
            correct += predicted.eq(train_label).sum().item()

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {running_loss:.4f}, Accuracy: {100. * correct / total:.2f}%")

# Evaluation function
def evaluate(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for test_data, test_label in test_loader:
            test_data, test_label = test_data.to(device), test_label.to(device)
            outputs = model(test_data)
            _, predicted = outputs.max(1)
            y_true.extend(test_label.cpu().tolist())
            y_pred.extend(predicted.cpu().tolist())

    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average='weighted') * 100
    recall = recall_score(y_true, y_pred, average='weighted') * 100
    f1 = f1_score(y_true, y_pred, average='weighted') * 100

    print(f"Evaluation - Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1: {f1:.2f}%")

# Main script
def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    train_dataset = TactileDataset(args.dataset, train=True)
    test_dataset = TactileDataset(args.dataset, train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = DeepTactile(
        growth_rate=32, block_config=(3, 3), num_init_features=64,
        bn_size=4, compression_rate=0.5, drop_rate=0, num_classes=10,
        data_path=args.dataset, k=0, useKNN=False, device=device
    ).to(device)

    # Initialize criterion and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.evaluate:
        # Load pre-trained model
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        evaluate(model, test_loader, device)
    else:
        # Train the model
        train(model, train_loader, criterion, optimizer, device, args.epochs)

        # Save the model
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, "model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate DeepTactile")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training/testing")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate a pre-trained model")
    parser.add_argument("--model-path", type=str, help="Path to the pre-trained model")
    parser.add_argument("--save-dir", type=str, default="./models", help="Directory to save the trained model")
    args = parser.parse_args()

    main(args)

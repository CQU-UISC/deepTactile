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

    # Load datasets
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

    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Variables to track best performance
    best_test_acc = 0.0
    best_test_metrics = None
    best_epoch = 0
    best_model_state = None

    # Create directory for saving models
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("Starting training and evaluation...")
    print("-" * 60)

    for epoch in range(0, args.epochs, 1):  # Train one epoch at a time
        # Training phase using the external train function
        train(model, train_loader, criterion, optimizer, device, num_epochs=1)
        
        # Testing phase
        model.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for test_data, test_label in test_loader:
                test_data, test_label = test_data.to(device), test_label.to(device)
                outputs = model(test_data)
                _, predicted = outputs.max(1)
                y_true.extend(test_label.cpu().tolist())
                y_pred.extend(predicted.cpu().tolist())

        # Calculate test metrics
        test_acc = accuracy_score(y_true, y_pred) * 100
        test_precision = precision_score(y_true, y_pred, average='weighted') * 100
        test_recall = recall_score(y_true, y_pred, average='weighted') * 100
        test_f1 = f1_score(y_true, y_pred, average='weighted') * 100

        # Print current epoch results
        print(f"Testing  - Accuracy: {test_acc:.2f}%, Precision: {test_precision:.2f}%, "
              f"Recall: {test_recall:.2f}%, F1: {test_f1:.2f}%")

        # Update best results if current test accuracy is better
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_metrics = {
                'accuracy': test_acc,
                'precision': test_precision,
                'recall': test_recall,
                'f1': test_f1
            }
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()

            # Save the best model
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_model_state,
                'test_metrics': best_test_metrics,
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.save_dir, "best_model.pth"))
            print(f"New best model saved! Test accuracy: {test_acc:.2f}%")

    # Print final results
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best results (Epoch {best_epoch}):")
    print(f"Test Accuracy: {best_test_metrics['accuracy']:.2f}%")
    print(f"Test Precision: {best_test_metrics['precision']:.2f}%")
    print(f"Test Recall: {best_test_metrics['recall']:.2f}%")
    print(f"Test F1-score: {best_test_metrics['f1']:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate DeepTactile")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training/testing")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--save-dir", type=str, default="./models", help="Directory to save the trained model")
    args = parser.parse_args()

    main(args)


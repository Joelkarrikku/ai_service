# File: ai_service/train.py

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.model_selection import train_test_split

# --- Configuration ---
DATA_DIR = "ai_service/data/RWF-2000"
MODEL_SAVE_PATH = "ai_service/custom_models/anomaly_detector.pth"
NUM_CLASSES = 2  # RWF-2000 has two classes: Fight, NoFight
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# --- 1. Dataset and Preprocessing ---

class VideoFrameDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        video_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Extract the middle frame of the video
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)

        # Convert from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            frame = self.transform(frame)
            
        return frame, label

def get_data_loaders():
    all_files = []
    all_labels = []
    class_map = {"NoFight": 0, "Fight": 1}

    for class_name, label in class_map.items():
        class_dir = os.path.join(DATA_DIR, class_name)
        for video_file in os.listdir(class_dir):
            all_files.append(os.path.join(class_dir, video_file))
            all_labels.append(label)

    X_train, X_val, y_train, y_val = train_test_split(all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

    # Define transforms for data augmentation and normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = VideoFrameDataset(X_train, y_train, transform=data_transforms['train'])
    val_dataset = VideoFrameDataset(X_val, y_val, transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader

# --- 2. Model Definition ---

def build_model():
    model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    # Freeze all the parameters in the pre-trained model
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return model

# --- 3. Training and Validation Loop ---

def train_model():
    print("Starting model training...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model().to(device)
    train_loader, val_loader = get_data_loaders()

    criterion = nn.CrossEntropyLoss()
    # Only train the parameters of the final layer
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Training Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = corrects.double() / len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f}")

    # --- 4. Save the Trained Model ---
    model_dir = os.path.dirname(MODEL_SAVE_PATH)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model trained and saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
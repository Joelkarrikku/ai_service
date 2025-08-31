import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import cv2
import os
from sklearn.model_selection import train_test_split

# --- Configuration ---
DATASET_PATH = 'ai_service/data/RWF-2000'
MODEL_SAVE_PATH = 'ai_service/custom_models/anomaly_detector.pth'
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# --- Custom Dataset Class to load video frames ---
class VideoDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        video_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            cap = cv2.VideoCapture(video_path)
            # Get the middle frame of the video
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            middle_frame_idx = frame_count // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                # If reading fails, get the next item
                return self.__getitem__((idx + 1) % len(self.file_paths))

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            return frame, label
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.file_paths))

# --- Main Training Logic ---
def train_model():
    print("Starting model training...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare dataset paths and labels
    fight_videos = [os.path.join(DATASET_PATH, 'Fight', f) for f in os.listdir(os.path.join(DATASET_PATH, 'Fight'))]
    nofight_videos = [os.path.join(DATASET_PATH, 'NoFight', f) for f in os.listdir(os.path.join(DATASET_PATH, 'NoFight'))]
    
    file_paths = fight_videos + nofight_videos
    labels = [0] * len(fight_videos) + [1] * len(nofight_videos) # 0: Fight, 1: NoFight

    # Split data into training and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(file_paths, labels, test_size=0.2, random_state=42, stratify=labels)

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create datasets and dataloaders
    train_dataset = VideoDataset(train_paths, train_labels, transform=data_transforms['train'])
    val_dataset = VideoDataset(val_paths, val_labels, transform=data_transforms['val'])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load pre-trained ResNet18 model and adapt it for our binary classification task
    model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # 2 classes: Fight, NoFight
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
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

        # Validation loop
        model.eval()
        val_loss, corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                corrects += torch.sum(preds == labels.data)
        
        val_loss /= len(val_loader.dataset)
        val_acc = corrects.double() / len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f}")

    # Save the trained model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model trained and saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_model()
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# Adaptive Margin Loss
class AdaptiveMarginLoss(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(AdaptiveMarginLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.weights = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, features, labels):
        # Handle tuple input if needed
        if isinstance(features, tuple):
            features = features[0]
            
        # Normalize weights and features
        norm_weights = F.normalize(self.weights, dim=1)
        norm_features = F.normalize(features, dim=1)

        # Compute cosine similarity
        cosine_similarity = torch.matmul(norm_features, norm_weights.T)

        # Calculate margin using L2 distance instead of cross product
        selected_weights = norm_weights[labels]  # Get the weights for the true classes
        l2_distances = torch.norm(norm_features - selected_weights, dim=1)
        margin = l2_distances.mean()

        # Calculate scale
        scale = math.log(self.num_classes) * math.exp(math.sqrt(2))

        # Apply adaptive margin and scale
        theta = torch.acos(torch.clamp(cosine_similarity, -1.0 + 1e-7, 1.0 - 1e-7))
        
        # Create one-hot encoding for target classes
        one_hot = torch.zeros_like(cosine_similarity)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Apply margin only to positive pairs
        theta_with_margin = torch.where(one_hot == 1, theta + margin, theta)
        logits = scale * torch.cos(theta_with_margin)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        return loss

# Dual Path Network+
class DPNPlus(nn.Module):
    def __init__(self, num_classes):
        super(DPNPlus, self).__init__()
        
        # Block-1: Enhanced feature extraction
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Subsequent blocks follow DPN structure
        self.block2 = self._make_block(64, 128, num_layers=3, stride=2)
        self.block3 = self._make_block(256, 256, num_layers=6, stride=2)
        self.block4 = self._make_block(512, 512, num_layers=10, stride=2)
        self.block5 = self._make_block(1024, 1024, num_layers=3, stride=2)

        # Global average pooling and FC layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _make_block(self, in_channels, out_channels, num_layers, stride):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=32),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.fc(features)
        
        return logits, features

# Add a simple dataset class (replace with your actual dataset)
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[cls]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # Progress bar for training
        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits, features = model(inputs)
            loss = criterion(features, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(logits, 1)
            running_corrects += torch.sum(preds == labels.data)
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            running_loss = 0.0
            running_corrects = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    logits, features = model(inputs)
                    loss = criterion(features, labels)
                    
                    running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(logits, 1)
                    running_corrects += torch.sum(preds == labels.data)
            
            val_loss = running_loss / len(val_loader.dataset)
            val_acc = running_corrects.double() / len(val_loader.dataset)
            
            print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, 'best_model.pth')
        
        print()

# Main training setup
def main():
    # Parameters
    num_classes = 85000  # MS1Mv2 dataset
    feature_dim = 2048
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset and DataLoader (replace paths with your actual data paths)
    train_dataset = FaceDataset(root_dir='path/to/train/data', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)
    
    val_dataset = FaceDataset(root_dir='path/to/val/data', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)
    
    # Model, criterion, and optimizer
    model = DPNPlus(num_classes=num_classes).to(device)
    criterion = AdaptiveMarginLoss(num_classes=num_classes, feature_dim=feature_dim).to(device)
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': criterion.parameters()}
    ], lr=learning_rate)
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

if __name__ == "__main__":
    main()

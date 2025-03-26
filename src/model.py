# %% [markdown]
# # Particle Classification Model
# 
# This notebook implements a ResNet-15 architecture for classifying electron and photon particles.

# %% [markdown]
# ## Imports and Setup

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm.notebook import tqdm
import h5py
import random
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# %% [markdown]
# ## Dataset Class Definition

# %%
# Define custom dataset class
class ParticleDataset(Dataset):
    def __init__(self, electron_path, photon_path, transform=None):
        """
        Args:
            electron_path: Path to electron HDF5 file
            photon_path: Path to photon HDF5 file
            transform: Optional transform to be applied on a sample
        """
        self.transform = transform
        
        # Load electron data
        with h5py.File(electron_path, 'r') as f:
            electron_energy = np.array(f['energy'])
            electron_time = np.array(f['time'])
        
        # Load photon data
        with h5py.File(photon_path, 'r') as f:
            photon_energy = np.array(f['energy'])
            photon_time = np.array(f['time'])
        
        # Stack energy and time as channels
        self.electron_data = np.stack([electron_energy, electron_time], axis=1)
        self.photon_data = np.stack([photon_energy, photon_time], axis=1)
        
        # Create labels (0 for electrons, 1 for photons)
        self.electron_labels = np.zeros(len(self.electron_data))
        self.photon_labels = np.ones(len(self.photon_data))
        
        # Combine data and labels
        self.data = np.vstack([self.electron_data, self.photon_data])
        self.labels = np.concatenate([self.electron_labels, self.photon_labels])
        
        # Shuffle data
        indices = np.arange(len(self.data))
        np.random.shuffle(indices)
        self.data = self.data[indices]
        self.labels = self.labels[indices]
        
        print(f"Dataset loaded with {len(self.data)} samples")
        print(f"Electron samples: {len(self.electron_data)}")
        print(f"Photon samples: {len(self.photon_data)}")
        print(f"Data shape: {self.data.shape}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.data[idx]
        label = self.labels[idx]
        
        # Apply transformations if specified
        if self.transform:
            sample = self.transform(sample)
        
        # Convert to torch tensors
        sample = torch.from_numpy(sample).float()
        label = torch.tensor(label).long()
        
        return sample, label

# %% [markdown]
# ## Model Architecture

# %%
# Define ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

# %%
# Define ResNet15
class ResNet15(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet15, self).__init__()
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        
        # Average pooling and fully connected layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

# %% [markdown]
# ## Training and Evaluation Functions

# %%
# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, device='cuda'):
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_model_weights = None
    patience = 10
    early_stop_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Track statistics
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)
        
        # Update learning rate scheduler
        scheduler.step(val_epoch_loss)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
        
        # Save best model
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model_weights = model.state_dict().copy()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model weights
    model.load_state_dict(best_model_weights)
    return model, history

# %%
# Evaluation function
def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    
    true_labels = []
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (photon)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    auc = roc_auc_score(true_labels, probabilities)
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Electron', 'Photon'], 
                yticklabels=['Electron', 'Photon'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': conf_matrix,
        'true_labels': true_labels,
        'predictions': predictions,
        'probabilities': probabilities
    }

# %% [markdown]
# ## Visualization Functions

# %%
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# %%
def visualize_samples(dataset, num_samples=5):
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3*num_samples))
    
    # Get random indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        data, label = dataset[idx]
        
        # Energy channel
        im0 = axes[i, 0].imshow(data[0], cmap='viridis')
        axes[i, 0].set_title(f"Sample {idx} - {'Photon' if label == 1 else 'Electron'} - Energy")
        axes[i, 0].axis('off')
        fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        # Time channel
        im1 = axes[i, 1].imshow(data[1], cmap='plasma')
        axes[i, 1].set_title(f"Sample {idx} - {'Photon' if label == 1 else 'Electron'} - Time")
        axes[i, 1].axis('off')
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# %%
def visualize_feature_maps(model, dataset, layer_name, device='cuda'):
    """
    Visualize feature maps from a specific layer of the model for a sample image
    """
    # Set model to evaluation mode
    model.eval()
    
    # Select a random sample
    idx = np.random.randint(0, len(dataset))
    input_img, label = dataset[idx]
    input_img = input_img.unsqueeze(0).to(device)  # Add batch dimension
    
    # Register hook to get intermediate activations
    activations = {}
    
    def hook_fn(module, input, output):
        activations[layer_name] = output.detach().cpu()
    
    # Register hook
    for name, module in model.named_modules():
        if name == layer_name:
            hook = module.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        output = model(input_img)
    
    # Remove hook
    hook.remove()
    
    # Get the feature maps
    feature_maps = activations[layer_name][0]  # First sample in batch
    num_features = min(16, feature_maps.size(0))  # Display at most 16 feature maps
    
    # Plot the feature maps
    fig, axes = plt.subplots(int(np.ceil(num_features/4)), 4, figsize=(12, 3*int(np.ceil(num_features/4))))
    fig.suptitle(f"Feature maps from {layer_name} for {'Photon' if label == 1 else 'Electron'}", fontsize=16)
    
    for i in range(num_features):
        ax = axes[i//4, i%4] if num_features > 4 else axes[i]
        im = ax.imshow(feature_maps[i], cmap='viridis')
        ax.set_title(f"Filter {i}")
        ax.axis('off')
    
    # Hide any empty subplots
    for i in range(num_features, int(np.ceil(num_features/4))*4):
        if i < int(np.ceil(num_features/4))*4:
            axes[i//4, i%4].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

# %% [markdown]
# ## Main Execution

# %%
def main():
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set correct paths to data
    electron_path = os.path.join('datasets', 'SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5')
    photon_path = os.path.join('datasets', 'SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5')
    
    # Create data directory if it doesn't exist
    if not os.path.exists('datasets'):
        os.makedirs('datasets')
        print("Created datasets directory")
    
    # Check if data files exist
    if not os.path.exists(electron_path):
        print(f"Error: Electron data file not found at {electron_path}")
        return
    if not os.path.exists(photon_path):
        print(f"Error: Photon data file not found at {photon_path}")
        return
        
    print(f"Using electron data from: {electron_path}")
    print(f"Using photon data from: {photon_path}")
    
    # Create dataset
    full_dataset = ParticleDataset(electron_path, photon_path)
    
    # Visualize some samples
    visualize_samples(full_dataset)
    
    # Split dataset into train (80%) and test (20%) sets
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Further split train set into train and validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = ResNet15(num_classes=2).to(device)
    print(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Train model
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=50,
        device=device
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model on test set
    test_results = evaluate_model(trained_model, test_loader, device)
    
    # Visualize feature maps
    visualize_feature_maps(trained_model, test_dataset, 'layer3.1.conv2', device)
    
    # Save model weights
    torch.save(trained_model.state_dict(), 'resnet15_electron_photon_classifier.pth')
    print("Model saved to resnet15_electron_photon_classifier.pth")
    
    # Save test results
    results_df = pd.DataFrame({
        'true_labels': test_results['true_labels'],
        'predictions': test_results['predictions'],
        'probabilities': test_results['probabilities']
    })
    results_df.to_csv('test_results.csv', index=False)
    print("Test results saved to test_results.csv")
    
    return trained_model, test_results

# %% [markdown]
# ## Experiments

# %%
def augmentation_experiment():
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set paths to data
    electron_path = 'electron_data.h5'  # Update with actual path
    photon_path = 'photon_data.h5'      # Update with actual path

    # Basic dataset without augmentation
    basic_dataset = ParticleDataset(electron_path, photon_path)
    
    # Define augmentation transforms
    class RandomNoise(object):
        def __init__(self, noise_level=0.05):
            self.noise_level = noise_level
        
        def __call__(self, sample):
            noise = np.random.normal(0, self.noise_level, sample.shape)
            noisy_sample = sample + noise
            return noisy_sample
    
    class RandomFlip(object):
        def __call__(self, sample):
            if np.random.random() > 0.5:
                return np.flip(sample, axis=1)  # Flip horizontally
            return sample
    
    class RandomRotate90(object):
        def __call__(self, sample):
            k = np.random.randint(0, 4)  # 0, 1, 2, or 3 times 90 degrees
            return np.rot90(sample, k=k, axes=(1, 2))
    
    # Create augmented dataset
    class AugmentedParticleDataset(ParticleDataset):
        def __init__(self, electron_path, photon_path):
            super().__init__(electron_path, photon_path)
            self.transform = transforms.Compose([
                RandomNoise(),
                RandomFlip(),
                RandomRotate90()
            ])
        
        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            
            sample = self.data[idx]
            label = self.labels[idx]
            
            # Apply data augmentation
            if self.transform:
                sample = self.transform(sample)
            
            # Convert to torch tensors
            sample = torch.from_numpy(sample).float()
            label = torch.tensor(label).long()
            
            return sample, label
    
    # Create augmented dataset
    augmented_dataset = AugmentedParticleDataset(electron_path, photon_path)
    
    # Split datasets
    train_size = int(0.8 * len(basic_dataset))
    test_size = len(basic_dataset) - train_size
    basic_train, basic_test = random_split(basic_dataset, [train_size, test_size])
    augmented_train, augmented_test = random_split(augmented_dataset, [train_size, test_size])
    
    # Create dataloaders
    batch_size = 64
    basic_train_loader = DataLoader(basic_train, batch_size=batch_size, shuffle=True, num_workers=4)
    basic_val_loader = DataLoader(basic_test, batch_size=batch_size, shuffle=False, num_workers=4)
    augmented_train_loader = DataLoader(augmented_train, batch_size=batch_size, shuffle=True, num_workers=4)
    augmented_val_loader = DataLoader(augmented_test, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Train model without augmentation
    basic_model = ResNet15(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(basic_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    basic_model, basic_history = train_model(
        model=basic_model,
        train_loader=basic_train_loader,
        val_loader=basic_val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=30,
        device=device
    )
    
    # Train model with augmentation
    augmented_model = ResNet15(num_classes=2).to(device)
    optimizer = optim.Adam(augmented_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    augmented_model, augmented_history = train_model(
        model=augmented_model,
        train_loader=augmented_train_loader,
        val_loader=augmented_val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=30,
        device=device
    )
    
    # Compare results
    plt.figure(figsize=(12, 5))
    
    # Plot validation loss
    plt.subplot(1, 2, 1)
    plt.plot(basic_history['val_loss'], label='Without Augmentation')
    plt.plot(augmented_history['val_loss'], label='With Augmentation')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(basic_history['val_acc'], label='Without Augmentation')
    plt.plot(augmented_history['val_acc'], label='With Augmentation')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("Final validation accuracy without augmentation:", basic_history['val_acc'][-1])
    print("Final validation accuracy with augmentation:", augmented_history['val_acc'][-1])

# %%
def hyperparameter_tuning():
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set paths to data
    electron_path = 'electron_data.h5'  # Update with actual path
    photon_path = 'photon_data.h5'      # Update with actual path
    
    # Create dataset
    full_dataset = ParticleDataset(electron_path, photon_path)
    
    # Split dataset into train (80%) and test (20%) sets
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Further split train set into train and validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Define hyperparameter combinations
    learning_rates = [0.01, 0.001, 0.0001]
    weight_decays = [1e-4, 1e-5, 1e-6]
    dropout_rates = [0.2, 0.3, 0.4]
    
    # Store results
    results = []
    
    # Grid search
    for lr in learning_rates:
        for wd in weight_decays:
            for dr in dropout_rates:
                print(f"Training with lr={lr}, weight_decay={wd}, dropout={dr}")
                
                # Create model with specified dropout rate
                class ResNet15WithDropout(ResNet15):
                    def __init__(self, num_classes=2, dropout_rate=0.3):
                        super().__init__(num_classes)
                        self.dropout = nn.Dropout(dropout_rate)
                
                model = ResNet15WithDropout(num_classes=2, dropout_rate=dr).to(device)
                
                # Define loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
                
                # Train model for fewer epochs for quicker tuning
                trained_model, history = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    num_epochs=20,  # Reduced epochs for quicker tuning
                    device=device
                )
                
                # Evaluate on validation set
                trained_model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = trained_model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                val_accuracy = val_correct / val_total
                
                # Store results
                results.append({
                    'learning_rate': lr,
                    'weight_decay': wd,
                    'dropout_rate': dr,
                    'val_accuracy': val_accuracy,
                    'final_val_loss': history['val_loss'][-1]
                })
    
    # Convert results to DataFrame and find best hyperparameters
    results_df = pd.DataFrame(results)
    best_config = results_df.loc[results_df['val_accuracy'].idxmax()]
    
    print("\nHyperparameter Tuning Results:")
    print(results_df.sort_values('val_accuracy', ascending=False))
    print("\nBest Configuration:")
    print(best_config)
    
    return best_config

# %%
def create_ensemble_model():
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set paths to data
    electron_path = 'electron_data.h5'  # Update with actual path
    photon_path = 'photon_data.h5'      # Update with actual path
    
    # Create dataset
    full_dataset = ParticleDataset(electron_path, photon_path)
    
    # Split dataset into train (80%) and test (20%) sets
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Create test loader
    batch_size = 64
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create 3 models with different architectures
    class ResNet15Wide(ResNet15):
        def __init__(self, num_classes=2):
            super(ResNet15, self).__init__()
            self.in_channels = 64
            
            # Initial convolution - wider channels
            self.conv1 = nn.Conv2d(2, 96, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(96)
            self.relu = nn.ReLU(inplace=True)
            
            # Residual blocks - wider channels
            self.layer1 = self._make_layer(96, 2, stride=1)
            self.layer2 = self._make_layer(192, 2, stride=2)
            self.layer3 = self._make_layer(384, 2, stride=2)
            
            # Average pooling and fully connected layer
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(384, num_classes)
            self.dropout = nn.Dropout(0.3)
    
    class ResNet15Deep(ResNet15):
        def __init__(self, num_classes=2):
            super(ResNet15, self).__init__()
            self.in_channels = 64
            
            # Initial convolution
            self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            
            # Residual blocks - deeper architecture
            self.layer1 = self._make_layer(64, 3, stride=1)  # One more block
            self.layer2 = self._make_layer(128, 3, stride=2)  # One more block
            self.layer3 = self._make_layer(256, 2, stride=2)
            
            # Average pooling and fully connected layer
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(256, num_classes)
            self.dropout = nn.Dropout(0.3)
    
    # Load pre-trained models
    model1 = ResNet15(num_classes=2).to(device)
    model1.load_state_dict(torch.load('resnet15_model1.pth'))
    
    model2 = ResNet15Wide(num_classes=2).to(device)
    model2.load_state_dict(torch.load('resnet15_wide_model2.pth'))
    
    model3 = ResNet15Deep(num_classes=2).to(device)
    model3.load_state_dict(torch.load('resnet15_deep_model3.pth'))
    
    # Function to evaluate ensemble
    def evaluate_ensemble(models, test_loader, device):
        for model in models:
            model.eval()
        
        true_labels = []
        ensemble_probs = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Evaluating Ensemble"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Get probabilities from each model
                probs_list = []
                for model in models:
                    outputs = model(inputs)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    probs_list.append(probs)
                
                # Average the probabilities
                ensemble_prob = torch.stack(probs_list).mean(dim=0)
                
                true_labels.extend(labels.cpu().numpy())
                ensemble_probs.extend(ensemble_prob[:, 1].cpu().numpy())  # Probability of class 1 (photon)
        
        # Convert probabilities to predictions
        ensemble_preds = [1 if p > 0.5 else 0 for p in ensemble_probs]
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, ensemble_preds)
        precision = precision_score(true_labels, ensemble_preds)
        recall = recall_score(true_labels, ensemble_preds)
        f1 = f1_score(true_labels, ensemble_preds)
        auc = roc_auc_score(true_labels, ensemble_probs)
        conf_matrix = confusion_matrix(true_labels, ensemble_preds)
        
        print(f"Ensemble Test Accuracy: {accuracy:.4f}")
        print(f"Ensemble Precision: {precision:.4f}")
        print(f"Ensemble Recall: {recall:.4f}")
        print(f"Ensemble F1 Score: {f1:.4f}")
        print(f"Ensemble ROC AUC: {auc:.4f}")
        print("Ensemble Confusion Matrix:")
        print(conf_matrix)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': conf_matrix
        }
    
    # Evaluate ensemble
    ensemble_models = [model1, model2, model3]
    ensemble_results = evaluate_ensemble(ensemble_models, test_loader, device)
    
    return ensemble_results

# %% [markdown]
# ## Run Everything

# %%
if __name__ == "__main__":
    model, test_results = main()
    
    print("\n=== Data Augmentation Experiment ===")
    augmentation_experiment()
    
    print("\n=== Hyperparameter Tuning ===")
    best_hyperparams = hyperparameter_tuning()
    
    print("\n=== Model Ensemble ===")
    ensemble_results = create_ensemble_model()
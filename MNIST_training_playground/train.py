import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import os
import random
import numpy as np
from PIL import Image

class CNN(nn.Module):
    def __init__(self, channels):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels[0], channels[1], kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels[1], channels[2], kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(channels[2], channels[3], kernel_size=3),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(channels[3] * 1 * 1, channels[3]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(channels[3], 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def get_optimizer(optimizer_name, model_params):
    if optimizer_name == 'adam':
        return optim.Adam(model_params, lr=0.001)
    elif optimizer_name == 'sgd':
        return optim.SGD(model_params, lr=0.01, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model_params, lr=0.001)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def save_test_examples(images, labels, predictions, model_name):
    img_dir = f'static/test_images/{model_name}'
    os.makedirs(img_dir, exist_ok=True)
    
    for file in os.listdir(img_dir):
        if file.endswith('.png'):
            os.remove(os.path.join(img_dir, file))

    examples = []
    for i in range(min(10, len(images))):
        img = images[i].cpu().squeeze().numpy()
        img = ((1 - img) * 255).astype('uint8')
        pil_img = Image.fromarray(img)
        
        img_path = f'test_images/{model_name}/digit_{i}.png'
        pil_img.save(f'static/{img_path}')
        
        examples.append({
            'image_path': f'static/{img_path}',
            'label': int(labels[i]),
            'prediction': int(predictions[i].cpu())
        })
    
    with open(f'static/test_examples_{model_name}.json', 'w') as f:
        json.dump(examples, f)

class TqdmToLogger:
    def __init__(self, log_writer, model_id):
        self.log_writer = log_writer
        self.model_id = model_id
        self.last_msg = ""

    def write(self, buf):
        # Only write if the message is different from the last one
        # and contains actual progress information
        msg = buf.strip()
        if msg and msg != self.last_msg and ('%' in msg or 'it/s' in msg):
            self.log_writer(msg, self.model_id)
            self.last_msg = msg

    def flush(self):
        pass

def save_model(model, model_id, config):
    os.makedirs('static/models', exist_ok=True)
    # Save model weights
    torch.save(model.state_dict(), f'static/models/{model_id}.pth')
    # Save model configuration
    with open(f'static/models/{model_id}_config.json', 'w') as f:
        json.dump({'channels': config['channels']}, f)

def train_model(config, metrics_dict, model_id, log_writer):
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log_writer(f"Training {model_id} on {device}", model_id)
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Model setup
    model = CNN(config['channels']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(config['optimizer'], model.parameters())

    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        
        # Create custom progress bar that writes to our log
        progress_bar = tqdm(
            train_loader,
            desc=f'Epoch {epoch+1}/{config["epochs"]}',
            file=TqdmToLogger(log_writer, model_id),
            ncols=80,  # Fixed width for cleaner output
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        
        for i, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i + 1) % 50 == 0:
                accuracy = evaluate(model, test_loader, device)
                avg_loss = running_loss / 50
                metrics_dict['loss'].append(avg_loss)
                metrics_dict['accuracy'].append(accuracy * 100)
                # Only log metrics without progress bar interference
                if i + 1 < len(train_loader):  # Don't show metrics at epoch end
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'accuracy': f'{accuracy*100:.2f}%'
                    })
                running_loss = 0.0

    # Signal completion and save model
    log_writer(f"Training completed for {model_id}!", model_id)
    save_model(model, model_id, config)
    
    # Store the model globally
    if model_id == 'model1':
        global model1
        model1 = model
    else:
        global model2
        model2 = model

    # Save final test examples
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images = images.to(device)
        outputs = model(images)
        predictions = torch.max(outputs, 1)[1]
        
        indices = torch.randperm(len(images))[:10]
        save_test_examples(
            images[indices],
            labels[indices],
            predictions[indices],
            model_id
        )

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total 
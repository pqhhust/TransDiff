import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchbnn.models import SDENet
import torch.nn as nn
import time
from tqdm import tqdm

import wandb
import gc

import datasets.cifar_loader 

def main():
    epochs = 3
    batch_size = 128
    train_loader, val_loader, _, nb_cls = datasets.cifar_loader.get_loader(
        'cifar10', './data/CIFAR10/train', './data/CIFAR10/val', './data/CIFAR10/test', 128
    )
    # num_samples = 45000
    input_size = (3, 32, 32)
    
    # # Create dummy dataset with images and dummy labels.
    # dummy_data = torch.randn(num_samples, *input_size)
    # dummy_labels = torch.randint(0, 10, (num_samples,))  # assuming 10 classes.
    # dataset = TensorDataset(dummy_data, dummy_labels)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model.
    model = SDENet(inhomogeneous=False, input_size=input_size, aug_dim=1)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(model)
    # model.ts = torch.tensor([0., 1.])  # a tiny t1 to pass torchsde's check
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            logits, logqp = model(data, dt=0.1, adjoint=False, method='midpoint', adaptive=False, adjoint_adaptive=False, rtol=1e-5, atol=1e-4)
            loss = criterion(logits, target) + 1e-3 * logqp  # incorporate logqp if needed
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} Batch {batch_idx}: Loss = {loss.item():.4f}")
                gc.collect()

if __name__ == "__main__":
    wandb.login()
    wandb.init(project="SDEBNN")
    start_time = time.time()
    main()
    end_time = time.time()
    print(end_time - start_time)
    wandb.finish()
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms, models
from torch.utils.data import DataLoader

from models import SimpleCNN, SimpleResNet
from preprocess import AMVDataset, ToTensor, Shift
from utils import save_model

def gen_trainloader(root_dir:str, batch_size:int=8, ens=40, lead=0, num_workers:int=1) -> DataLoader:
    trainset = AMVDataset(root_dir=root_dir, ens=ens, lead=lead, transform=transforms.Compose([
        ToTensor(),
        Shift()
    ]))
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers)
    return trainloader


def train(model, trainloader, num_epochs, lr, optimizer, criterion,
    verbose=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    print_every=10
    ) -> list:

    print(f"Training {type(model)} model.")	            
    print("========== HYPERPARAMETERS ==========")
    print(f"num_epochs: {num_epochs}")	            
    print(f"lr: {lr}")	
    print(f"optimizer: {optimizer}")
    print(f"criterion: {criterion}")
    print(f"device: {device}")
    print("\n")


    model.to(device)
    model.train()

    train_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        for i, sample in enumerate(trainloader, 0):
            inputs = sample["data"].to(device)
            labels = sample["label"].to(device)
            optimizer.zero_grad()
            print(labels)

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            if verbose:
                running_loss += loss.item()
                if i % print_every == 0 and i != 0:  
                    print(f"[epoch: {epoch}, datapoint: {i}] \t loss: {round(running_loss / print_every, 3)}")
                    running_loss = 0.0
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(trainloader)) #this is buggy

    return train_losses


if __name__ == "__main__":
    model = SimpleCNN(num_classes=3)
    # model = models.resnet50(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False
    # model.fc = nn.Linear(in_features=model.fc.in_features, out_features=3)
    trainloader = gen_trainloader(root_dir="data", batch_size=32, num_workers=1)
    num_epochs = 2
    lr = 1e-4
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = train(
        model=model,
        trainloader=trainloader,
        num_epochs=num_epochs,
        lr=lr,
        optimizer=optimizer,
        criterion=criterion,
        verbose=True
    )
    print(losses)
    # save_model(model, "basic")

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms, models
from torch.utils.data import DataLoader

from models import SimpleCNN, SimpleResNet
from preprocess import AMVDataset, ToTensor, Shift
from utils import save_model, test_total_accuracy, test_class_accuracy

def gen_loader(istrain:bool, percent_train:float, root_dir:str, batch_size:int=8, ens=40, lead=0, num_workers:int=1) -> DataLoader:
    trainset = AMVDataset(istrain=istrain, percent_train=percent_train, root_dir=root_dir, ens=ens, lead=lead, transform=transforms.Compose([
        ToTensor(),
        Shift()
    ]))
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers)
    return trainloader


def train(model, trainloader, testloader, num_epochs, lr, optimizer, criterion,
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
    val_losses = []
    total_accuracies = []
    class_accuracies = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        for i, sample in enumerate(trainloader, 0):
            inputs = sample["data"].to(device)
            labels = sample["label"].to(device)
            optimizer.zero_grad()

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

        val_loss = validate(model, testloader, criterion)
        print(f"----- epoch: {epoch} round val loss: {round(val_loss, 3)} -----")
        val_losses.append(val_loss)
        total_accuracies.append(test_total_accuracy(model, testloader))
        class_accuracies.append(test_class_accuracy(model, testloader, num_classes=3))

    return train_losses, val_losses, total_accuracies, class_accuracies


def validate(model, testloader, criterion, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    val_loss = 0.0
    for i, sample in enumerate(testloader, 0):
        inputs = sample["data"].to(device)
        labels = sample["label"].to(device)

        prediction = model(inputs)
        #y_pred = np.argmax(prediction.detach().cpu().numpy(), axis=1)
        #y_label = labels.detach().cpu().numpy().squeeze()

        #accuracy = np.sum(y_pred == y_label)/y_label.shape[0]
        #accuracies = np.concatenate([accuracies, accuracy])
        val_loss += criterion(prediction, labels).item() / len(testloader)

    return val_loss


if __name__ == "__main__":
    model = SimpleCNN(num_classes=3)
    # model = models.resnet50(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False
    # model.fc = nn.Linear(in_features=model.fc.in_features, out_features=3)
    trainloader = gen_loader(
        istrain=True,
        percent_train=.9,
        root_dir="data",
        batch_size=32,
        num_workers=4
    )
    testloader = gen_loader(
        istrain=False,
        percent_train=.9,
        root_dir="data",
        batch_size=32,
        num_workers=4
    )
    num_epochs = 2
    lr = 1e-4
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, total_accuracies, class_accuracies = train(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        num_epochs=num_epochs,
        lr=lr,
        optimizer=optimizer,
        criterion=criterion,
        verbose=True
    )
    print("train loss:", train_losses)
    print("val losses:", val_losses)
    print("total acc:", total_accuracies)
    print("class acc:", class_accuracies)
    name = "SimpleResNet_20"
    os.makedirs(f"saved_models/{name}", exist_ok=True)
    np.save(f"saved_models/{name}/trainlosses", np.array(train_losses))
    np.save(f"saved_models/{name}/vallosses", np.array(val_losses))
    np.save(f"saved_models/{name}/totalaccuracies", np.array(total_accuracies))
    np.save(f"saved_models/{name}/classaccuracies", np.array(class_accuracies))
    save_model(model, f"{name}/{name}")

import os
import torch
import numpy as np

def save_model(model, name):
    save_dir = os.path.join("saved_models", name)
    torch.save(model.state_dict(), os.path.join("saved_models", f"{name}.pkl"))


def load_model(model, name):
    model.load_state_dict(torch.load(os.path.join("saved_models", f"{name}.pkl")))
    return model

def test_total_accuracy(model, testloader, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    compute the (pure) accuracy over a test set
    Args:
        model (torch.nn.Module): [description]
        testloader (torch.utils.data.Dataloader): the test set dataloader
        device (str or pytorch device, optional): where to evaluate pytorch variables. Defaults to "cpu".
    Returns:
        (float): the accuracy
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for sample in testloader:
            inputs, labels = sample["data"].to(device), sample["label"].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            print("all ones?", np.all(predicted.cpu().numpy() == 1))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def test_class_accuracy(model, testloader, num_classes, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    compute (pure) accuracy per class in the test set
    Args:
        model (torch.nn.Module): the model to evaluate
        testloader (torch.utils.data.Dataloader): the test set dataloader
        device (str or pytorch device, optional): where to evaluate pytorch variables. Defaults to "cpu".
    """
    class_correct = np.array([0. for i in range(num_classes)])
    class_total = np.array([0. for i in range(num_classes)])
    with torch.no_grad():
        for sample in testloader:
            inputs, labels = sample["data"].to(device), sample["label"].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(num_classes):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    return class_correct / class_total

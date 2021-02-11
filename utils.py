import os
import torch

def save_model(model, name):
    save_dir = os.path.join("saved_models", name)
    torch.save(model.state_dict(), os.path.join("saved_models", f"{name}.pkl"))


def load_model(model, name):
    model.load_state_dict(torch.load(os.path.join("saved_models", f"{name}.pkl")))
    return model

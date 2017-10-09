import torch

def freeze_layers(model, n_layers):
    i = 0
    for child in model.children():
        if i >= n_layers:
            break
        print(i, "freezing", child)
        for param in child.parameters():
            param.requires_grad = False
        i += 1

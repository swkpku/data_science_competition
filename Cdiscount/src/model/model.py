import torch

class AssembledModel(torch.nn.Module):
    def __init__(self, model, classifier):
        super().__init__()
        self.__class__.__name__ = "AssembledModel"
        self.model = model
        self.classifier = classifier

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def assemble_model(model, cut, fin, num_classes):
    # cut the classifier layer
    model = torch.nn.Sequential(*list(model.children())[:cut])
    
    # create a new classifier
    classifier_layers = [
        torch.nn.Linear(in_features=fin, out_features=num_classes),
    ]
    classifier = torch.nn.Sequential(*classifier_layers)
    
    # return the assembled model
    return AssembledModel(model, classifier)
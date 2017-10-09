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
    
class ConcatenateModel(torch.nn.Module):
    def __init__(self, model, layer, classifier):
        super().__init__()
        self.__class__.__name__ = "ConcatenateModel"
        self.model = model
        self.layer = layer
        self.classifier = classifier
        
    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
def cut_and_concatenate_model(model, cut, layer, fin, num_classes):
    model = torch.nn.Sequential(*list(model.children())[:cut])
    
    # create a last layer
    classifier_layers = [
        torch.nn.Linear(in_features=fin, out_features=num_classes),
    ]
    classifier = torch.nn.Sequential(*classifier_layers)
    
    # return the assembled model
    return ConcatenateModel(model, layer, classifier)
    

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

def assemble_model_with_classifier(model, cut, classifier):
    # cut the classifier layer
    model = torch.nn.Sequential(*list(model.children())[:cut])
    
    # return the assembled model
    return AssembledModel(model, classifier)
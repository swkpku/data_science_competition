import torch
import torchvision.models as models

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

def diff_states(dict_canonical, dict_subset):
    names1, names2 = (list(dict_canonical.keys()), list(dict_subset.keys()))
    
    #Sanity check that param names overlap
    #Note that params are not necessarily in the same order
    #for every pretrained model
    not_in_1 = [n for n in names1 if n not in names2]
    not_in_2 = [n for n in names2 if n not in names1]
    assert len(not_in_1) == 0
    assert len(not_in_2) == 0

    for name, v1 in dict_canonical.items():
        v2 = dict_subset[name]
        assert hasattr(v2, 'size')
        if v1.size() != v2.size():
            yield (name, v1)    

def load_model_merged(name, num_classes):
    model = models.__dict__[name](num_classes=num_classes)
    
    #Densenets don't (yet) pass on num_classes, hack it in
    if "densenet" in name:
        if name == 'densenet169':
            return models.DenseNet(num_init_features=64, growth_rate=32, \
                                   block_config=(6, 12, 32, 32), num_classes=num_classes)
        
        elif name == 'densenet121':
            return models.DenseNet(num_init_features=64, growth_rate=32, \
                                   block_config=(6, 12, 24, 16), num_classes=num_classes)
        
        elif name == 'densenet201':
            return models.DenseNet(num_init_features=64, growth_rate=32, \
                                   block_config=(6, 12, 48, 32), num_classes=num_classes)

        elif name == 'densenet161':
             return models.DenseNet(num_init_features=96, growth_rate=48, \
                                    block_config=(6, 12, 36, 24), num_classes=num_classes)
        else:
            raise ValueError("Cirumventing missing num_classes kwargs not implemented for %s" % name)
    
    pretrained_state = model_zoo.load_url(model_urls[name])

    #Diff
    diff = [s for s in diff_states(model.state_dict(), pretrained_state)]
    print("Replacing the following state from initialized", name, ":", \
          [d[0] for d in diff])
    
    for name, value in diff:
        pretrained_state[name] = value
    
    assert len([s for s in diff_states(model.state_dict(), pretrained_state)]) == 0
    
    #Merge
    model.load_state_dict(pretrained_state)
    return model, diff
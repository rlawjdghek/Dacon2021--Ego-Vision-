import torch.nn as nn
import timm

class SAModels(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = timm.create_model(args.pt, pretrained=True)
        modules = list(self.model.children())[:-1]
        self.model = nn.Sequential(*modules)
        if args.pt == 'resnet34':
            self.fc = nn.Linear(512, 157)
        else:
            self.fc = nn.Linear(2048, 157)

    def forward(self, x):
        output = self.model(x)
        output = self.fc(output) # (bs, 1)
        return output
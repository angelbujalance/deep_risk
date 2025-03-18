import torch
from torchvision.models import restnet50

class RestNet50(torch.nn.Module):
    def __init__(self, in_channels:int=2, return_latent_space:bool=False, pretrained:bool=False, contrastive_learning:bool=False):
        super(ResNet50, self).__init__()

        # if True, returns latent space for contrastive learning
        self.return_latent_space = return_latent_space

        self.model = restnet50(pretrained=pretrained)

        # Modify first layer to handle different input channels
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Freeze backbone during contrastive learning
        if contrastive_learning:
            for param in self.model.parameters():
                param.requires_grad = False

        # Remove the final classification layer (keep feature extractor)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])

        # Adaptive Pooling for variable input sizes
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected layer (latent space)
        self.fc = nn.Linear(2048, latent_dim)

        # Output layer for regression (num_outputs = number of heart-related features)
        self.output_layer = nn.Linear(latent_dim, num_outputs)
        
    def forward(self, x):

        x = self.model(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)  # Latent space representation

        return x if self.return_latent_space else self.output_layer(x)
    
    def save_checkpoint():
        """TODO"""
    
    def load_checkpoint():
        """"TODO"""

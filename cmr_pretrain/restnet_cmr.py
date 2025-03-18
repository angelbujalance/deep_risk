import torch
from torchvision.models import restnet50

class RestNet50(torch.nn.Module):
    def __init__(return_latent_space: bool = False):
        super(ResNet50, self).__init__()

        # if True, returns latent space for contrastive learning
        self.return_latent_space = return_latent_space

        self.model = restnet50(pretrained=pretrained)
    
    def save_checkpoint():
        """TODO"""
    
    def load_checkpoint():
        """"TODO"""

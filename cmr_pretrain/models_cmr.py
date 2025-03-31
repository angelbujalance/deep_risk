import torch
from torch import nn
from torchvision.models import resnet50
from abc import ABC, abstractmethod


class CMRModel(torch.nn.Module, ABC):
    def __init__(self, input_size:int=2, return_latent_space:bool=False,
                 pretrained:bool=False, contrastive_learning:bool=False,
                 latent_dim:int=200, num_outputs:int=1, args=None):
        super().__init__()
        # Explicitly convert to Python native types
        self.input_size = int(input_size)
        self.return_latent_space = bool(return_latent_space)
        self.pretrained = bool(pretrained)
        self.contrastive_learning = bool(contrastive_learning)

        # Ensure these are proper integers
        if latent_dim is None:
            self.latent_dim = 200
        else:
            try:
                self.latent_dim = int(latent_dim)
            except (ValueError, TypeError):
                print(f"Warning: Invalid latent_dim value '{latent_dim}', using default 200")
                self.latent_dim = 200
        
        if num_outputs is None:
            self.num_outputs = 1
        else:
            try:
                self.num_outputs = int(num_outputs)
            except (ValueError, TypeError):
                print(f"Warning: Invalid num_outputs value '{num_outputs}', using default 1")
                self.num_outputs = 1
        
        self.args = args

        # Print debug info
        print(f"Initializing CMRModel with latent_dim={self.latent_dim}, num_outputs={self.num_outputs}")

        self.model = self.initialize_model(pretrained=self.pretrained)

        # Adaptive Pooling for variable input sizes
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Get the correct feature dimension from the backbone
        self.feature_dim = self._get_feature_dim()

        # Fully Connected layer (latent space) - corrected this to use latent_dim
        self.fc = nn.Linear(self.feature_dim, self.latent_dim)

        # Output layer for regression
        self.output_layer = nn.Linear(self.latent_dim, self.num_outputs)

    def forward(self, x):
        x = self.model(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)  # Latent space representation

        return x if self.return_latent_space else self.output_layer(x)

    # Abstract methods that are implemented in subclasses
    @abstractmethod
    def initialize_model(self, pretrained: bool):
        """It is not possible to initialize the base model.
            Must be implemented in the model's subclass."""
        pass

    @abstractmethod
    def _get_feature_dim(self):
        """Helper method to determine the feature dimension from the backbone"""
        pass


class ResNet50(CMRModel):
    def initialize_model(self, pretrained: bool):
        # Get ResNet50 model
        if pretrained:
            model = resnet50(weights='DEFAULT')
        else:
            model = resnet50()

        # Modify first layer to handle different input channels
        model.conv1 = nn.Conv2d(self.input_size, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Freeze backbone during contrastive learning
        if self.contrastive_learning:
            for param in model.parameters():
                param.requires_grad = False

        # For ResNet50, we'll keep the full model until the final FC layer
        # This approach keeps the right architecture for feature extraction
        return nn.Sequential(*list(model.children())[:-2])  # Remove avg pool and fc
    
    def _get_feature_dim(self):
        """Return the feature dimension for ResNet50"""
        return 2048  # ResNet50 always outputs 2048 features


class SwinTransformer(CMRModel):
    def initialize_model(self, pretrained:bool):
        # TODO: Implement
        raise NotImplementedError("SwinTransformer implementation is incomplete")

    def _get_feature_dim(self):
        """Override to provide the correct feature dimension for Swin model"""
        return 768  # Example for Swin-T


def get_model(model_name, args):
    # Debug the input args
    print(f"Creating model: {model_name}")
    
    try:
        # Validate that critical args are not None and are of expected types
        input_size = int(args.input_size) # if args.input_size is not None else 2
        latent_dim = int(args.latent_dim) # if args.latent_dim is not None else 200
        num_outputs = int(args.num_outputs) # if args.num_outputs is not None else 1
        
        print(f"Args: input_size={input_size}, latent_dim={latent_dim}, num_outputs={num_outputs}")
        
        if model_name == "ResNet50":
            return ResNet50(
                input_size=input_size,
                return_latent_space=args.return_latent_space,
                pretrained=args.pretrained,
                contrastive_learning=args.contrastive_learning,
                latent_dim=latent_dim,
                num_outputs=num_outputs,
                args=args
            )
        elif model_name == "SwinTransformer":
            return SwinTransformer(
                input_size=input_size,
                return_latent_space=args.return_latent_space,
                pretrained=args.pretrained,
                contrastive_learning=args.contrastive_learning,
                latent_dim=latent_dim,
                num_outputs=num_outputs,
                args=args
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    except Exception as e:
        print(f"Error creating model: {e}")
        # Provide fallback with hardcoded values if there's an error
        if model_name == "ResNet50":
            print("Using fallback values for ResNet50")
            return ResNet50(
                input_size=2,
                return_latent_space=False,
                pretrained=False,
                contrastive_learning=False,
                latent_dim=200,
                num_outputs=1,
                args=args
            )
        else:
            raise
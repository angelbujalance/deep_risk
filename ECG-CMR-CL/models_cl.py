import torch
from torch import nn
from torchvision.models import resnet50
from abc import ABC, abstractmethod
from lightly.models.modules import SimCLRProjectionHead
import os
from ECGEncoder import ECGEncoder, ECGEncoder_dict
from collections import OrderedDict

from utils.clip_loss import CLIPLoss
from cmr_pretrain.models_cmr import get_model
from cmr_pretrain.engine_cmr import load_checkpoint, load_resnet50, load_resnet50_3D


class ClipContrastiveLearning(torch.nn.Module, ABC):
    def __init__(self, input_size:int=2, return_latent_space:bool=True,
                 pretrained:bool=False, contrastive_learning:bool=True,
                 latent_dim:int=200, device:str='cuda', args=None):
        super().__init__()
        # Explicitly convert to Python native types
        self.input_size = int(args.input_size)
        self.return_latent_space = bool(return_latent_space)
        self.pretrained = bool(pretrained)
        self.contrastive_learning = bool(contrastive_learning)
        self.latent_dim = latent_dim
        self.args = args
        self.num_outputs = 0

        # Print debug info
        print(f"Initializing CMRModel with latent_dim={self.latent_dim}")

        # model_name, input_size, latent_dim, num_outputs, args
        print("Type args", type(args))
        cmr_model = get_model(model_name=self.args.model_name,
                                     args=self.args)

        if os.path.exists(self.args.checkpoint_path_cmr):
            checkpoint = torch.load(self.args.checkpoint_path_cmr, map_location=device)
            
            if self.args.model_name in ["ResNet50", "ResNet50-3D"]:
                print("loading ResNet50 for 2D image inputs...")
                load_resnet50(cmr_model=cmr_model, checkpoint_path_cmr=self.args.checkpoint_path_cmr, device=device)
            elif self.args.model_name == "ResNet50-4D":
                load_resnet50_3D(cmr_model=cmr_model, checkpoint_path_cmr=self.args.checkpoint_path_cmr, device=device)
            else:
                checkpoint = torch.load(self.args.checkpoint_path_cmr, map_location=device)
                print(f"Model keys: {checkpoint.keys()}")
                cmr_model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Model {self.args.model_name} succesfully loaded from checkpoint")
        else:
            print("No pre-trained CMR encoder loaded")

        # Get the correct feature dimension from the backbone to use in the projection head
        self.cmr_feature_dim = cmr_model._get_feature_dim()

        self.cmr_encoder = cmr_model.encoder

        if cmr_model.lstm:
            self.lstm = cmr_model.lstm

        # Apply Average Pooling to the CMR encoder
        if self.args.model_name == "ResNet50-4D":
            self.cmr_encoder = nn.Sequential(*list(self.cmr_encoder.children()), 
                                            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)))
        elif self.args.model_name == "ResNet50-3D":
            self.cmr_encoder = cmr_model
        else:
            # In 2D resnet, temporal dimension is not provided
            self.cmr_encoder = nn.Sequential(*list(self.cmr_encoder.children()), 
                                            nn.AdaptiveAvgPool2d(output_size=(1, 1))) # or 

        print("self.cmr_feature_dim", self.cmr_feature_dim)
        print("self.args.projection_dim", self.args.projection_dim)
        self.projection_head_cmr = SimCLRProjectionHead(self.cmr_feature_dim,
                                    self.cmr_feature_dim, self.args.projection_dim)
        
        print("args.ecg_input_size", args.ecg_input_size)
        print("args.drop_path", args.drop_path)
        self.ecg_encoder = ECGEncoder_dict[args.ecg_model](
            img_size=self.args.ecg_input_size,
            # patch_size=self.args.patch_size,
            in_chans=self.input_size, # 1, corresponds to the channels in the ECG
            # num_classes=args.num_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool)

        # retrieve the pre-trained model from a checkpoint
        model_checkpoint = {}
        if os.path.exists(self.args.checkpoint_path_ecg):
            checkpoint = torch.load(self.args.checkpoint_path_ecg, map_location=device, weights_only=False)
            print("checkpoint.keys", list(checkpoint.keys()))
            for key, value in checkpoint["model"].items():
                # Get the encoder part of the ViT MAE pre-trained model
                if not any(key.startswith(prefix) for prefix in ['decoder', 'mask_token', 'predictor', 'projector']):
                    model_checkpoint[key] = value

            # "head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias" are missing, and initialized
            self.ecg_encoder.load_state_dict(model_checkpoint, strict=False)
        else:
            print("No pre-trained ECG encoder loaded")

        # required to read out the attention map of the last layer
        self.ecg_encoder.blocks[-1].attn.forward = self._attention_forward_wrapper(self.ecg_encoder.blocks[-1].attn)

        # print("self.ecg_encoder.embed_dim", self.ecg_encoder.embed_dim) # 768 
        # print("self.args.projection_dim", self.args.projection_dim) # 128

        if self.args.model_name == "ResNet50-3D":
            self.projection_head_cmr = SimCLRProjectionHead(self.cmr_encoder.latent_dim,
                                self.cmr_encoder.latent_dim, self.args.projection_dim)

        # Manually established the embed dimensions to match the ECG encoder output
        self.projection_head_ecg = SimCLRProjectionHead(1000,
                                                        1000, self.args.projection_dim)

    def forward(self, x_cmr, x_ecg):

        # https://github.com/oetu/MMCL-ECG-CMR/blob/main/mmcl/models/MultimodalSimCLR.py

        # the encoder returns the latent representations
        cmr_features = self.cmr_encoder(x_cmr)  # Shape: torch.Size([20, 2048, 1, 1])

        # print("cmr_features succesful")

        # Reshape/flatten the CMR features
        # print("cmr_features:", cmr_features.shape)  # Shape: torch.Size([64, 102400]) --> 2048 x 50
        cmr_features = cmr_features.view(cmr_features.size(0), -1)  # Shape: torch.Size([20, 2048])

        # print("cmr_features (II):", cmr_features.shape)
        cmr_proj = self.projection_head_cmr(cmr_features)

        # print("cmr_proj:", cmr_proj.shape)
        ecg_features = self.ecg_encoder(x_ecg)  # Shape: torch.Size([20, 1000])
        ecg_proj = self.projection_head_ecg(ecg_features)

        return cmr_proj, ecg_proj

    def _attention_forward_wrapper(self, attn_obj):
        """
        Adapted from: https://github.com/oetu/MMCL-ECG-CMR/blob/main/mmcl/models/ECGSimCLR.py
        Modified version of def forward() of class Attention() in timm.models.vision_transformer
        """
        def my_forward(x):
            B, N, C = x.shape # C = embed_dim

            # (3, B, Heads, N, head_dim)
            qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

            # (B, Heads, N, N)
            attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
            attn = attn.softmax(dim=-1)
            attn = attn_obj.attn_drop(attn)
            # (B, Heads, N, N)
            attn_obj.attn_map = attn # this was added 

            # (B, N, Heads*head_dim)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = attn_obj.proj(x)
            x = attn_obj.proj_drop(x)
            return x
        return my_forward

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
        self.args = args

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
        if self.args.model_name == "ResNet50-4D":
            self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Get the correct feature dimension from the backbone
        self.feature_dim = self._get_feature_dim()

        # Fully Connected layer (latent space) - corrected this to use latent_dim
        self.fc = nn.Linear(self.feature_dim, self.latent_dim)

        # Output layer for regression
        self.output_layer = nn.Linear(self.latent_dim, self.num_outputs)

    def forward(self, x):

        # print("self:", self)
        # exit()

        if self.args.model_name == "ResNet50":
            x = self.model(x)
        else:
            encoded_frames = []
            timepoints = x.shape[2]

            # encode individually the slices for each time point
            for i in range(timepoints):
                frame = x[:, :, i] # Shape: (batch_size, channels, height, width)

                encoded_frame = self.model(frame) # Encode each frame
                encoded_frames.append(encoded_frame)

            x = torch.stack(encoded_frames, dim=1)

        x = self.global_avg_pool(x)

        if self.lstm:
            x = torch.flatten(x, 2)

            x, _ = self.lstm(x)
            # select the last step from the lstm
            x = x[:,-1]

        else:
            x = torch.flatten(x, 1)

        # Latent space representation
        x = self.fc(x)

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
            model = resnet50(weights='IMAGENET1K_V1') # weights='IMAGENET1K_V1'
        else:
            model = resnet50()

        # Modify first layer to handle different input channels
        model.conv1 = nn.Conv2d(self.input_size, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove avg pool and fc to get the CMR encoder
        self.encoder = nn.Sequential(*list(model.children())[:-2]) 
        # Freeze backbone during contrastive learning
        if self.contrastive_learning:
            for param in model.parameters():
                param.requires_grad = False
            return model

        # For ResNet50, we'll keep the full model until the final FC layer
        # This approach keeps the right architecture for feature extraction
        return self.encoder

    def _get_feature_dim(self):
        """Return the feature dimension for ResNet50"""
        return 2048  # ResNet50 always outputs 2048 features


class ResNet503D(CMRModel):
    def initialize_model(self, pretrained: bool):
        # Get ResNet50 model for 3D data          
        if pretrained:
            model = resnet50(weights='IMAGENET1K_V1') # weights='IMAGENET1K_V1'
        else:
            model = resnet50()

        # Modify first layer to handle different input channels
        model.conv1 = nn.Conv2d(self.input_size, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove avg pool and fc to get the CMR encoder
        self.encoder = nn.Sequential(*list(model.children())[:-2])

        self.lstm_layers = 1

        # LSTM on top of the encoder to process the temporal dimension of the data
        self.lstm = nn.LSTM(input_size=self._get_feature_dim(), #//self.timepoints * 2,
                    hidden_size=self._get_feature_dim(),
                    num_layers=self.lstm_layers, batch_first=True)

        print(self.encoder)

        # Freeze backbone during contrastive learning
        if self.contrastive_learning:
            for param in model.parameters():
                param.requires_grad = False

        # For ResNet50, we'll keep the full model until the final FC layer
        # This approach keeps the right architecture for feature extraction
        return self.encoder

    def _get_feature_dim(self):
        """Return the feature dimension for ResNet50"""
        return 2048  # ResNet50 always outputs 2048 features


class ResNet504D(CMRModel):
    def initialize_model(self, pretrained: bool):
        # Get ResNet50 model            
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=pretrained)

        # Modify first layer to handle different input channels
        print("self.input_size", self.input_size)

        # Modify the first convolution layer to match number of channels, grayscale images (MRI) 
        first_conv_layer = model.blocks[0].conv
        model.blocks[0].conv = nn.Conv3d(
            self.input_size, 
            first_conv_layer.out_channels,
            kernel_size=first_conv_layer.kernel_size,
            stride=first_conv_layer.stride,
            padding=first_conv_layer.padding,
            bias=False
        )

        # Remove last block, to return latent space
        self.encoder = nn.Sequential(*model.blocks[:-1])

        print(model)

        # Freeze backbone during contrastive learning
        if self.contrastive_learning:
            for param in model.parameters():
                param.requires_grad = False

        # For ResNet50, we'll keep the full model until the final FC layer
        # This approach keeps the right architecture for feature extraction
        return self.encoder

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

    print(f"Trying args: input_size={args.input_size}, latent_dim={args.latent_dim}, num_outputs={args.num_outputs}")

    try:
        # Validate that critical args are not None and are of expected types
        input_size = int(args.input_size)
        latent_dim = int(args.latent_dim)
        num_outputs = int(args.num_outputs)

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
        elif model_name == "ResNet50-3D":
            return ResNet503D(
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







# class TEMPCODER3D(nn.Module):
#     def __init__(self, time_shape=(None, 15, 10, 384, 256, 2), cfg=None):
#         super(TEMPCODER3D, self).__init__()
#         self.time_shape = time_shape
#         self.cfg = cfg
#         self.contrastive_training = False
#         self.double_head = False
#         self.timepoints = cfg['params']['timepoints']
#         self.mlp_layers = cfg['model_params']['mlp_layers']
#         self.n_classes = cfg['params']['n_classes']
#         self.mlp_dropout = cfg['train_params']['mlp_dropout']
#         self.backbone = cfg['train_params']['backbone']
#         self.end_dimension = cfg['train_params']['end_dimension']
#         self.pretrained = cfg['train_params']['pretrained']
#         self.bidirectional = cfg['train_params']['bidirectional']
#         self.embedding = False
#         self.max_pool = cfg['train_params']['max_pool']
#         self.pool_dropout = cfg['train_params']['pool_dropout']
#         self.transformer = False
#         self.lstm = cfg['train_params']['lstm']
#         self.lstm_hidden = cfg['model_params']['lstm_hidden']
#         self.lstm_layers = cfg['model_params']['lstm_layers']
#         self.lstm_last = cfg['model_params']['lstm_last_hidden']
#         # Shape of input (B, C, H, W, D). B - batch size, C - channels, H - height, W - width, D - depth
#         if self.double_head:
#             pass
#         else:
#             pass

#         #.to('cuda' if torch.cuda.is_available() else 'cpu')
#         # Output shape: torch.Size([2, 512]
#         #self.time_distributed = TimeDistributed(self.encoder_model)

#         if not self.max_pool and not self.lstm:
#             self.end_dimension = self.end_dimension*self.timepoints

#         # self.positional_embedding = PositionalEmbedding(sequence_length=self.end_dimension, output_dim=self.end_dimension)

#         if self.lstm:
#             self.lstm = nn.LSTM(input_size=self.end_dimension, #//self.timepoints * 2,
#                     hidden_size=self.lstm_hidden,
#                     num_layers=self.lstm_layers, batch_first=True)
#         if self.double_head:
#             self.lstm_dvf = nn.LSTM(input_size=self.end_dimension,
#                     hidden_size=self.lstm_hidden,
#                     num_layers=self.lstm_layers, batch_first=True)
#         if self.bidirectional: 
#             self.lstm = nn.LSTM(input_size=self.end_dimension,
#                     hidden_size=self.lstm_hidden,
#                     num_layers=self.lstm_layers,
#                     batch_first=True,
#                     bidirectional=True)
#         self.max_pool = nn.AdaptiveMaxPool2d((1,self.end_dimension)) if self.max_pool else None
#         self.dropout = nn.Dropout(self.pool_dropout)

#         mlp = []
#         #input_dim = self.lstm_hidden*self.timepoints if self.double_head else self.lstm_hidden*self.timepoints
#         if self.lstm and self.double_head:
#             input_dim = self.lstm_hidden*self.timepoints*2
#         elif self.lstm:
#             input_dim = self.lstm_hidden*self.timepoints
#         if self.bidirectional: 
#             input_dim = self.lstm_hidden*2*self.timepoints
#         elif self.double_head: 
#             input_dim = self.end_dimension*2
#         else: 
#             input_dim = self.end_dimension

#         if self.lstm_last: input_dim = self.lstm_hidden

#         for layer_size in self.mlp_layers:
#             mlp.append(nn.Linear(input_dim, layer_size))
#             mlp.append(nn.BatchNorm1d(layer_size))
#             mlp.append(nn.ReLU(inplace=True)) # Gelu?
#             mlp.append(nn.Dropout(self.mlp_dropout))
#             input_dim = layer_size
#             # Adding the final layer to output the desired number of classes
#             mlp.append(nn.Linear(input_dim, self.n_classes))
#             # mlp.append(nn.Sigmoid())

#         self.output_net = nn.Sequential(*mlp)


#     def forward(self, x, emb=None, epoch=None):
#         #if epoch and epoch == 0: print('DEBUG EMB:', emb.shape)
#         if len(x.shape) == 5:
#             x = x.unsqueeze(2)
#             batch, timepoints, channels, depth, height, width = x.shape

#         # Apply encoder model to each frame in the sequence


#         if not self.double_head:
#             encoded_frames = []
#         for i in range(timepoints):
#             # time_shape = (8, 15, 3, 384, 256 ,10)
#             frame = x[:, i] # Shape: (batch_size, channels, depth, height, width)
#             encoded_frame = self.encoder_model(frame) # Encode each frame
#             encoded_frames.append(encoded_frame)
#             x = torch.stack(encoded_frames, dim=1) # Shape: (batch_size, timepoints, encoded_dim)
#         if self.contrastive_training: emb = torch.stack([encoded_frames[0], encoded_frames[timepoints//2]], dim=0).detach() # Shape: (batch_size, timepoints, encoded_dim)
#         else:
#         #encoded_frames = torch.zeros(batch, timepoints, 2 * self.end_dimension, device=x.device)
#             encoded_frames = []
#         # encoded_frames_dvf = []
#         # for i in range(timepoints):
#         # frame = x[:, i, 0]
#         # #xpan dim to keep dimensions consistent
#         # frame = frame.unsqueeze(1)
#         # #print('FRAME SHAPE:', frame.shape)
#         # dvf = x[:, i, 1:]
#         # #print('DVF SHAPE:', dvf.shape)
#         # encoded_frame = self.encoder_model(frame)
#         # encoded_frame_dvf = self.encoder_model_dvf(dvf)
#         # #encoded_frames[:, i] = torch.cat((encoded_frame, encoded_frame_dvf), dim=1)
#         # #encoded_frames.append(torch.cat((encoded_frame, encoded_frame_dvf), dim=1))
#         # encoded_frames.append(encoded_frame)
#         # encoded_frames_dvf.append(encoded_frame_dvf)
#         # x = torch.stack(encoded_frames, dim=1)
#         # x_dvf = torch.stack(encoded_frames_dvf, dim=1)
#         #print('DEBUG X SHAPE AFTER ENCODER IN SOUBLE HEAD:', x.shape)
#         #print('DEBUG X_DVF SHAPE AFTER ENCODER IN SOUBLE HEAD:', x_dvf.shape)


#         #print('DEBUG:', encoded_frames.shape)

#         # print('DEBUG one encoding:', encoded_frames[0].shape)
#         # print('len encodingd:', len(encoded_frames))
#         # if self.transformer:
#         # x = self.positional_embedding(x)
#         # # print('DEBUG after positional encodings:', x.shape)
#         # x = self.transformer(x)
#         #print('DEBUG after transformer:', x.shape)
#         if self.lstm:
#         #print('DEBUG before lstm:', x.shape)
#             x, _= self.lstm(x) #In shape (bs, t, end_dimension) # Out Shape: (batch_size, timepoints, lstm_hidden)
#         #print('DEBUG after lstm:', x.shape)
#         # if self.double_head:
#         # x_dvf, _= self.lstm_dvf(x_dvf) #In shape (bs, t, end_dimension) # Out Shape: (batch_size, timepoints, lstm_hidden)
#         #print('DEBUG after lstm:', x_dvf.shape)
#         #print('DEBUG after lstm:', x.shape)
#         # if self.max_pool:
#         # x = self.max_pool(x).squeeze(1)
#         elif self.lstm_last:
#             x = x[:,-1] #(batch_size, timepoints, lstm_hidden) -> (batch_size, lstm_hidden)
#         # else:
#         # # rearrange to combine the time and feature dimensions
#         # x = rearrange(x, 'b t f -> b (t f)')
#         # if self.double_head:
#         # x_dvf = rearrange(x_dvf, 'b t f -> b (t f)')
#         # x = torch.cat((x, x_dvf), dim=1)
#         #print('DEBUG after concat doubke head:', x.shape)
#         #print('DEBUG after arrange: ', x.shape)
#         # # concatenate embeddings
#         # if self.embedding:
#         # if epoch and epoch == 0: print('DEBUG before concat:', x.shape)
#         # x = torch.cat((x, emb), dim=1)
#         # if epoch and epoch == 0: print('DEBUG after concat:', x.shape)

#         # x = self.dropout(x)
#         #print('DEBUG after dropout:', x)
#         #print('DEBUG after rearrange: ', x.shape)
#         # x = self.output_net(x)
#         #print('DEBUG after output_net:', x)
#         # print('DEBUG after output_net:', x.shape)

#         return x
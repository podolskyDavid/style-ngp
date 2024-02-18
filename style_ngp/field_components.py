import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F


class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        # Extracting the first 5 blocks of VGG16
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        return

    def forward(self, X):
        h = self.slice1(X)
        h = self.slice2(h)
        h = self.slice3(h)
        h = self.slice4(h)
        h = self.slice5(h)
        h_relu5_3 = h.mean(dim=1, keepdim=True).view(h.size(0), -1)  # Flatten the feature map
        # # TODO: potentially incorporate multiple feature maps from different layers
        return h_relu5_3


# class HyperMLP(nn.Module):
#     def __init__(self, in_dim, num_layers, layer_width, out_dim=None, activation=nn.ReLU(),
#                  out_activation=None):
#         super().__init__()
#         self.in_dim = in_dim
#         self.num_layers = num_layers
#         self.layer_width = layer_width
#         self.out_dim = out_dim if out_dim is not None else layer_width
#         self.activation = activation
#         self.out_activation = out_activation
#         # Calculate total parameter count
#         self.total_param_count = self._calculate_total_param_count()
#
#         # Print overview of whole config
#         print(f"MLP Configuration: {self.in_dim} -> {self.num_layers} * {self.layer_width} -> {self.out_dim}")
#         print(f"Total Parameters: {self.total_param_count}")
#         print(f"Activation: {self.activation}")
#         print(f"Output Activation: {self.out_activation}")
#
#     def _calculate_total_param_count(self):
#         # Include logic to calculate total parameters based on layer configurations
#         # Consider weights and biases for each layer
#         param_count = 0
#         # Example calculation here - adjust based on actual MLP configuration
#         for i in range(self.num_layers):
#             if i == 0:
#                 param_count += (self.in_dim + 1) * self.layer_width  # +1 for bias
#             else:
#                 input_dim = self.layer_width
#                 output_dim = self.out_dim if i == self.num_layers - 1 else self.layer_width
#                 param_count += (input_dim + 1) * output_dim  # +1 for bias
#         return param_count
#
#     def forward(self, x, params):
#         print(f"magnitude of params: {params.abs().mean()}")
#         print(f"smallest param: {params.min()}")
#         print(f"largest param: {params.max()}")
#
#         # Implement the logic to apply weights dynamically using torch.nn.functional
#         offset = 0
#         for i in range(self.num_layers):
#             # Determine the number of weights and biases for the current layer
#             input_dim = self.in_dim if i == 0 else self.layer_width
#             output_dim = self.out_dim if i == self.num_layers - 1 else self.layer_width
#             weight_count = input_dim * output_dim
#             bias_count = output_dim
#
#             # Extract and reshape weights and biases
#             layer_weights = params[:, offset:offset + weight_count].view(output_dim, input_dim)
#             layer_biases = params[:, offset + weight_count:offset + weight_count + bias_count].view(-1)
#             offset += weight_count + bias_count
#
#             # Convert tensors to the same data type
#             x = x.float()
#             layer_weights = layer_weights.float()
#             layer_biases = layer_biases.float()
#
#             # Apply the layer using functional API
#             x = F.linear(x, layer_weights, layer_biases)
#             if i < self.num_layers - 1:
#                 x = self.activation(x)
#
#         if self.out_activation is not None:
#             x = self.out_activation(x)
#
#         return x
#
#
# class HyperNetwork(nn.Module):
#     def __init__(self, params_target_net):
#         super().__init__()
#         self.hypernetwork = nn.Sequential(
#             # TODO: hard-coded input dim
#             nn.Linear(1280, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, params_target_net)
#         )
#
#     def forward(self, features):
#         # Concatenate the features along the feature dimension
#         features = torch.cat(features, dim=-1)
#         params = self.hypernetwork(features)
#         return params

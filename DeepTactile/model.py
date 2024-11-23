import torch
from torch import nn
from torch_geometric.nn import TAGConv
from tdlayer import *
import torch.nn.functional as F
from to_graph import *
from to_STMnistgraph import *
import numpy as np

# Hyperparameters
thresh = 0.5  # Neuronal threshold
lens = 0.5  # Hyperparameter for approximate function
decay = 0.2  # Decay rate

# CNN configuration
cfg_cnn = [
    (2, 64),
    (64, 128),
    (128, 256),
    (256, 128),
    (128, 64),
]

# Kernel size configuration
cfg_s = [39, 39]

# Fully connected layer configuration
cfg_fc = [128, 256]


class DeepSequential(nn.Sequential):
    """
    Custom Sequential class for graph-based input.
    Each module accepts the input tensor and the edge index.
    """

    def forward(self, input, edge_index):
        for module in self:
            input = module(input, edge_index)
        return input


class _DeepLayer(nn.Module):
    """
    Basic unit of a DeepBlock, consisting of graph convolutions,
    batch normalization, and LIF spiking activation.
    """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, device='cuda:0'):
        super(_DeepLayer, self).__init__()
        self.device = device
        self.norm1 = EBNorm(num_input_features)
        self.spike1 = LIFSpike()
        self.Conv1 = TF_Layer(TAGConv(num_input_features, bn_size * growth_rate, K=1), device=device)
        self.norm2 = EBNorm(bn_size * growth_rate)
        self.spike2 = LIFSpike()
        self.Conv2 = TF_Layer(TAGConv(bn_size * growth_rate, growth_rate, K=3), device=device)
        self.drop_rate = drop_rate

    def forward(self, x, edge_index):
        new_features = self.norm1(x)
        new_features = self.spike1(new_features)
        new_features = self.Conv1(new_features, edge_index)
        new_features = self.norm2(new_features)
        new_features = self.spike2(new_features)
        new_features = self.Conv2(new_features, edge_index)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], dim=2)


class _DeepBlock(nn.ModuleDict):
    """
    A block of multiple _DeepLayers, stacked sequentially.
    """
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, device='cuda:0'):
        super(_DeepBlock, self).__init__()
        self.device = device
        for i in range(num_layers):
            layer = _DeepLayer(
                num_input_features + i * growth_rate, growth_rate, bn_size,
                drop_rate=drop_rate, device=device
            )
            self.add_module(f"deeplayer{i + 1}", layer)

    def forward(self, input, edge_index):
        for layer in self.values():
            input = layer(input, edge_index)
        return input


class _DeepTransition(nn.Module):
    """
    Transition layer between DeepBlocks to compress features.
    """
    def __init__(self, num_input_features, num_output_features, device="cuda:0"):
        super(_DeepTransition, self).__init__()
        self.device = device
        self.norm = EBNorm(num_input_features)
        self.spike = LIFSpike()
        self.Conv = TF_Layer(TAGConv(num_input_features, num_output_features, K=1), device=device)

    def forward(self, x, edge_index):
        x = self.norm(x)
        x = self.spike(x)
        x = self.Conv(x, edge_index)
        return x


class DeepTactile(nn.Module):
    """
    Main DeepTactile model architecture for tactile graph-based classification.
    """
    def __init__(self, growth_rate=32, block_config=(3, 3), num_init_features=64,
                 bn_size=4, compression_rate=0.5, drop_rate=0, num_classes=36, 
                 data_path=None, k=0, useKNN=False, device="cuda:0"):
        """
        Args:
            growth_rate: Number of filters in each _DeepLayer.
            block_config: Number of layers in each DeepBlock.
            num_init_features: Initial number of filters for the first convolution.
            bn_size: Bottleneck size for feature compression.
            compression_rate: Compression rate in transition layers.
            drop_rate: Dropout rate applied after layers.
            num_classes: Number of output classes.
            data_path: Path to data for graph construction.
            k: Number of neighbors for KNN graph.
            useKNN: Whether to use KNN for graph construction.
            device: Device to run the model on.
        """
        super(DeepTactile, self).__init__()
        self.device = device
        self.num_classes = num_classes

        # Graph constructor
        if data_path == 'ST-MNIST-SPLIT/':
            self.graph = TactileGraphSTMnist(k, useKNN=useKNN)
        else:
            self.graph = TactileGraph(k, useKNN=useKNN)

        # Initial convolution
        self.Conv0 = TF_Layer(TAGConv(2, num_init_features, K=1), device=device)
        self.norm0 = EBNorm(num_init_features)
        self.spike0 = LIFSpike()

        # Deep features and transition layers
        self.features = DeepSequential()
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DeepBlock(num_layers, num_features, bn_size, growth_rate, drop_rate, device=device)
            self.features.add_module(f"DeepBlock{i + 1}", block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                transition = _DeepTransition(num_features, int(num_features * compression_rate), device=device)
                self.features.add_module(f"Transition{i + 1}", transition)
                num_features = int(num_features * compression_rate)

        # Final normalization and spiking layer
        self.normL = EBNorm(num_features)
        self.spikeL = LIFSpike()

        # Fully connected layers
        self.fc1 = nn.Linear(100 * 176, cfg_fc[0])  # Adjust dimensions accordingly
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.fc3 = nn.Linear(cfg_fc[1], num_classes)

    def forward(self, input, training=True):
        data = input.to(self.device)
        sizes = data.size()
        time_window = sizes[-1]

        # Initialize hidden states
        h1_mem = h1_spike = h1_sumspike = torch.zeros(sizes[0], cfg_fc[0], device=self.device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(sizes[0], cfg_fc[1], device=self.device)
        h3_mem = h3_spike = h3_sumspike = torch.zeros(sizes[0], self.num_classes, device=self.device)

        # Graph construction
        graph_data = self.graph(data[..., 0])
        edge_index = graph_data.edge_index.to(self.device)

        # Initial convolution and deep feature extraction
        x = self.Conv0(data, edge_index)
        x = self.norm0(x)
        x = self.spike0(x)
        x = self.features(x, edge_index)
        x = self.normL(x)
        x = self.spikeL(x)
        x = x.split(1, dim=len(sizes) - 1)

        # Temporal processing
        for step in range(time_window):
            x_ = x[step].squeeze(-1).view(sizes[0], -1)
            h1_mem, h1_spike = mem_update_FC(self.fc1, x_, h1_mem, h1_spike)
            h1_sumspike += h1_spike
            h2_mem, h2_spike = mem_update_FC(self.fc2, h1_spike, h2_mem, h2_spike)
            h2_sumspike += h2_spike
            h3_mem, h3_spike = mem_update_FC(self.fc3, h2_spike, h3_mem, h3_spike)
            h3_sumspike += h3_spike

        # Output
        outputs = h3_sumspike / time_window
        return outputs


if __name__ == '__main__':
    net = DeepTactile(
        growth_rate=32, block_config=(3, 3), num_init_features=64,
        bn_size=4, compression_rate=0.5, drop_rate=0, num_classes=36,
        data_path='', k=0, useKNN=False, device='cuda:0'
    )
    print(net)

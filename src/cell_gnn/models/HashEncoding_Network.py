"""
HashEncodingMLP: InstantNGP-style hash encoding + MLP for signal external_input learning

Architecture:
    time (1D) → tcnn.Encoding (hash grid) → features → PyTorch MLP → n_output_dims

This hybrid approach:
- Uses tcnn's fast hash encoding for capturing temporal features
- Uses standard PyTorch linear layers to expand to many outputs (e.g., 1000 neurons)
- Allows gradients to flow through both components
"""

import torch
import torch.nn as nn

try:
    import tinycudann as tcnn
    TCNN_AVAILABLE = True
except ImportError:
    TCNN_AVAILABLE = False
    # print("Warning: tinycudann not available. HashEncodingMLP will not work.")


class HashEncodingMLP(nn.Module):
    """
    InstantNGP-style hash encoding + MLP for signal external_input learning.

    Maps time (1D normalized in [0, 1]) to n_output_dims outputs using:
    1. Multi-resolution hash grid encoding (tinycudann)
    2. PyTorch MLP to expand to desired output dimensions

    Note: tinycudann HashGrid requires at least 2D input, so 1D time input
    is padded to 2D by duplicating: [t] -> [t, t]

    Args:
        n_input_dims: Input dimensions (default: 1 for time)
        n_output_dims: Output dimensions (default: 1000 for neurons)
        n_levels: Number of hash grid levels (default: 24)
        n_features_per_level: Features per level (default: 2)
        log2_hashmap_size: Log2 of hash table size (default: 22)
        base_resolution: Base grid resolution (default: 16)
        per_level_scale: Resolution scale per level (default: 1.4)
        n_neurons: Hidden layer width for MLP (default: 128)
        n_hidden_layers: Number of hidden layers in MLP (default: 4)
        output_activation: Output activation ('none', 'sigmoid', 'tanh')
    """

    def __init__(
        self,
        n_input_dims=1,
        n_output_dims=1000,
        n_levels=24,
        n_features_per_level=2,
        log2_hashmap_size=22,
        base_resolution=16,
        per_level_scale=1.4,
        n_neurons=128,
        n_hidden_layers=4,
        output_activation='none'
    ):
        super().__init__()

        if not TCNN_AVAILABLE:
            raise ImportError(
                "tinycudann is required for HashEncodingMLP. "
                "Install it from: https://github.com/NVlabs/tiny-cuda-nn"
            )

        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims

        # tinycudann HashGrid requires at least 2D input
        # For 1D time input, we pad to 2D by duplicating: [t] -> [t, t]
        self.tcnn_input_dims = max(2, n_input_dims)
        self.pad_1d_to_2d = (n_input_dims == 1)

        # Store config for reference
        self.encoding_config = {
            "otype": "HashGrid",
            "n_levels": n_levels,
            "n_features_per_level": n_features_per_level,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": base_resolution,
            "per_level_scale": per_level_scale,
        }

        # Hash encoding (tinycudann) - use padded input dims
        self.encoding = tcnn.Encoding(
            n_input_dims=self.tcnn_input_dims,
            encoding_config=self.encoding_config
        )

        # Encoding output size
        self.encoding_dim = n_levels * n_features_per_level

        # PyTorch MLP to expand to n_output_dims
        # This handles the large output dimension that tcnn can't do efficiently
        layers = []

        # Input layer: encoding_dim -> n_neurons
        layers.append(nn.Linear(self.encoding_dim, n_neurons))
        layers.append(nn.ReLU())

        # Hidden layers: n_neurons -> n_neurons
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.ReLU())

        # Output layer: n_neurons -> n_output_dims
        layers.append(nn.Linear(n_neurons, n_output_dims))

        # Output activation
        if output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif output_activation == 'tanh':
            layers.append(nn.Tanh())
        # 'none' - no activation

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize MLP weights with Xavier uniform."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, n_input_dims) with values in [0, 1]
               For time input, this is normalized time t/T_period

        Returns:
            Output tensor of shape (batch, n_output_dims)
        """
        # Ensure input is float32 and in [0, 1] for hash encoding
        x = x.float()
        x = torch.clamp(x, 0.0, 1.0)

        # Pad 1D input to 2D if needed (tinycudann requires at least 2D)
        if self.pad_1d_to_2d:
            # [t] -> [t, t] by concatenating along dim 1
            x = torch.cat([x, x], dim=1)

        # Hash encoding: (batch, tcnn_input_dims) -> (batch, encoding_dim)
        # tcnn returns half precision, we convert to float for PyTorch MLP
        features = self.encoding(x)
        features = features.float()

        # MLP: (batch, encoding_dim) -> (batch, n_output_dims)
        output = self.mlp(features)

        return output

    def get_param_count(self):
        """Return parameter counts for encoding and MLP separately."""
        encoding_params = sum(p.numel() for p in self.encoding.parameters())
        mlp_params = sum(p.numel() for p in self.mlp.parameters())
        return {
            'encoding': encoding_params,
            'mlp': mlp_params,
            'total': encoding_params + mlp_params
        }

    def __repr__(self):
        param_counts = self.get_param_count()
        pad_info = f" (padded to {self.tcnn_input_dims}D)" if self.pad_1d_to_2d else ""
        return (
            f"HashEncodingMLP(\n"
            f"  input_dims={self.n_input_dims}{pad_info}, output_dims={self.n_output_dims}\n"
            f"  encoding: {self.encoding_config}\n"
            f"  encoding_dim: {self.encoding_dim}\n"
            f"  mlp: {self.mlp}\n"
            f"  params: encoding={param_counts['encoding']:,}, mlp={param_counts['mlp']:,}, total={param_counts['total']:,}\n"
            f")"
        )


def create_hash_encoding_mlp(config):
    """
    Factory function to create HashEncodingMLP from config.

    Args:
        config: Config object with ngp_* parameters

    Returns:
        HashEncodingMLP instance
    """
    return HashEncodingMLP(
        n_input_dims=1,  # Time is always 1D
        n_output_dims=getattr(config, 'output_size_nnr_f', 1000),
        n_levels=getattr(config, 'ngp_n_levels', 24),
        n_features_per_level=getattr(config, 'ngp_n_features_per_level', 2),
        log2_hashmap_size=getattr(config, 'ngp_log2_hashmap_size', 22),
        base_resolution=getattr(config, 'ngp_base_resolution', 16),
        per_level_scale=getattr(config, 'ngp_per_level_scale', 1.4),
        n_neurons=getattr(config, 'ngp_n_neurons', 128),
        n_hidden_layers=getattr(config, 'ngp_n_hidden_layers', 4),
        output_activation='none'
    )

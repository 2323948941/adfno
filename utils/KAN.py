import torch
import torch.nn as nn
import torch.nn.functional as F

class KAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3, base_activation=nn.GELU, grid_range=[-1, 1]):
        """
        Initialize the KAN model.

        Parameters:
        ----------
        layers_hidden : list
            List of dimensions for the neural network hidden layers, including input and output layer dimensions.
        grid_size : int, optional
            Number of points for the spline interpolation grid, default is 5.
        spline_order : int, optional
            Order of the spline interpolation, default is 3.
        base_activation : nn.Module, optional
            Activation function for the initial input transformation, default is nn.GELU.
        grid_range : list, optional
            Value range defining the spline interpolation grid, default is [-1, 1].
        """
        super(KAN, self).__init__()
        # List of hidden layer dimensions for the neural network
        self.layers_hidden = layers_hidden
        # Number of points for the spline interpolation grid
        self.grid_size = grid_size
        # Order of the spline interpolation
        self.spline_order = spline_order
        # Activation function instance for initial input transformation
        self.base_activation = base_activation()
        # Value range defining the spline interpolation grid
        self.grid_range = grid_range

        # Initialize parameters and layer normalization
        # Parameter list for linear transformation of each layer
        self.base_weights = nn.ParameterList()
        # Parameter list for spline transformation of each layer
        self.spline_weights = nn.ParameterList()
        # Module list of layer normalization for stable training
        self.layer_norms = nn.ModuleList()
        # Module list of PReLU activation to introduce nonlinearity
        self.prelus = nn.ModuleList()
        # Store grid values for spline computation of each layer
        self.grids = []

        # Iterate through layers to initialize weights, normalization layers and grids
        for i, (in_features, out_features) in enumerate(zip(layers_hidden, layers_hidden[1:])):
            # Randomly initialize base weights for linear transformation
            self.base_weights.append(nn.Parameter(torch.randn(out_features, in_features)))
            # Randomly initialize spline weights for spline transformation
            self.spline_weights.append(nn.Parameter(torch.randn(out_features, in_features, grid_size + spline_order)))
            # Add layer normalization to stabilize layer output
            self.layer_norms.append(nn.LayerNorm(out_features))
            # Add PReLU activation for learnable nonlinearity
            self.prelus.append(nn.PReLU())

            # Compute grid values based on the specified range and grid size
            h = (self.grid_range[1] - self.grid_range[0]) / grid_size
            grid = torch.linspace(
                # Grid start value, extended left by spline order intervals
                self.grid_range[0] - h * spline_order,
                # Grid end value, extended right by spline order intervals
                self.grid_range[1] + h * spline_order,
                # Total number of grid points
                grid_size + 2 * spline_order + 1,
                dtype=torch.float32
            ).expand(in_features, -1).contiguous()
            # Register grid as buffer so it is not treated as a model parameter
            self.register_buffer(f'grid_{i}', grid)
            # Append grid values to the list
            self.grids.append(grid)

        # Initialize weights with Kaiming uniform distribution for better initial values
        for weight in self.base_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')
        for weight in self.spline_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')

    def forward(self, x):
        """
        Forward pass method, defining the model's computation flow.

        Parameters:
        x (torch.Tensor): Input tensor

        Returns:
        torch.Tensor: Output tensor of the model
        """
        # Iterate through each layer, process with base weights, spline weights, layer norm and activation
        for i, (base_weight, spline_weight, layer_norm, prelu) in enumerate(zip(self.base_weights, self.spline_weights, self.layer_norms, self.prelus)):
            # Get grid data for the current layer from buffers
            grid = self._buffers[f'grid_{i}']
            # Move input tensor to the same device as the weights
            x = x.to(base_weight.device)

            # Perform base linear transformation: apply base activation to input, then linear transformation
            base_output = F.linear(self.base_activation(x), base_weight)
            # Expand input tensor dimensions for spline operations
            x_uns = x.unsqueeze(-1)
            # Compute base intervals for splines: determine which grid intervals input values fall into
            bases = ((x_uns >= grid[:, :-1]) & (x_uns < grid[:, 1:])).to(x.dtype)

            # Compute multi-order spline basis functions
            for k in range(1, self.spline_order + 1):
                # Left intervals
                left_intervals = grid[:, :-(k + 1)]
                # Right intervals
                right_intervals = grid[:, k:-1]
                # Compute interval delta, avoid division by zero
                delta = torch.where(right_intervals == left_intervals, torch.ones_like(right_intervals), right_intervals - left_intervals)
                # Update spline basis functions following the formula
                bases = ((x_uns - left_intervals) / delta * bases[:, :,:-1]) + \
                        ((grid[:, k + 1:] - x_uns) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])
            # Ensure contiguous memory for computational efficiency
            bases = bases.contiguous()

            # Compute spline transformation: linear combination of basis functions and spline weights
            spline_output = F.linear(bases.view(x.size(0), -1), spline_weight.view(spline_weight.size(0), -1))
            # Sum base and spline outputs, then apply layer normalization and PReLU activation
            x = prelu(layer_norm(base_output + spline_output))

        return x
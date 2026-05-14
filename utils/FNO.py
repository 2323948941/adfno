import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fix random seeds to ensure experiment reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Activation function dictionary: contains common activations including GELU mentioned in the paper
act_dict = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "softplus": nn.Softplus(),
    "sigmoid": nn.Sigmoid(),
    "idt": nn.Identity(),  # Identity mapping (no activation)
    "gelu": nn.GELU()     # Activation used in the paper to enhance nonlinear fitting capability
}

# Weight initialization dictionary: corresponds to the model parameter initialization strategy in the paper
init_dict={
    "xavier_normal": nn.init.xavier_normal_,    # Xavier normal initialization
    "xavier_uniform": nn.init.xavier_uniform_,  # Xavier uniform initialization (adopted in the paper)
    "uniform": nn.init.uniform_,                # Uniform distribution initialization
    "norm": nn.init.normal_                     # Normal distribution initialization
}

################################################################
# Fourier Convolutional Layer (Core module, corresponds to Fourier layer in Section 2.3 of the paper)
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, init_func):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier Convolutional Layer: Implements end-to-end operations of Fourier transform,
        linear transformation in frequency domain, and inverse Fourier transform
        
        Corresponds to Equation (18) in the paper: Convert spatial domain features to frequency domain
        via Fourier transform, perform feature interaction, then convert back to spatial domain
        
        Parameters:
        -----------
        in_channels  : Number of input feature channels (lifted dimension)
        out_channels : Number of output feature channels
        modes1       : Number of frequency modes retained in the first dimension (e.g., y-direction)
                       (truncate high frequencies, retain low-frequency dominant modes)
        modes2       : Number of frequency modes retained in the second dimension (e.g., z-direction)
                       (truncation strategy adopted in the paper to reduce computation)
        init_func    : Parameter initialization method (corresponds to network initialization in the paper)
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of truncated modes in the first frequency dimension (max floor(N/2)+1)
        self.modes2 = modes2  # Number of truncated modes in the second frequency dimension

        # Define frequency-domain weight parameters (real + imaginary parts stored separately, 2 groups for positive/negative frequencies)
        fourier_weight = [nn.Parameter(torch.FloatTensor(
            in_channels, out_channels, modes1, modes2, 2)) for _ in range(2)]
        self.fourier_weight = nn.ParameterList(fourier_weight)

        # Initialize frequency-domain weights (Xavier uniform initialization specified in the paper)
        if init_func in init_dict.keys():
            print("init_func:", init_func)
            init_method = init_dict[init_func]
            for param in self.fourier_weight:
                # Scale initialization range by input/output channels for training stability
                init_method(param, gain=1/(in_channels * out_channels))
        else:
            print("init_func: kaiming normal")
            for param in self.fourier_weight:
                nn.init.kaiming_normal_(param)  # Alternative initialization method

    def compl_mul2d(self, input, weights):
        """Complex matrix multiplication: implements feature interaction between input and frequency-domain weights"""
        # Dimension notation: (batch, in_channel, x, y) × (in_channel, out_channel, x, y) → (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """Forward pass: Spatial domain → Frequency domain → Feature transformation → Spatial domain"""
        batchsize = x.shape[0]  # Batch size

        # 1. Spatial → Frequency domain: 2D Fourier transform on input features (rfft2 saves computation)
        x_ft = torch.fft.rfft2(x)

        # 2. Frequency-domain feature transformation: operate only on truncated low-frequency modes
        # Initialize frequency-domain output (complex type)
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, 
            dtype=torch.cfloat, device=x.device
        )
        # Apply first weight set to the first modes1 positive frequency modes
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(
                x_ft[:, :, :self.modes1, :self.modes2], 
                torch.complex(self.fourier_weight[0][...,0], self.fourier_weight[0][...,1])
            )
        # Apply second weight set to the last modes1 negative frequency modes (utilize frequency symmetry)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(
                x_ft[:, :, -self.modes1:, :self.modes2], 
                torch.complex(self.fourier_weight[1][...,0], self.fourier_weight[1][...,1])
            )

        # 3. Frequency → Spatial domain: inverse Fourier transform back to spatial domain
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))  # Match output size with input

        return x


# 2D Fourier Neural Operator network (corresponds to the full FNO architecture in Section 2.3 of the paper)
class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, n_out, layer_num=4, last_size=128, 
                 act_func="gelu", init_func='xavier_normal', set_bn=False, residual=False):
        super(FNO2d, self).__init__()

        """
        Overall network architecture: Contains lifting layer, multiple Fourier layers, projection layer
        1. Lifting layer: Projects input to high-dimensional feature space
        2. Fourier layers: Multiple rounds of local + non-local feature interaction (Eq.17-18 in paper)
        3. Projection layer: Maps high-dimensional features to output dimension (electromagnetic field components)
        
        Parameters:
        -----------
        modes1    : Number of frequency modes in the first dimension (e.g., y-direction)
        modes2    : Number of frequency modes in the second dimension (e.g., z-direction)
        width     : Number of lifted feature channels (set to 32 in the paper)
        n_out     : Output dimension (2 for real and imaginary parts of EM field components)
        layer_num : Number of Fourier layers (6 used in paper, 4 in this example)
        last_size : Feature channels before projection layer
        act_func  : Activation function (GELU used in the paper)
        set_bn    : Whether to use batch normalization for training stability
        residual  : Whether to use residual connections to mitigate vanishing gradients
        """
        self.set_bn = set_bn        # Enable batch normalization
        self.residual = residual    # Enable residual connections
        self.padding = 9            # Boundary padding (handles non-periodic inputs)
        self.layer_num = layer_num  # Number of Fourier layers

        # Activation function (default: GELU from the paper)
        if act_func in act_dict.keys():
            self.activation = act_dict[act_func]
        else:
            raise KeyError("Activation function not found in act_dict")

        # 1. Lifting layer: Lift input (1 channel, e.g., β=σ·ω) to width dimension
        self.fc0 = nn.Linear(1, width)  

        # 2. Fourier layer block: Fourier conv (non-local) + 1x1 conv (local)
        self.fno = nn.ModuleList()    # Fourier convolutional layers (non-local feature processing)
        self.conv = nn.ModuleList()   # 1x1 convolutional layers (local feature processing, Wv_j term in paper)
        self.bn = nn.ModuleList()     # Batch normalization layers (optional)
        for _ in range(layer_num):
            self.fno.append(SpectralConv2d(width, width, modes1, modes2, init_func))
            self.conv.append(nn.Conv2d(width, width, 1))  # 1x1 conv for local feature transformation
            if set_bn:
                self.bn.append(nn.BatchNorm2d(width))

        # 3. Projection layer: Map high-dimensional features to output dimension
        self.fc1 = nn.Linear(width, last_size)  # Intermediate projection layer
        self.fc2 = nn.Linear(last_size, n_out) # Final output layer (real + imaginary parts of EM field)

    def forward(self, x):
        '''
        Forward pass: Input → Lifting layer → Fourier layers → Projection layer → Output
        
        Input shape  : (batch, y, z, 1)  # 1-channel input (spatial distribution of β=σ·ω)
        Output shape : (batch, y, z, n_out)  # Output EM field components
        '''
        # 1. Lifting layer: project input from 1 channel to width channels
        x = self.fc0(x)  # Shape: (batch, y, z, width)
        # Permute to (batch, channels, y, z) for convolutional operations
        x = x.permute(0, 3, 1, 2)
        # Boundary padding: avoid boundary effects of Fourier transform for non-periodic inputs
        x = F.pad(x, [0, self.padding, 0, self.padding])

        # 2. Fourier layer block: multiple local + non-local feature interactions
        for i in range(self.layer_num):
            res = x  # Residual connection input
            # Fourier convolution (non-local) + 1x1 convolution (local)
            x1 = self.fno[i](x)  # Fourier layer output
            x2 = self.conv[i](x)  # 1x1 conv output
            x = x1 + x2  # Fuse non-local and local features
            x = self.activation(x)  # Nonlinear activation
            # Residual connection
            if self.residual:
                x = res + x
            else:
                x = x
            # Batch normalization
            x = self.bn[i](x) if self.set_bn else x

        # Remove padding to restore original size
        x = x[..., :-self.padding, :-self.padding]
        # Permute back to (batch, y, z, channels) for linear layers
        x = x.permute(0, 2, 3, 1)

        # 3. Projection layer: high-dimensional features → output
        x = self.fc1(x)  # Shape: (batch, y, z, last_size)
        x = self.activation(x)
        x = self.fc2(x)  # Shape: (batch, y, z, n_out) → final EM field output

        return x
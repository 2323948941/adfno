import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Preserve original activation/initialization dictionaries to ensure compatibility of basic components
act_dict = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "softplus": nn.Softplus(),
    "sigmoid": nn.Sigmoid(),
    "idt": nn.Identity(),
    "gelu": nn.GELU()
}

init_dict = {
    "xavier_normal": nn.init.xavier_normal_,
    "xavier_uniform": nn.init.xavier_uniform_,
    "uniform": nn.init.uniform_,
    "norm": nn.init.normal_
}


class InputNormalizer(nn.Module):
    """Normalization layer that preserves original input dimensions, no modifications"""
    def __init__(self, mean_x, std_x, mean_grid, std_grid, mean_freq, std_freq):
        super().__init__()
        self.mean_x = torch.tensor(mean_x).view(1, 1, 1, 1)  # Match x shape: (B,dy,dz,1)
        self.std_x = torch.tensor(std_x).view(1, 1, 1, 1)
        self.mean_grid = torch.tensor(mean_grid).view(1, 1, 1)  # Match grid shape: (B,dy,dz)
        self.std_grid = torch.tensor(std_grid).view(1, 1, 1)
        self.mean_freq = torch.tensor(mean_freq).view(1, 1, 1)  # Match freq shape: (B,n_freq,1)
        self.std_freq = torch.tensor(std_freq).view(1, 1, 1)

    def forward(self, x, grid, freq):
        x_norm = (x - self.mean_x.to(x.device)) / (self.std_x.to(x.device) + 1e-8)
        grid_norm = (grid - self.mean_grid.to(grid.device)) / (self.std_grid.to(grid.device) + 1e-8)
        freq_norm = (freq - self.mean_freq.to(freq.device)) / (self.std_freq.to(freq.device) + 1e-8)
        return x_norm, grid_norm, freq_norm  # Output dimensions are identical to input


class MultiDimDropout(nn.Module):
    """Simplified multi-dimensional Dropout, reduce dropout rate to avoid feature loss"""
    def __init__(self, p=0.02):  # Original 0.05→0.02, weaken regularization
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        return F.dropout(x, self.p, self.training)


class StochasticDropPath(nn.Module):
    """Keep drop path but reduce probability, only used at critical nodes"""
    def __init__(self, drop_prob=0.02):  # Original 0.05→0.02
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = True

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        
        return x / keep_prob * random_tensor if self.scale_by_keep else x * random_tensor


class PhysGuidedFreqAttention(nn.Module):
    """Weaken physical priors: remove strong negative mean initialization, simplify network layers"""
    def __init__(self, hidden_dim=16, dropout_rate=0.02):
        super().__init__()
        # Original 3 Linear layers→2 layers, reduce parameter redundancy
        self.attn_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            MultiDimDropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        # Remove strong physical initialization, use universal initialization
        self._init_weights()

    def _init_weights(self):
        """Universal Xavier initialization, avoid prior constraints"""
        for m in self.attn_net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, log_freq):
        # Input and output dimensions unchanged: (B,n_freq,1)→(B,n_freq,1)
        attn_weights = self.attn_net(log_freq)
        weighted_freq = log_freq * attn_weights
        return weighted_freq, attn_weights


class PhysGuidedSpatialAttention(nn.Module):
    """Weaken physical priors: remove Sobel kernel strong initialization, simplify convolution layers"""
    def __init__(self, in_channels, dropout_rate=0.02):
        super().__init__()
        # Original 3 Conv layers→2 layers, reduce parameter redundancy
        self.attn_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=3, padding='same'),  # Original //2→//4, reduce channels
            nn.BatchNorm2d(in_channels//4, momentum=0.01),
            nn.GELU(),
            MultiDimDropout(dropout_rate),
            nn.Conv2d(in_channels//4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # Remove Sobel kernel initialization, use universal initialization
        self._init_weights()

    def _init_weights(self):
        """Universal Xavier initialization, remove gradient direction constraints"""
        for m in self.attn_net:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, spatial_feat, sig):
        # Input and output dimensions unchanged: spatial_feat(B,C,dy,dz)→weighted_feat(B,C,dy,dz)
        sig_gradient = F.avg_pool2d(torch.abs(sig[:, :, 1:, :] - sig[:, :, :-1, :]), kernel_size=2)
        sig_gradient = F.interpolate(sig_gradient, size=spatial_feat.shape[2:])
        
        attn_weights = self.attn_net(spatial_feat)
        attn_weights = attn_weights * (1 + sig_gradient)  # Keep gradient guidance, remove strong kernel constraints
        weighted_feat = spatial_feat * attn_weights
        return weighted_feat, attn_weights


class DeepONetTrunk(nn.Module):
    """Simplify spatial encoding: reduce convolution layers and channel redundancy"""
    def __init__(self, hidden_dim, dropout_rate=0.02):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Original 3 Conv layers→2 layers, channels reduced from hidden_dim//2→hidden_dim//4, fewer parameters
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim//4, kernel_size=3, padding='same'),
            nn.BatchNorm2d(hidden_dim//4, momentum=0.01),
            nn.GELU(),
            MultiDimDropout(dropout_rate),
            nn.Conv2d(hidden_dim//4, hidden_dim, kernel_size=3, padding='same'),  # Directly upscale to hidden_dim
            nn.BatchNorm2d(hidden_dim, momentum=0.01),
            nn.GELU(),
            MultiDimDropout(dropout_rate),
        )
        self.spatial_attn = PhysGuidedSpatialAttention(in_channels=hidden_dim, dropout_rate=dropout_rate)
        self.drop_path = StochasticDropPath(drop_prob=0.02)  # Only keep critical drop path

    def forward(self, grid, sig):
        # Input and output dimensions unchanged: grid(B,1,dy,dz)→spatial_feat(B,hidden_dim,dy,dz)
        spatial_feat = self.spatial_encoder(grid)
        weighted_spatial_feat, attn_weights = self.spatial_attn(spatial_feat, sig)
        weighted_spatial_feat = self.drop_path(weighted_spatial_feat)
        return weighted_spatial_feat, attn_weights


class DeepONetBranch(nn.Module):
    """Simplify feature fusion: remove redundant concatenation, use element-wise multiplication (EFDO-inspired efficient fusion)"""
    def __init__(self, hidden_dim, dropout_rate=0.02):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Simplify sig encoding: 2 Conv layers→1 layer, reduce channels
        self.sig_encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding='same'),  # Direct encode to hidden_dim, reduce steps
            nn.BatchNorm2d(hidden_dim, momentum=0.01),
            nn.GELU(),
            MultiDimDropout(dropout_rate),
        )
        
        self.freq_attn = PhysGuidedFreqAttention(hidden_dim=hidden_dim//4, dropout_rate=dropout_rate)
        # Simplify freq encoding: 2 Linear layers→1 layer, directly match hidden_dim
        self.freq_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),  # Output dim=hidden_dim, avoid subsequent broadcast redundancy
            nn.GELU(),
            MultiDimDropout(dropout_rate),
        )

    def forward(self, sig, freq):
        # Input: sig(B,1,dy,dz), freq(B,n_freq,1)
        # Output: fused_feat(B,n_freq,hidden_dim,dy,dz) (dimensions unchanged)
        sig_feat = self.sig_encoder(sig)  # (B, hidden_dim, dy, dz)
        
        # Frequency processing: keep original dimensions, simplify encoding
        weighted_freq, freq_attn_weights = self.freq_attn(freq)  # (B,n_freq,1)
        freq_feat = self.freq_encoder(weighted_freq)  # (B,n_freq,hidden_dim)
        
        # Core improvement: element-wise multiplication replaces concatenation, fewer parameters + faster fusion
        # freq_feat broadcast: (B,n_freq,hidden_dim) → (B,n_freq,hidden_dim,1,1)
        # sig_feat broadcast: (B,hidden_dim,dy,dz) → (B,1,hidden_dim,dy,dz)
        fused_feat = freq_feat.unsqueeze(-1).unsqueeze(-1) * sig_feat.unsqueeze(1)
        
        return fused_feat, freq_attn_weights


class SpectralConv2d(nn.Module):
    """Keep original Fourier layer logic, no modifications (ensure frequency domain feature extraction)"""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class Unet(nn.Module):
    """Simplify U-Net: remove redundant conv layers, reduce downsampling complexity (core lightweighting)"""
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate=0.02):
        super().__init__()
        # Original 3 downsampling layers→2 layers, remove redundant conv2_1/conv3_1
        self.conv1 = self._conv_block(input_channels, output_channels, kernel_size, 2, dropout_rate)
        self.conv2 = self._conv_block(output_channels, 2*output_channels, kernel_size, 2, dropout_rate)
        
        # Original 3 upsampling layers→2 layers, match downsampling dimensions
        self.deconv1 = self._deconv_block(2*output_channels, output_channels)
        self.deconv0 = self._deconv_block(2*output_channels, input_channels)  # Concatenated channels=output_channels+output_channels=2*output_channels
        
        self.output_layer = self._conv_block(2*input_channels, output_channels, 1, 1, dropout_rate)
        self.drop_path = StochasticDropPath(drop_prob=0.02)  # Only keep 1 drop path at critical position

    def _conv_block(self, in_planes, out_channels, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(in_planes, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            MultiDimDropout(dropout_rate)
        )

    def _deconv_block(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.GELU()
        )

    def forward(self, x):
        # Input and output dimensions unchanged: x(B,C,dy,dz)→out(B,C,dy,dz)
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)  # Direct downsampling, remove conv2_1
        
        # Simplified upsampling: only 2 concatenations
        out_deconv1 = self.deconv1(out_conv2)
        out_deconv1 = out_deconv1[:, :, :out_conv1.shape[2], :out_conv1.shape[3]]
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        concat1 = self.drop_path(concat1)  # Only keep drop path at critical position

        out_deconv0 = self.deconv0(concat1)
        out_deconv0 = out_deconv0[:, :, :x.shape[2], :x.shape[3]]
        concat0 = torch.cat((x, out_deconv0), 1)

        return self.output_layer(concat0)


class UFNOBlock(nn.Module):
    """Simplify UFNO block: reduce U-Net calls, optimize residual connection (avoid gradient conflict)"""
    def __init__(self, channels, modes1, modes2, kernel_size=3, dropout_rate=0.02):
        super().__init__()
        self.fourier = SpectralConv2d(channels, channels, modes1, modes2)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)  # Keep 1x1 conv to ensure channel matching
        self.unet = Unet(channels, channels, kernel_size, dropout_rate)  # Use simplified U-Net
        self.norm = nn.BatchNorm2d(channels, momentum=0.01)
        self.activation = nn.GELU()
        self.drop_path = StochasticDropPath(drop_prob=0.02)

    def forward(self, x):
        # Input: x(B,n_freq,C,dy,dz) → Output: x(B,n_freq,C,dy,dz) (dimensions unchanged)
        B, n_freq, C, dy, dz = x.shape
        x_reshaped = x.view(B * n_freq, C, dy, dz)  # Merge B and n_freq for U-Net compatibility
        
        # Simplify feature fusion: feed Fourier + 1x1 conv directly into U-Net
        x_fourier = self.fourier(x_reshaped)
        x_conv = self.conv(x_reshaped)
        x_unet = self.unet(x_fourier + x_conv)  # Reduce intermediate steps, accelerate gradient propagation
        
        x_unet = self.drop_path(x_unet)
        x_out = self.norm(x_unet + x_reshaped)  # Residual connection: ensure consistent input/output distribution
        x_out = self.activation(x_out)
        
        return x_out.view(B, n_freq, C, dy, dz)  # Restore original dimensions

class ADFNO(nn.Module):
    """
    Improved ADFNO: Separate mapping logic for magnetic field components and forward response
    Input: x(B, dy, dz, 1), grid(B, dy, dz), freq(B, n_freq, 1)
    Output: pred_ex(B, n_freq, dy-1, dz-1, em_out_channels)  -> (20, 16, 43, 43, 4)
            pred_y(B, n_freq, dy, rho_phase_out_channels)    -> (20, 16, 44, 4)
    """
    def __init__(self, 
                 em_out_channels=4,  
                 rho_phase_out_channels=4, 
                 fno_modes1=18, 
                 fno_modes2=18, 
                 hidden_dim=64, 
                 num_ufno_blocks=3,  
                 dropout_rate=0.02):
        super().__init__()
        self.em_out_channels = em_out_channels
        self.rho_phase_out_channels = rho_phase_out_channels
        self.padding = 8  # Preserve edge features
        
        # --- Core extraction modules remain unchanged ---
        self.trunk = DeepONetTrunk(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.branch = DeepONetBranch(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.ufno_blocks = nn.ModuleList([
            UFNOBlock(hidden_dim, fno_modes1, fno_modes2, dropout_rate=dropout_rate) 
            for _ in range(num_ufno_blocks)
        ])

        self.feature_projection = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )

        # ==============================================================================
        # Output head for pred_ex: Smooth transition from 44x44 -> 43x43
        # Physical meaning: Conversion from grid nodes to cell centers
        # ==============================================================================
        self.grad_output_conv = nn.Sequential(
            # Core change: kernel=2, padding=0 to reduce spatial dimension by exactly 1
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dim, momentum=0.01),
            nn.GELU(),
            MultiDimDropout(dropout_rate),
            
            # Deepen non-linear mapping to enhance fitting of magnetic field distribution
            nn.Conv2d(hidden_dim, hidden_dim//2, kernel_size=3, padding='same'),
            nn.BatchNorm2d(hidden_dim//2, momentum=0.01),
            nn.GELU(),
            MultiDimDropout(dropout_rate),
            
            nn.Conv2d(hidden_dim//2, em_out_channels, kernel_size=1)
        )
        
        # ==============================================================================
        # Output head for pred_y: Strengthen feature isolation and 1D mapping capability
        # Physical meaning: Extract surface response data from 2D field, independent distribution requiring stronger 1D non-linear fitting
        # ==============================================================================
        # Stage 1: 2D feature enhancement (refine field features before averaging)
        self.y_spatial_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding='same'),
            nn.BatchNorm2d(hidden_dim, momentum=0.01),
            nn.GELU()
        )
        # Stage 2: 1D sequence enhancement (after dz mean pooling, use Conv1d for forward response mapping)
        self.y_output_conv1d = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=3, padding='same'),
            nn.BatchNorm1d(hidden_dim//2, momentum=0.01),
            nn.GELU(),
            MultiDimDropout(dropout_rate),
            nn.Conv1d(hidden_dim//2, rho_phase_out_channels, kernel_size=1)
        )

    def forward(self, x, grid, freq):
        # --- Input Preprocessing & Padding ---
        B, dy, dz, _ = x.shape
        n_freq = freq.shape[1]
        
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding]) 
        
        grid = grid.unsqueeze(1)
        grid = F.pad(grid, [0, self.padding, 0, self.padding]) 

        # --- Feature Extraction & UFNO ---
        spatial_feat, spatial_attn = self.trunk(grid, x) 
        fused_feat, freq_attn = self.branch(x, freq)      
        
        spatial_feat_broadcast = spatial_feat.unsqueeze(1).repeat(1, n_freq, 1, 1, 1)
        total_feat = fused_feat + spatial_feat_broadcast  

        B_n, n_freq, C, dy_pad, dz_pad = total_feat.shape
        total_feat_reshaped = total_feat.view(B_n * n_freq, C, dy_pad, dz_pad)
        total_feat_reshaped = self.feature_projection(total_feat_reshaped)
        total_feat = total_feat_reshaped.view(B_n, n_freq, C, dy_pad, dz_pad)

        ufno_feat = total_feat
        for block in self.ufno_blocks:
            ufno_feat = block(ufno_feat) 

        # --- Crop Padding to restore original size ---
        ufno_feat = ufno_feat[:, :, :, :-self.padding, :-self.padding]  # Now [B, n_freq, C, dy, dz]
        B, n_freq, C, cur_dy, cur_dz = ufno_feat.shape
        ufno_feat_reshaped = ufno_feat.view(B * n_freq, C, cur_dy, cur_dz) 

        # ==========================================================
        # Output Branch 1: pred_grad (magnetic gradient components)
        # ==========================================================
        # After kernel=2 no-padding operation in grad_output_conv
        # Dimension change: (B*n_freq, C, dy, dz) -> (B*n_freq, 4, dy-1, dz-1)
        grad_conv = self.grad_output_conv(ufno_feat_reshaped) 
        
        # Reshape to target format: [B, n_freq, dy-1, dz-1, 4]
        pred_grad = grad_conv.view(B, n_freq, self.em_out_channels, cur_dy - 1, cur_dz - 1).permute(0, 1, 3, 4, 2)

        # ==========================================================
        # Output Branch 2: pred_y (forward response)
        # ==========================================================
        # 1. Refine features related to data response in 2D space
        y_spatial = self.y_spatial_conv(ufno_feat_reshaped) 
        
        # 2. Mean pooling along dz(depth) dimension, compress 2D field to surface/line 1D signal
        # Dimension change: (B*n_freq, C, dy, dz) -> (B*n_freq, C, dy)
        y_1d = y_spatial.mean(dim=-1) 
        
        # 3. Use powerful Conv1d for non-linear mapping, adapt to distribution different from EM field
        # Dimension change: (B*n_freq, C, dy) -> (B*n_freq, 4, dy)
        y_conv = self.y_output_conv1d(y_1d) 
        
        # Reshape to target format: [B, n_freq, dy, 4]
        pred_y = y_conv.view(B, n_freq, self.rho_phase_out_channels, cur_dy).permute(0, 1, 3, 2)
        
        return pred_grad, pred_y
    
if __name__ == "__main__":
    pass
import numpy as np  # For numerical calculations
import torch  # Deep learning framework for tensor operations (GPU acceleration)
import torch.nn as nn  # Neural network module for building model layers
import scipy.sparse as scipa  # For sparse matrix operations (1D reference model calculation)
import scipy.sparse.linalg as scilg  # For solving sparse linear systems
from typing import Optional, List  # Type hints for improved code readability

II = complex(0, 1)  # Imaginary unit i (matches complex form of electromagnetic field equations in the paper)


def sig_decrete(dy, dz, sig):
    """
    Compute volume-averaged conductivity at grid nodes (σ)
    Corresponds to Equation (24) in the paper: Equivalent conductivity at nodes via weighted average
    of surrounding grid conductivities
    Parameters:
        dy: Grid spacing in y-direction
        dz: Grid spacing in z-direction
        sig: Raw conductivity grid data (shape: n_samples × nz × ny)
    Returns:
        sigc0: Volume-averaged conductivity at nodes (shape: n_samples × (nz+1) × (ny+1))
    """
    n_samples, mm, nn = sig.shape  # mm: number of z-grid cells; nn: number of y-grid cells
    device = sig.device  # Get computation device (GPU/CPU)
    sigc0 = torch.ones((n_samples, mm + 1, nn + 1), device=device)  # Initialize node conductivity matrix (with boundaries)

    ny = len(dy)  # Number of y-direction grids
    nz = len(dz)  # Number of z-direction grids
    # Generate grid spacing matrices for weight calculation
    dz0, dy0 = torch.meshgrid(dz, dy, indexing='ij')

    # Calculate area weights of four adjacent grids (for volume averaging)
    w1 = dy0[0:nz-1, 0:ny-1] * dz0[0:nz-1, 0:ny-1]  # Top-left grid area
    w2 = dy0[0:nz-1, 1:ny] * dz0[0:nz-1, 0:ny-1]    # Top-right grid area
    w3 = dy0[0:nz-1, 0:ny-1] * dz0[1:nz, 0:ny-1]    # Bottom-left grid area
    w4 = dy0[0:nz-1, 1:ny] * dz0[1:nz, 0:ny-1]      # Bottom-right grid area
    area = (w1 + w2 + w3 + w4) / 4.0  # Average area around the node

    # Volume-weighted average for node conductivity (discrete implementation of Eq.24)
    sigc = (sig[:, 0:nz-1, 0:ny-1] * w1 + 
            sig[:, 0:nz-1, 1:ny] * w2 + 
            sig[:, 1:nz, :ny-1] * w3 + 
            sig[:, 1:nz, 1:ny] * w4) / (area * 4.0)

    # Fill conductivity for boundary nodes (edge handling)
    sigc0[:, 0, :] = sig[0, 0, 0]  # Top boundary
    sigc0[:, -1, :] = sig[0, -1, -1]  # Bottom boundary
    sigc0[:, 1:-1, 0] = sigc[:, :, 0]  # Left boundary (inner nodes)
    sigc0[:, 1:-1, -1] = sigc[:, :, -1]  # Right boundary (inner nodes)
    sigc0[:, 1:-1, 1:-1] = sigc  # Inner nodes

    return sigc0


def impose_bc(u, u_lr):
    '''
    Apply hard boundary conditions (corresponds to "hard constraint" strategy in the paper)
    Directly fix electromagnetic field values at boundaries to satisfy Dirichlet boundary conditions
    (discrete implementation of Equation 6)
    Parameters:
        u: Network-predicted EM field components (real + imaginary, shape: batch × nz × ny × 2)
        u_lr: EM field values at boundaries (analytical solution from 1D reference model)
    '''
    _, _, ny, _ ,_= u.size()  # Get number of y-direction grids
    # Left boundary (y=0): fix real and imaginary parts
    u[:, :,:, 0, 0] = u_lr.real  
    u[:, :, :,0, 1] = u_lr.imag  
    # Right boundary (y=ny-1): fix real and imaginary parts
    u[:, :,:, -1, 0] = u_lr.real  
    u[:, :, :,-1, 1] = u_lr.imag  
    # Top boundary (z=0): fix real and imaginary parts (repeat along y-direction)
    u[:,:, 0, :, 0] = u_lr[:, :,0:1].real.repeat(1, 1,ny)  
    u[:, :,0, :, 1] = u_lr[:, :,0:1].imag.repeat(1, 1,ny)  
    # Bottom boundary (z=nz-1): fix real and imaginary parts (repeat along y-direction)
    u[:, :,-1, :, 0] = u_lr[:, :,-1:].real.repeat(1, 1,ny)  
    u[:, :,-1, :, 1] = u_lr[:, :,-1:].imag.repeat(1, 1,ny)  
    return u

def sig_add_tensor(sig: torch.Tensor) -> torch.Tensor:
    '''
    Expand the dimensions of the conductivity tensor (add one row and column)
    Purpose: Match conductivity grid size to EM field node grid size (for PyTorch tensor operations)
    Parameters:
        sig: Raw conductivity tensor (shape: samples × z_grids × y_grids, e.g. [50, 84, 84])
    Returns:
        sig0: Expanded conductivity tensor (shape: samples × (z+1) × (y+1), e.g. [50, 85, 85])
    '''
    # 1. Get original tensor shape (use .shape attribute directly for tensors)
    n_samples, mm, nn = sig.shape  # mm: z-grid count; nn: y-grid count
    
    # 2. Initialize expanded tensor (match device/dtype of input sig)
    sig0 = torch.zeros((n_samples, mm + 1, nn + 1), device=sig.device, dtype=sig.dtype)
    
    # 3. Fill raw conductivity values (same slicing logic as NumPy)
    sig0[:, :-1, :-1] = sig  # Fill top-left with original data
    
    # 4. Fill bottom boundary (copy last element of first sample)
    sig0[:, -1, :] = sig[0, -1, -1]  # Fill new bottom row with fixed value
    
    # 5. Fill right boundary (copy last column of original tensor)
    sig0[:, :-1, -1] = sig[:, :, -1]  # Fill new right column
    
    return sig0

def TE_pde_loss(u0, beta0, u_lr, dy, dz):
    """
    Compute PDE residual loss for TE mode (xy-mode) (corresponds to Equations 21-23 in the paper)
    Quantify violation of governing equations for predicted EM field component (E_x)
    using finite difference approximation of second derivatives
    Parameters:
        u0: Network output EM field (real + imaginary, shape: batch × nz × ny × 2)
        beta0: Input β=σ·ω (conductivity × angular frequency, shape: batch × nz × ny × 1)
        u_lr: Boundary conditions (analytical solution from 1D reference model)
        dy: Grid spacing in y-direction
        dz: Grid spacing in z-direction
    Returns:
        Average PDE residual (MAE, loss metric in the paper)
    """
    # Compute β at nodes (volume averaging, same as sig_decrete)
    beta = sig_decrete(dy, dz, beta0[..., :-1, :-1, 0])
    n_samples = len(beta)  # Number of samples
    beta = 2.0 * np.pi * beta  # Convert to angular frequency (ω=2πf)

    # Apply hard boundary conditions (enforce physical constraints at boundaries)
    # impose_bc(u0, u_lr)

    # Combine real + imaginary parts into complex EM field component E_x (u = E_x^s, secondary field)
    u = u0[..., 0] + II * u0[..., 1]
    
    mu = 4.0e-7 * np.pi  # Vacuum permeability μ₀ (μ=μ₀ in the paper)
    ny = len(dy)  # Number of y-grids
    nz = len(dz)  # Number of z-grids

    # Compute primary field (analytical solution from 1D reference model) for secondary field separation
    ex1d = u_lr.unsqueeze(-1).repeat(1, 1, ny + 1)
    u = u - ex1d  # Secondary field = Total field - Primary field (SFM method, Section 2.1)

    # Generate grid spacing matrices for finite differences
    dy00, dz00 = torch.meshgrid(dy, dz,indexing="ij")
    dy0 = dy00.T.repeat(n_samples, 1, 1)  # Batch-expanded y-spacing
    dz0 = dz00.T.repeat(n_samples, 1, 1)  # Batch-expanded z-spacing

    # Compute average grid spacing (for second derivative approximation)
    dyc = (dy0[:, 0:nz-1, 0:ny-1] + dy0[:, 0:nz-1, 1:ny]) / 2.0  # Central y-spacing
    dzc = (dz0[:, 0:nz-1, 0:ny-1] + dz0[:, 1:nz, 0:ny-1]) / 2.0  # Central z-spacing

    # Calculate area weights of four adjacent grids (same as sig_decrete)
    w1 = dy0[:, 0:nz-1, 0:ny-1] * dz0[:, 0:nz-1, 0:ny-1]
    w2 = dy0[:, 0:nz-1, 1:ny] * dz0[:, 0:nz-1, 0:ny-1]
    w3 = dy0[:, 0:nz-1, 0:ny-1] * dz0[:, 1:nz, 0:ny-1]
    w4 = dy0[:, 0:nz-1, 1:ny] * dz0[:, 1:nz, 0:ny-1]
    area = (w1 + w2 + w3 + w4) / 4.0  # Average area around node

    # Finite difference coefficients (for second derivative approximation)
    dzdy1 = dzc / dy0[:, 0:nz-1, 0:ny-1]  # Left y-difference coefficient
    dzdy2 = dzc / dy0[:, 0:nz-1, 1:ny]    # Right y-difference coefficient
    dydz1 = dyc / dz0[:, 0:nz-1, 0:ny-1]  # Upper z-difference coefficient
    dydz2 = dyc / dz0[:, 1:nz, 0:ny-1]    # Lower z-difference coefficient
    val = dzdy1 + dzdy2 + dydz1 + dydz2   # Combined coefficient for second derivative

    # Compute conductivity difference term (Δβ=β-β_ref, β_ref = reference model β)
    beta_ref = beta[:, :, 0:1].repeat(1, 1, ny + 1)  # Reference β (1D model)
    beta_diff = beta - beta_ref  # Conductivity difference
    beta_diff_d = beta_diff[:, 1:-1, 1:-1]  # Inner node difference

    # RHS of governing equation (primary field term, Δβ·E_x^p in Eq.21)
    coef = II * mu * beta_diff_d * area  # Coefficient
    rhs = coef * ex1d[:, 1:nz, 1:ny]  # Right-hand side term

    # LHS of governing equation (second derivative + decay term of E_x^s, Eq.21 LHS)
    beta = beta[:, 1:-1, 1:-1]  # Inner node β
    mtx1 = II * mu * beta * area - val  # LHS coefficient

    # Compute PDE residual (LHS - RHS, residual of Eq.21)
    f = (u[:, :-2, 1:-1] * dydz1 +  # Upper z second derivative
         u[:, 1:-1, :-2] * dzdy1 +  # Left y second derivative
         u[:, 1:-1, 2:] * dzdy2 +   # Right y second derivative
         u[:, 2:, 1:-1] * dydz2 +   # Lower z second derivative
         u[:, 1:-1, 1:-1] * mtx1 +  # Decay term
         rhs)  # Subtract RHS

    # Normalize residual (avoid magnitude imbalance during training)
    MF = 1  # Adjustable based on coefficient matrix magnitude
    return torch.mean(torch.abs(f / MF ))  # Return mean absolute error (MAE)


def compute_complex_jacobian(
    y: torch.Tensor, x: torch.Tensor, freq_idx: int, 
    create_graph: bool = True, use_checkpoint: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gradient of E_x field w.r.t. σ (sensitivity matrix)"""
    def _single_grad(single_field: torch.Tensor, input_x: torch.Tensor):
        grad = torch.autograd.grad(
            outputs=single_field,
            inputs=input_x,
            grad_outputs=torch.ones_like(single_field, device=input_x.device),
            create_graph=create_graph,
            retain_graph=True,
            only_inputs=True
        )[0]
        return grad.squeeze(dim=-1)  # [batch, nx, ny]

    # Extract real and imaginary parts for current frequency
    y_real = y[:, freq_idx, :, :, 0].to(x.device)
    y_imag = y[:, freq_idx, :, :, 1].to(x.device)

    jac_real = _single_grad(y_real, x)
    jac_imag = _single_grad(y_imag, x)

    return jac_real[:, :-1, :-1], jac_imag[:, :-1, :-1]  



def preprocess_gradient(
    grad: torch.Tensor, clip_min: float = 1e-3, clip_max: float = 1e3, 
    target_range: tuple = (-1, 1)
) -> torch.Tensor:
    """Unified preprocessing for true/predicted gradients (clipping + normalization)"""
    if grad is None:
        return None
    
    # Clip extreme values
    grad_clipped = torch.clamp(grad, min=clip_min, max=clip_max)
    
    # Normalize to target range
    min_val, max_val = target_range
    grad_min, grad_max = clip_min, clip_max
    if grad_max == grad_min:
        return torch.full_like(grad_clipped, (min_val + max_val) / 2)
    grad_norm = (grad_clipped - grad_min) / (grad_max - grad_min) * (max_val - min_val) + min_val
    
    return torch.clamp(grad_norm, min_val, max_val)


def mt2dhyhz_torch(
    self,
    freq: torch.Tensor,  # [50, 16, 1] → batch frequencies
    dy: torch.Tensor,    # [63] → y-direction grid spacing
    dz: torch.Tensor,    # [63] → z-direction grid spacing
    sig: torch.Tensor,   # [50, 63, 63] → batch conductivity
    ex: torch.Tensor,    # [50, 16, 64, 64] → batch complex electric field
) -> tuple[torch.Tensor, torch.Tensor]:
    device = ex.device
    dtype = ex.dtype
    miu = torch.tensor(4 * torch.pi * 1e-7, dtype=dtype, device=device)
    II = torch.tensor(1j, dtype=dtype, device=device)

    # Dimension preprocessing: align B/F dimensions for all variables
    B, F, _, _ = ex.shape  # B=50, F=16
    ny = dy.shape[0]       # ny=63

    # Frequency processing: [50,16,1] → [50,16] (B,F)
    freq = freq.squeeze(dim=-1)
    omega = 2 * torch.pi * freq  # [50,16] → correct shape

    # Conductivity processing: [50,63,63] → [50,1,63,63] (add F dim=1) → expand F to 16
    sig = sig.unsqueeze(dim=1)  # [50,1,63,63] (B=50, F=1, Z=63, Y=63)
    sig = sig.expand(-1, F, -1, -1)  # Expand F dim to 16 → [50,16,63,63]

    # Grid spacing processing: [63] → [1,1,63] (B=1, F=1, Y/Z=63)
    dy = dy.unsqueeze(dim=0).unsqueeze(dim=0)  # [1,1,63]
    dz = dz.unsqueeze(dim=0).unsqueeze(dim=0)  # [1,1,63]
    kk=self.nza
    delz = dz[..., kk]  # [1,1] → broadcast to [50,16]

    # -------------------------- 2. Batch compute Hy --------------------------
    hys = torch.zeros((B, F, ny + 1), dtype=dtype, device=device)  # [50,16,64]

    # 2.1 Top-left Hy (y=0): fix dimension mismatch
    ex_kk_y0 = ex[..., kk, 0]      # [50,16] → correct (B,F)
    ex_k1_y0 = ex[..., kk + 1, 0]  # [50,16] → correct (B,F)
    # Conductivity: [50,16,63,63] → take kk layer y=0 → [50,16] (F=16)
    sig_kk_y0 = sig[..., kk, 0]    # Critical fix: no squeeze → [50,16]
    # Both terms [50,16], valid addition
    c0 = (-1 / (II * omega * miu * delz)) + (3 / 8) * sig_kk_y0 * delz  # [50,16]
    c1 = (1 / (II * omega * miu * delz)) + (1 / 8) * sig_kk_y0 * delz   # [50,16]
    hys[..., 0] = c0 * ex_kk_y0 + c1 * ex_k1_y0  # Shape match

    # 2.2 Top-right Hy (y=ny, ny=63): fix dimension mismatch
    ex_kk_yny = ex[..., kk, ny]      # [50,16]
    ex_k1_yny = ex[..., kk + 1, ny]  # [50,16]
    sig_kk_yny = sig[..., kk, ny-1]    # [50,16] (F=16)
    c0 = (-1 / (II * omega * miu * delz)) + (3 / 8) * sig_kk_yny * delz
    c1 = (1 / (II * omega * miu * delz)) + (1 / 8) * sig_kk_yny * delz
    hys[..., ny] = c0 * ex_kk_yny + c1 * ex_k1_yny  # Shape match

    # 2.3 Middle Hy (y=1~ny-1): fix sigc dimension
    dyj = dy[..., :-1] + dy[..., 1:]  # [1,1,62]
    sig_kk = sig[..., kk, :]  # [50,16,63] (B,F,Y=63)
    # Compute sigc: shape [50,16,62] (F=16), broadcast compatible
    sigc = (sig_kk[..., :-1] * dy[..., :-1] + sig_kk[..., 1:] * dy[..., 1:]) / dyj  # [50,16,62]
    # cc: omega → [50,16,1] broadcast with dyj[1,1,62] → [50,16,62]
    cc = delz / (4 * II * omega.unsqueeze(dim=-1) * miu * dyj)  # [50,16,62]
    # c0_mid/c1_mid: [50,16,62], no conflict
    term1 = -1.0 / (II * omega.unsqueeze(dim=-1) * miu * delz)  # [50,16,1] → [50,16,62]
    term2 = (3 / 8) * sigc * delz  # [50,16,62]
    term3 = 3 * cc * (1 / dy[..., 1:] + 1 / dy[..., :-1])  # [50,16,62]
    c0_mid = term1 + term2 - term3  # [50,16,62]

    term1 = 1 / (II * omega.unsqueeze(dim=-1) * miu * delz)  # [50,16,62]
    term2 = (1 / 8) * sigc * delz  # [50,16,62]
    term3 = 1 * cc * (1 / dy[..., 1:] + 1 / dy[..., :-1])  # [50,16,62]
    c1_mid = term1 + term2 - term3  # [50,16,62]

    # ex slices: [50,16,62], match coefficient shapes
    ex_kk_y_l = ex[..., kk, :-2]    # [50,16,62]
    ex_kk_y_m = ex[..., kk, 1:-1]   # [50,16,62]
    ex_kk_y_r = ex[..., kk, 2:]     # [50,16,62]
    ex_k1_y_l = ex[..., kk + 1, :-2]  # [50,16,62]
    ex_k1_y_m = ex[..., kk + 1, 1:-1] # [50,16,62]
    ex_k1_y_r = ex[..., kk + 1, 2:]   # [50,16,62]

    c0l = 3.0 * cc / dy[...,0:ny-1]
    c0r = 3.0 * cc / dy[...,1:ny]
    c1l = 1.0 * cc / dy[...,0:ny-1]
    c1r = 1.0 * cc / dy[...,1:ny]

    hys_mid = (c0l * ex_kk_y_l + c0_mid * ex_kk_y_m + c0r * ex_kk_y_r +
               c1l * ex_k1_y_l + c1_mid * ex_k1_y_m + c1r * ex_k1_y_r)  # [50,16,62]
    hys[..., 1:ny] = hys_mid

    # -------------------------- 3. Batch compute Hz --------------------------
    hzs = torch.zeros((B, F, ny + 1), dtype=dtype, device=device)  # [50,16,64]

    # 3.1 Left Hz (y=0): shape matched
    ex_kk_y1 = ex[..., kk, 1]    # [50,16]
    ex_kk_y0 = ex[..., kk, 0]    # [50,16]
    denom = II * omega * miu * dy[..., 0]  # [50,16]
    hzs[..., 0] = (-1 / denom) * (ex_kk_y1 - ex_kk_y0)  # Shape match

    # 3.2 Right Hz (y=ny): shape matched
    ex_kk_yny = ex[..., kk, ny]      # [50,16]
    ex_kk_yn1 = ex[..., kk, ny - 1]  # [50,16]
    denom = II * omega * miu * dy[..., -1]  # [50,16]
    hzs[..., ny] = (-1 / denom) * (ex_kk_yny - ex_kk_yn1)  # Shape match

    # 3.3 Middle Hz: shape matched
    ex_kk_y_r = ex[..., kk, 2:]     # [50,16,62]
    ex_kk_y_l = ex[..., kk, :-2]    # [50,16,62]
    ex_diff = ex_kk_y_r - ex_kk_y_l  # [50,16,62]
    denom = II * omega.unsqueeze(dim=-1) * miu * dyj  # [50,16,62]
    hzs_mid = (-1 / denom) * ex_diff  # [50,16,62]
    hzs[..., 1:ny] = hzs_mid

    return hys, hzs  # [50,16,64], [50,16,64]

def mt2dzxy_torch(freq, exr, hyr):
    """
    Batch compute apparent resistivity and phase for TE mode
    Parameters:
        freq: (B,F) torch float
        exr: (B,F,O) torch complex
        hyr: (B,F,O) torch complex
    Returns:
        rhote, phste: (B,F,O) torch float
    """
    dtype=exr.real.dtype
    device=exr.device
    miu = torch.tensor(4 * torch.pi * 1e-7, dtype=dtype, device=device)
    omega = 2.0 * torch.pi * freq # (B,F,1)
    zxy = exr / hyr
    rhote = (torch.abs(zxy)**2) / (omega*miu)
    phste = torch.atan2(zxy.imag, zxy.real) * 180.0/torch.pi
    return rhote, phste

def interp1d_torch(x, xp, fp):
    """
    PyTorch linear interpolation
    x: (O,) observation points
    xp: (Y,) grid points
    fp: (B,F,Y) or (B,F,Y+1)
    Returns: (B,F,O)
    """
    device = fp.device  # Match device of fp (CPU/GPU consistency)
    dtype = fp.real.dtype 
    x = torch.tensor(x, dtype=dtype, device=device) 
    xp = torch.tensor(xp, dtype=dtype, device=device)

    inds = torch.searchsorted(xp, x) - 1  # (O,)
    inds = torch.clamp(inds, 0, len(xp)-2)
    x0, x1 = xp[inds], xp[inds+1]
    f0, f1 = fp[:,:,inds], fp[:,:,inds+1]
    w = (x - x0)/(x1-x0+1e-12)
    w = w.unsqueeze(0).unsqueeze(0)  # (1,1,O)
    return f0*(1-w)+f1*w


def TE_pde_with_configurable_loss(
    self, u0, ex_te, sig0, grad_te,freq, u_lr,ep,y,y_normalizer,pred_y,
    # Loss switches
    enable_data_loss: bool = True,
    enable_ig_loss: bool = False,
    criterion_mse: Optional[nn.Module] = None,
    # Device
    device: Optional[torch.device] = None
):
    # 1. Device & loss initialization (ensure all tensors on same device)
    # 2. Initialize loss variables (create on target device to avoid later transfer)
    """
    Two loss terms only:
    1. Data loss: MSE between predicted E_x and true E_x
    2. Sensitivity loss: MSE between predicted and true gradients (stage-wise enabled)
    """    
    device=self.device
    mse_criterion = nn.MSELoss().to(device)
    
    # -------------------------- 1. Data loss (always enabled) --------------------------
    if enable_data_loss:
        ytrue=(y[...,0:2])
        data_loss=self.mse_loss(pred_y,y_normalizer.encode(ytrue))     

    # -------------------------- 2. Sensitivity loss (stage-wise enabled) --------------------------
    grad_loss = torch.tensor(0.0, device=device)
    if enable_ig_loss:
        freq_sample=self.freq_sample
        second_ep=self.second_ep
        grad_loss_total = 0.0
        n_freq = grad_te.shape[1]
        sample_freqs = np.linspace(0, n_freq-1, freq_sample, dtype=int)  # Multi-frequency sampling
        
        for freq_idx in sample_freqs:
            # Compute and preprocess predicted gradients
            pred_jac_real, pred_jac_imag = compute_complex_jacobian(
                y=u0, x=sig0, freq_idx=freq_idx, create_graph=True
            )
            pred_jac_real = preprocess_gradient(pred_jac_real)
            pred_jac_imag = preprocess_gradient(pred_jac_imag)
            
            # Preprocess true gradients (sync with predicted)
            true_jac_real = grad_te[:, freq_idx, :, :, 0].to(device)
            true_jac_imag = grad_te[:, freq_idx, :, :, 1].to(device)
            true_jac_real = preprocess_gradient(true_jac_real)
            true_jac_imag = preprocess_gradient(true_jac_imag)
            
            # Accumulate single-frequency loss
            grad_loss_real = mse_criterion(pred_jac_real, true_jac_real)
            grad_loss_imag = mse_criterion(pred_jac_imag, true_jac_imag)
            grad_loss_total += (grad_loss_real + grad_loss_imag) / 2
        
        grad_loss = grad_loss_total / freq_sample  # Average multi-frequency loss
        

    return data_loss, grad_loss

def get_response1(out0, sig0, dy, dz, ry, yn, nza, freq0,f_idx):
    """
    Compute magnetotelluric response parameters (apparent resistivity, phase)
    from predicted EM field components
    Corresponds to Equations (7)-(9) in the paper:
    Z_xy=E_x/H_y, ρ_xy=|Z_xy|²/(ωμ), φ_xy=arg(Z_xy)
    Parameters:
        out0: Network output EM field (real + imaginary, E_x)
        sig: Conductivity distribution
        dy/dz: Grid spacing
        ry: Observation point positions (y-direction)
        yn: Original y-direction grid coordinates
        nza: Number of air layer grids
        freq: Frequency
    Returns:
        rhoxy: Apparent resistivity (TE mode)
        phsxy: Phase (TE mode)
    """
    # Repeat conductivity to match frequency count (f_idx[1] frequencies per sample)
    np_dtype = np.float32  # Define data type (memory saving + precision consistency)
    n_train=out0.shape[0]
    out0 = torch.repeat_interleave(torch.from_numpy(out0.detach().numpy().astype(np_dtype)), f_idx[1], dim=0)
    sig = torch.repeat_interleave(torch.from_numpy(sig0.detach().numpy().astype(np_dtype)), f_idx[1], dim=0)
    # Repeat frequency to match sample count (n_train samples per frequency)
    freq= torch.from_numpy(freq0.astype(np_dtype)).repeat(n_train, 1, 1)

    out0=out0.cpu().numpy()
    sig=sig.cpu().numpy()
    dy=dy.cpu().numpy()
    dz=dz.cpu().numpy()
    freq=freq.cpu().numpy()

    # Convert output to complex E_x (total field)
    ex = out0[..., 0] + II * out0[..., 1]

    # Compute magnetic field component H_y (numerical differentiation of E_x, Eq.2)
    hys, _ = mt2dhyhz(freq, dy, dz, sig, ex, nza)

    # Extract E_x at surface (bottom of air layer)
    exs = ex[:, nza, :]
    n_samples = len(sig)  # Number of samples

    # Initialize E_x and H_y at observation points
    exr = np.zeros((n_samples, len(ry)), dtype=np.complex64)  # E_x at observations
    hyr = np.zeros((n_samples, len(ry)), dtype=np.complex64)  # H_y at observations

    # Interpolate to observation points (map grid data to physical observation positions)
    for ii in range(n_samples):
        exr[ii, :] = np.interp(ry, yn, exs[ii])  # E_x interpolation
        hyr[ii, :] = np.interp(ry, yn, hys[ii])  # H_y interpolation

    # Compute impedance, apparent resistivity, phase (Eqs.7-9)
    _, rhoxy, phsxy = mt2dzxy(freq[:, :, 0], exr, hyr)
    return rhoxy, phsxy

def get_response(out0, sig, dy, dz, ry, yn, nza, freq):
    """
    Compute magnetotelluric response parameters (apparent resistivity, phase)
    from predicted EM field components
    Corresponds to Equations (7)-(9) in the paper:
    Z_xy=E_x/H_y, ρ_xy=|Z_xy|²/(ωμ), φ_xy=arg(Z_xy)
    Parameters:
        out0: Network output EM field (real + imaginary, E_x)
        sig: Conductivity distribution
        dy/dz: Grid spacing
        ry: Observation point positions (y-direction)
        yn: Original y-direction grid coordinates
        nza: Number of air layer grids
        freq: Frequency
    Returns:
        rhoxy: Apparent resistivity (TE mode)
        phsxy: Phase (TE mode)
    """
    # Convert output to complex E_x (total field)
    ex = out0[..., 0] + II * out0[..., 1]

    # Compute magnetic field component H_y (numerical differentiation of E_x, Eq.2)
    hys, _ = mt2dhyhz(freq, dy, dz, sig, ex, nza)

    # Extract E_x at surface (bottom of air layer)
    exs = ex[:, nza, :]
    n_samples = len(sig)  # Number of samples

    # Initialize E_x and H_y at observation points
    exr = np.zeros((n_samples, len(ry)), dtype=np.complex64)  # E_x at observations
    hyr = np.zeros((n_samples, len(ry)), dtype=np.complex64)  # H_y at observations

    # Interpolate to observation points (map grid data to physical observation positions)
    for ii in range(n_samples):
        exr[ii, :] = np.interp(ry, yn, exs[ii])  # E_x interpolation
        hyr[ii, :] = np.interp(ry, yn, hys[ii])  # H_y interpolation

    # Compute impedance, apparent resistivity, phase (Eqs.7-9)
    _, rhoxy, phsxy = mt2dzxy(freq[:, :, 0], exr, hyr)
    return rhoxy, phsxy


def mt2dhyhz(freq, dy0, dz0, sig, ex, nza):
    """
    Compute magnetic field component H_y for TE mode
    (finite difference from E_x, Eq.2 in the paper)
    Parameters:
        freq: Frequency
        dy0/dz0: Grid spacing
        sig: Conductivity
        ex: Electric field component E_x
        nza: Number of air layer grids
    Returns:
        hys: Magnetic field component H_y
        hzs: Magnetic field component H_z (unused in TE mode)
    """
    omega0 = 2.0 * np.pi * freq  # Angular frequency
    mu = 4.0e-7 * np.pi  # Vacuum permeability
    ny = np.size(dy0)  # Number of y-grids
    n_samples = len(sig)  # Number of samples

    # Expand dy to batch dimension
    dy = dy0.reshape(1, -1) * np.ones((n_samples, len(dy0)))
    dz = dz0  # z-direction grid spacing

    # Initialize H_y (shape: samples × (ny+1))
    hys = np.zeros((n_samples, ny + 1), dtype=complex)    

    # 1. Compute H_y at top boundary (bottom of air layer) - corner points
    kk = nza  # Air layer bottom index
    delz = dz[kk]  # z-spacing at this layer
    # Left corner (y=0)
    sigc = sig[:, kk, 0]  # Conductivity
    omega = omega0[:, 0, 0]  # Angular frequency
    c0 = -1.0 / (II * omega * mu * delz) + (3.0 / 8.0) * sigc * delz  # Difference coefficient
    c1 = 1.0 / (II * omega * mu * delz) + (1.0 / 8.0) * sigc * delz
    hys[:, 0] = c0 * ex[:, kk, 0] + c1 * ex[:, kk + 1, 0]  # Left H_y
    # Right corner (y=ny)
    sigc = sig[:, kk, ny - 1]
    hys[:, ny] = c0 * ex[:, kk, ny] + c1 * ex[:, kk + 1, ny]  # Right H_y

    # 2. Compute H_y at inner nodes (second-order finite difference)
    dyj = dy[:, 0:ny-1] + dy[:, 1:ny]  # Sum of adjacent y-spacings
    sigc = (sig[:, kk, 0:ny-1] * dy[:, 0:ny-1] + sig[:, kk, 1:ny] * dy[:, 1:ny]) / dyj  # Average conductivity
    omega = omega0[:, 0, :]  # Angular frequency
    cc = delz / (4.0 * II * omega * mu * dyj)  # Difference coefficient
    # Combined coefficients (discretization of Eq.2)
    c0 = -1.0/(II*omega*mu*delz) + (3.0/8.0)*sigc*delz - cc*3.0*(1.0/dy[:,1:ny]+1.0/dy[:,0:ny-1])
    c1 = 1.0/(II*omega*mu*delz) + (1.0/8.0)*sigc*delz - cc*1.0*(1.0/dy[:,1:ny]+1.0/dy[:,0:ny-1])
    c0l = 3.0 * cc / dy[:, 0:ny-1]
    c0r = 3.0 * cc / dy[:, 1:ny]
    c1l = 1.0 * cc / dy[:, 0:ny-1]
    c1r = 1.0 * cc / dy[:, 1:ny]
    # Compute inner node H_y
    hys[:, 1:ny] = (c0l * ex[:, kk, 0:ny-1] + c0 * ex[:, kk, 1:ny] + c0r * ex[:, kk, 2:ny+1] +
                    c1l * ex[:, kk+1, 0:ny-1] + c1 * ex[:, kk+1, 1:ny] + c1r * ex[:, kk+1, 2:ny+1])

    hzs = np.zeros((n_samples, ny + 1), dtype=complex)  # H_z (unused in TE mode)
    return hys, hzs


def mt2dzxy(freq, exr, hyr):
    """
    Compute TE mode impedance, apparent resistivity, and phase (Eqs.7-9)
    Parameters:
        freq: Frequency
        exr: Electric field E_x at observation points
        hyr: Magnetic field H_y at observation points
    Returns:
        zxy: Impedance Z_xy = E_x / H_y
        rhote: Apparent resistivity ρ_xy
        phste: Phase φ_xy
    """
    omega = 2.0 * np.pi * freq  # Angular frequency
    mu = 4.0e-7 * np.pi  # Vacuum permeability
    zxy = np.array(exr / hyr, dtype=np.complex64)  # Impedance (Eq.7)
    rhote = abs(zxy) ** 2 / (omega * mu)  # Apparent resistivity (Eq.8)
    phste = np.arctan2(zxy.imag, zxy.real) * 180.0 / np.pi  # Phase (Eq.9, rad → deg)
    return zxy, rhote, phste


def mt1dte(freq, dz0, sig0,n_add):
    """
    Compute TE mode EM field for 1D reference model
    (generate primary field and boundary conditions, analytical solution from Wait (1953))
    Parameters:
        freq: Frequency
        dz: z-direction grid spacing
        sig: 1D conductivity distribution
    Returns:
        ex: 1D model electric field E_x (analytical solution for boundary conditions)
    """
    miu = 4.0e-7 * np.pi  # Vacuum permeability
    II = complex(0, 1)   
    omega = 2.0 * np.pi * freq  # Angular frequency

    # Refine grid for higher accuracy 1D field calculation
    dz = np.array([dz0[i] / n_add * np.ones(n_add) for i in range(np.size(dz0))]).flatten()
    sig = np.array([sig0[i] * np.ones(n_add) for i in range(np.size(sig0))]).flatten()
    nz = np.size(sig)

    # Extend conductivity/grid (add bottom half-space)
    sig = np.hstack((sig, sig[nz-1]))
    dz = np.hstack((dz, np.array(np.sqrt(2.0 / (sig[nz] * omega * miu)), dtype=float)))

    # Build 1D TE finite difference matrix (tridiagonal)
    diagA = II * omega * miu * (sig[0:nz] * dz[0:nz] + sig[1:nz+1] * dz[1:nz+1]) - 2.0 / dz[0:nz] - 2.0 / dz[1:nz+1]
    offdiagA = 2.0 / dz[1:nz]  # Off-diagonal elements

    # Build sparse matrix and solve linear system (BC: E_x=1 at top, E_x=0 at bottom)
    mtxA = scipa.diags(diagA, format='csc') + scipa.diags(offdiagA, 1, format='csc') + scipa.diags(offdiagA, -1, format='csc')
    rhs = np.zeros((nz, 1), dtype=float)
    rhs[0] = -2.0 / dz[0]  # Top boundary contribution
    lup = scilg.splu(mtxA)  # Sparse LU decomposition
    ex = lup.solve(rhs)  # Solve for E_x
    ex = np.array(np.concatenate(([1.0], ex.reshape(-1))), dtype=complex)
    idx = np.arange(np.size(sig0) + 1) * n_add
    ex = ex[idx]
    return ex.reshape(-1)


# Loss function classes for evaluating relative error between predicted/true responses
# (accuracy metric in the paper)
class LpLoss(object):
    """Compute relative error of Lp-norm (for single-output evaluation)"""
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0  # Positive dimension and norm order
        self.d = d
        self.p = p
        self.reduction = reduction  # Whether to reduce loss
        self.size_average = size_average  # Whether to average loss

    def rel(self, x, y):
        num_examples = x.shape[0]
        
        # Compute Lp-norm ratio of prediction vs reference
        diff_norms = np.linalg.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = np.linalg.norm(y.reshape(num_examples, -1), self.p, 1)
        # Reduce or average
        if self.reduction:
            return np.mean(diff_norms / y_norms) if self.size_average else np.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class LpLoss1(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        """
        Initialize LpLoss instance for Lp loss calculation
        Parameters:
        d (int, optional): Spatial dimension. Default=2.
        p (int, optional): Lp-norm type. Default=2 (L2 norm).
        size_average (bool, optional): Average loss. Default=True.
        reduction (bool, optional): Reduce loss (mean/sum). Default=True.
        """
        # Ensure positive spatial dimension and Lp-norm order
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        """
        Compute absolute Lp loss between x and y
        Parameters:
        x (torch.Tensor): Predicted values (num_examples, ...)
        y (torch.Tensor): True values (same shape as x)
        Returns:
        torch.Tensor: Absolute Lp loss
        """
        num_examples = x.size()[0]

        # Uniform grid spacing h
        h = 1.0 / (x.size()[1] - 1.0)

        # Compute Lp norm per sample
        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            return torch.mean(all_norms) if self.size_average else torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        """
        Compute relative Lp loss between x and y
        Parameters:
        x (torch.Tensor): Predicted values (num_examples, ...)
        y (torch.Tensor): True values (same shape as x)
        Returns:
        torch.Tensor: Relative Lp loss
        """
        num_examples = x.size()[0]

        # Lp norm of prediction error
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        # Lp norm of true values
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            return torch.mean(diff_norms/y_norms) if self.size_average else torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms
    def __call__(self, x, y):
        """
        Callable instance: compute relative Lp loss
        Parameters:
        x (torch.Tensor): Predicted values
        y (torch.Tensor): True values
        Returns:
        torch.Tensor: Relative Lp loss
        """
        return self.rel(x, y)
    
class LpLoss_out(object):
    """Compute Lp-norm relative error for multi-output (e.g. apparent resistivity + phase)"""
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        assert d > 0 and p > 0
        self.d = d
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        # Compute Lp-norm ratio for multi-output
        diff_norms = np.linalg.norm(x - y, axis=(1))
        y_norms = np.linalg.norm(y, axis=(1))
        # Reduce or average
        if self.reduction:
            return np.mean(diff_norms / y_norms) if self.size_average else np.sum(diff_norms / y_norms) / x.shape[-1]
        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class TorchLpLossOut(nn.Module):
    """
    PyTorch multi-output Lp-norm relative error loss
    Supports autograd for training loss calculation
    Purpose: Compute combined Lp error for multi-output (e.g. apparent resistivity + phase)
    """
    def __init__(self, p=2, size_average=True, reduction=True, eps=1e-8):
        super(TorchLpLossOut, self).__init__()
        assert p > 0  # Positive norm order
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.eps = eps  # Avoid division by zero

    def rel(self, x, y):
        # Ensure PyTorch tensors
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        
        # Lp norm of prediction error (axis=1)
        diff = x - y
        diff_norms = torch.norm(diff, p=self.p, dim=1)
        
        # Lp norm of true values (axis=1) + eps
        y_norms = torch.norm(y, p=self.p, dim=1) + self.eps
        
        # Relative error
        rel_errors = diff_norms / y_norms
        
        # Reduce/average
        if self.reduction:
            if self.size_average:
                return torch.mean(rel_errors)
            else:
                return torch.sum(rel_errors) / x.size(-1)
        return rel_errors

    def forward(self, x, y):
        # Forward pass: compute loss
        return self.rel(x, y)
    

class LpLoss_out2(object):
    """Compute separate Lp-norm relative error for multi-output (individual metrics)"""
    def __init__(self, axis=(1, 2), size_average=True, reduction=True):
        self.reduction = reduction
        self.size_average = size_average
        self.axis = axis  # Reduction axes

    def rel(self, x, y):
        # Compute norm ratio along specified axes
        diff_norms = np.linalg.norm(x - y, axis=self.axis)
        y_norms = np.linalg.norm(y, axis=self.axis)
        # Reduce or average
        if self.reduction:
            return np.mean(diff_norms / y_norms, axis=0) if self.size_average else np.sum(diff_norms / y_norms, axis=0)

    def __call__(self, x, y):
        return self.rel(x, y)
    
class TorchLpLossOut2(nn.Module):
    """
    PyTorch multi-output Lp-norm relative error loss
    Supports autograd for training loss calculation
    Purpose: Compute separate Lp error for multi-output (e.g. apparent resistivity/phase)
    """
    def __init__(self, axis=(1, 2), p=2, size_average=True, reduction=True, eps=1e-8):
        super(TorchLpLossOut2, self).__init__()
        self.axis = axis
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.eps = eps  # Avoid division by zero

    def rel(self, x, y):
        # Ensure PyTorch tensors
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        
        # Lp norm of prediction error
        diff = x - y
        diff_norms = torch.norm(diff, p=self.p, dim=self.axis)
        
        # Lp norm of true values + eps
        y_norms = torch.norm(y, p=self.p, dim=self.axis) + self.eps
        
        # Relative error
        rel_errors = diff_norms / y_norms
        
        # Reduce/average
        if self.reduction:
            if self.size_average:
                return torch.mean(rel_errors, dim=0)
            else:
                return torch.sum(rel_errors, dim=0)
        return rel_errors

    def forward(self, x, y):
        # Forward pass: compute loss
        return self.rel(x, y)
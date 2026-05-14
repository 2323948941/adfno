import numpy as np  # Numerical computation library
import scipy.io as scio  # For reading .mat format data files
import torch  # Deep learning framework for tensor operations
from utils.derivative import mt1dte  # Import 1D reference model EM field calculation function (generates boundary conditions, corresponds to Wait (1953) analytical solution in the paper)


class MaxMinNormalizer(object):
    
    def __init__(self, x, eps=1e-8):
        super(MaxMinNormalizer, self).__init__()
        self.min = torch.min(x)
        self.max = torch.max(x)
        self.eps = eps

    def encode(self, x):
        x_norm = (x - self.min) / (self.max - self.min + self.eps)
        return x_norm
    
    def decode(self, x_norm):
        x_original = x_norm * (self.max - self.min + self.eps) + self.min
        return x_original


class LogNormalizer(object):
    '''
    Implements log normalization (log10 scaled by global maximum) and denormalization
    Normalization formula: $\tilde{d}_i = \frac{\log_{10}(d_i)}{\max_{j}(\log_{10}(d_j))}$
    Denormalization formula: $d_i = 10^{\tilde{d}_i \times \max_{j}(\log_{10}(d_j))}$
    '''
    def __init__(self, x, eps=1e-8):
        super(LogNormalizer, self).__init__()
        log_x = torch.log10(x)  # Apply log10 to all elements of input tensor
        self.max_log = torch.max(log_x)  # Compute global maximum after log10 transformation
        self.eps = eps  # Prevent division by zero (theoretical non-zero max_log, retained for robustness)

    def encode(self, x):
        '''Encode raw values to normalized values'''
        log_x = torch.log10(x)
        x_norm = log_x / (self.max_log + self.eps)  # Divide by global maximum of log10 values
        return x_norm
    
    def decode(self, x_norm):
        '''Decode normalized values back to raw values'''
        # 1. Restore log10(raw value): normalized value × global max of log10 values
        log_x_original = x_norm * (self.max_log + self.eps)
        # 2. Restore raw value: power operation of 10 (inverse log10)
        x_original = torch.pow(10, log_x_original)
        return x_original


def get_data(file_name):
    '''
    Load raw data from file
    Corresponds to the paper: Data preparation step in Section 3.1 "Generating conductivity structures",
    loads conductivity, frequency, response data for training/testing
    
    Parameters:
        file_name: Path to .npy data file
    Returns:
        nza: Number of air layer grids; zn/yn: z/y direction grid coordinates; ry: Receiver positions;
        sig: Conductivity structure; freq: Frequency; response: MT response (apparent resistivity, phase)
    '''
    key_map = ['rhoxy', 'phsxy', 'rhoyx', 'phsyx']  # MT response parameter keys (corresponds to output Eqs.7-9 in the paper)

    data = np.load(file_name)  # Load .npy file

    # Extract grid coordinates (z and y directions, corresponding to spatial discretization in the paper)
    zn = data['zn0']  # z-direction grid node coordinates (depth direction)
    yn = data['yn0']  # y-direction grid node coordinates (horizontal direction)
    ry = data['ry']   # Surface receiver positions (for calculating MT response)
    sig = data['sig']   # Conductivity structure (3D array: num_samples × z_grids × y_grids)
    freq = data['freq'] # Frequency array (corresponds to multi-frequency generalization scenario in the paper)
    nza = data['nza'].item()  # Number of air layer grids (upper grids with extremely low conductivity)
    grad_te = data['grad_te']
    grad_tm = data['grad_tm']
    
    # Combine MT response parameters (apparent resistivity and phase, corresponding to ρ_xy, φ_xy, etc.)
    response = np.stack([data[key_map[i]] for i in range(len(key_map))], axis=-1)
    response = (abs(response)) # Take absolute value of phase
    
    grad_tetm = np.concatenate((grad_te, grad_tm), axis=-1)
    
    return nza, zn, yn, freq, ry, sig, response, grad_tetm   


def sig_add(sig):
    '''
    Expand the size of conductivity matrix (add one row and one column)
    Function: Match conductivity grid size with EM field node grid size for subsequent finite difference calculations
    (engineering implementation of grid discretization in the paper)
    
    Parameters:
        sig: Raw conductivity matrix (num_samples × z_grids × y_grids)
    Returns:
        sig0: Expanded conductivity matrix (num_samples × (z+1) × (y+1))
    '''
    n_samples, mm, nn = np.shape(sig)  # mm: z-direction grid count; nn: y-direction grid count
    sig0 = np.zeros((n_samples, mm + 1, nn + 1))  # Initialize expanded matrix
    sig0[:, :-1, :-1] = sig  # Fill raw conductivity values
    sig0[:, -1, :] = sig[0, -1, -1]  # Bottom boundary padding (copy last row for boundary grid handling)
    sig0[:, :-1, -1] = sig[:, :, -1]  # Right boundary padding (copy last column)
    return sig0


def get_loader(train_file, test_file, n_train, n_test, batch_size, f_idx):
    '''
    Generate training/testing data loaders (fixed Tensor type error)
    '''
    np_dtype = np.float32
    
    # ---------------------- Training Data Processing ----------------------
    nza, zn, yn, freq0_train, ry, sig0, response0, grad_tetm = get_data(train_file)
    
    # Grid parameters
    n_freq = freq0_train.shape[0]
    dy = torch.from_numpy((yn[1:] - yn[:-1]).astype(np_dtype))
    dz = torch.from_numpy((zn[1:] - zn[:-1]).astype(np_dtype))
    Y, Z = np.meshgrid(yn, zn)
    grid = torch.from_numpy(Y.astype(np_dtype))  # (64, 64)
    grid = grid / 1e5
    
    # Frequency sampling
    freq_train0 = freq0_train
    
    # Data slicing and conversion (critical: ensure all data converted to Tensor)
    sig = sig0[:n_train]
    # Convert response data to Tensor
    response = torch.from_numpy(
        response0[:n_train, ...].astype(np_dtype)
    )
    # Convert EM gradient data to Tensor
    grad_tetm_train = torch.from_numpy(grad_tetm[:n_train].astype(np_dtype))
    
    # Conductivity processing
    sigc_train = torch.from_numpy(sig_add(sig).astype(np_dtype))
    x_train = sigc_train.unsqueeze(-1)
    
    # Reshape frequency dimension
    freq_train = torch.from_numpy(freq_train0.astype(np_dtype))\
                     .repeat(n_train, 1, 1)\
                     .permute(0, 2, 1)
    
    # Boundary conditions (complex Tensor)
    sig_back = np.ones_like(sig) * sig[..., 0:1]
    sig_diff = sig - sig_back

    n_add = 5
    imsize = sigc_train.shape[-2]
    u_bc_train = np.zeros((n_freq, imsize), dtype=np.complex64)
    for ii in range(n_freq):
        u_bc_train[ii, :] = mt1dte(freq_train0[ii], dz.numpy(), (sig-sig_diff)[0, :, 0], n_add)
    u_bc_train = torch.from_numpy(u_bc_train).repeat(n_train, 1, 1)
    grid_train = grid.repeat(n_train, 1, 1)
    
    # Verify consistency of sample count across all tensors
    assert x_train.shape[0] == n_train, f"x_train sample count mismatch: {x_train.shape[0]} vs {n_train}"
    assert response.shape[0] == n_train, f"response sample count mismatch: {response.shape[0]} vs {n_train}"
    
    x_train = 1 / x_train
    x_normalizer = LogNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    y_normalizer = LogNormalizer(response)
    response = y_normalizer.encode(response)

    grad_normalizer = MaxMinNormalizer(grad_tetm_train)
    grad_tetm_train = grad_normalizer.encode(grad_tetm_train)

    train_data = torch.utils.data.TensorDataset(
        x_train, response, freq_train, u_bc_train, grad_tetm_train, grid_train
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True
    )
    
    # ---------------------- Testing Data Processing ----------------------
    _, _, _, freq0_test, _, sig0_test, response0_test, grad_tetm_test = get_data(test_file)
    
    freq_test0 = freq0_test
    
    # Convert testing data to Tensor
    sig_test = sig0_test[:n_test, ...]
    response_test = torch.from_numpy(
        response0_test[:n_test, ...].astype(np_dtype)
    )
    # Convert EM gradient data to Tensor
    grad_tetm_test = torch.from_numpy(grad_tetm_test[:n_test].astype(np_dtype))

    # Conductivity processing
    sigc_test = torch.from_numpy(sig_add(sig_test).astype(np_dtype))
    x_test = sigc_test.unsqueeze(-1)
    
    # Reshape frequency dimension
    freq_test = torch.from_numpy(freq_test0.astype(np_dtype))\
                    .repeat(n_test, 1, 1)\
                    .permute(0, 2, 1)
    
    # Testing boundary conditions
    u_bc_test = np.zeros((n_freq, imsize), dtype=np.complex64)
    for ii in range(n_freq):
        u_bc_test[ii, :] = mt1dte(freq_test0[ii], dz.numpy(), sig_test[0, :, 0], n_add)
    u_bc_test = torch.from_numpy(u_bc_test).repeat(n_test, 1, 1)
    grid_test = grid.repeat(n_test, 1, 1)

    # Create testing dataset
    x_test = 1 / x_test
    x_test = x_normalizer.encode(x_test)
    response_test = y_normalizer.encode(response_test)
    grad_tetm_test = grad_normalizer.encode(grad_tetm_test)
    
    test_data = torch.utils.data.TensorDataset(
        x_test, response_test, freq_test, u_bc_test, grad_tetm_test, grid_test
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, drop_last=False
    )
    
    # Receiver locations for training
    ry1 = torch.from_numpy(ry.astype(np_dtype))
    freq_train01 = torch.from_numpy(freq_train0.astype(np_dtype))
    obs = ry1 / torch.max(ry1) 
    # Generate meshgrid of frequency and receiver positions
    loc1, loc2 = torch.meshgrid(freq_train01, obs)
    loc_train = torch.cat((loc1.reshape(-1,1), loc2.reshape(-1,1)), -1) 

    return (freq_train0, nza, zn.astype(np_dtype), yn.astype(np_dtype), 
            dz, dy, ry.astype(np_dtype), train_loader, test_loader, 
            x_normalizer, y_normalizer, loc_train)
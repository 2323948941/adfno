'''
2-D Magnetotelluric (MT) Forward Modeling Code using Finite Difference Method (FDM)
Adopts the Secondary Field Method (different from the Total Field Method)
'''

# import ray
# # Initialize Ray parallel computing environment with 20 CPU cores, no GPU
# ray.init(num_cpus=20, num_gpus=0)
import numpy as np
import scipy.io as scio  # For reading and writing .mat files
import scipy.sparse as scipa  # For creating sparse matrices
import scipy.sparse.linalg as scilg  # For solving sparse matrix equations
import cmath as cm  # For complex number operations
import torch
import torch.nn.functional as F

# Define Ray remote class for parallel computing
class MT2DFD(object):
    """2-D Magnetotelluric Finite Difference Forward Modeling Class"""
    
    def __init__(self, nza, zn, yn, freq, ry, sig, n_add=5):
        '''
        Initialize forward modeling parameters
        
        Parameters:
        nza : int, number of air layers
        zn : np.array, shape (nz+1,), z-direction node positions, starting from 0, positive downward
        yn : np.array, shape (ny+1,), y-direction node positions
        freq : np.array, shape (n,), array of frequencies for calculation
        ry : array of observation point positions
        sig : model conductivity distribution, shape (nz, ny)
        n_add : int, number of interpolation points per original grid for 1D field calculation, default 5
        '''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nza = nza  # Number of air layers
        self.miu = 4.0e-7 * np.pi  # Vacuum permeability
        self.II = cm.sqrt(-1)  # Imaginary unit i
        
        # Z-direction grid parameters
        self.zn = zn
        self.nz = len(zn)  # Number of z-direction nodes
        self.dz = zn[1:] - zn[:-1]  # Z-direction grid spacing
        
        # Y-direction grid parameters
        self.yn = yn
        self.ny = len(yn)  # Number of y-direction nodes
        self.dy = yn[1:] - yn[:-1]  # Y-direction grid spacing
        
        # Frequency parameters
        self.freq = freq
        self.nf = np.size(freq)  # Number of frequency points
        
        # Observation system
        self.ry = ry
        self.nry = len(ry)  # Number of observation points
        self.dtype = np.float32
        
        # Model conductivity
        self.sig = sig
        
        # Interpolation points for 1D field
        self.n_add = n_add
        
        # Check if conductivity matrix dimensions match the grid
        if np.shape(sig) != (self.nz-1, self.ny-1):
            raise ValueError("Conductivity matrix dimensions error, must be (nz-1, ny-1)")
        
        # Background conductivity (use left boundary conductivity as background)
        self.sig_back = np.ones_like(sig) * sig[:, 0:1]
        
        # Anomalous conductivity (total conductivity - background conductivity)
        self.sig_diff = self.sig - self.sig_back

    def sigte_add_tensor(self, sig):
        '''
        Extend the dimensions of the conductivity tensor (add one row and one column)
        Function: Match conductivity grid dimensions with EM field node grid (for PyTorch tensor calculations)
        Parameters:
            sig: Original conductivity tensor (shape: samples × z_grids × y_grids, e.g., [50, 84, 84])
        Returns:
            sig0: Extended conductivity tensor (shape: samples × (z+1) × (y+1), e.g., [50, 85, 85])
        '''
        # 1. Get original tensor shape (use .shape attribute instead of np.shape)
        mm, nn = sig.shape  # mm: z-direction grid count; nn: y-direction grid count
        
        # 2. Initialize extended tensor (replace np.zeros, critical: match device and dtype of input sig)
        # device=sig.device: ensure same device (CPU/CUDA); dtype=sig.dtype: ensure consistent data type (e.g., float32)
        sig0 = torch.tensor(np.zeros((mm + 1, nn + 1)), device=sig.device, dtype=sig.dtype)
        
        # 3. Fill original conductivity values (same slicing logic as NumPy, tensors support identical indexing)
        sig0[:-1, :-1] = sig  # Fill top-left with original data (exclude new last row/column)
        
        # 4. Fill bottom boundary (copy last element of first sample, align with original NumPy logic)
        sig0[-1, :] = sig[-1, -1]  # New last row (z+1) filled with fixed value
        
        # 5. Fill right boundary (copy last column of original tensor, same slicing logic)
        sig0[:-1, -1] = sig[:, -1]  # New last column (y+1) filled with original last column
        
        return sig0

    def mt2d(self, mode="TETM"):
        """
        Main function for 2D MT forward modeling
        
        Parameters:
        mode : calculation mode, "TE", "TM" or "TETM" (default)
        
        Returns:
        Apparent resistivity, phase and impedance tensor elements
        """
        dy = self.dy
        dz = self.dz
        sig = self.sig
        sig_diff = self.sig_diff
        yn = self.yn
        ry = self.ry
        nza = self.nza
        n_add = self.n_add
        
        # Initialize output result arrays
        rhoxy = np.zeros((self.nf, self.nry), dtype=self.dtype)
        phsxy = np.zeros((self.nf, self.nry), dtype=self.dtype)
        rhoyx = np.zeros((self.nf, self.nry), dtype=self.dtype)
        phsyx = np.zeros((self.nf, self.nry), dtype=self.dtype)

        # Initialize EM field result arrays
        Ex_all = np.zeros((self.nf, (self.nz), (self.ny)), dtype=np.complex64)
        Hx_all = np.zeros((self.nf, (self.nz)-nza, (self.ny)), dtype=np.complex64)
        grad_te = np.zeros((self.nf, (self.nz)-1, (self.ny)-1), dtype=np.complex64)
        grad_tm = np.zeros((self.nf, (self.nz)-1-nza, (self.ny)-1), dtype=np.complex64)
            
        # Select TE, TM or both based on mode
        if mode == "TE":
            # TE mode calculation
            for kf in range(0, self.nf):
                # Calculate electric field component Ex for TE mode
                ex = self.mt2dte(self.freq[kf], dy, dz, sig, sig_diff, n_add)
                # Calculate magnetic field components Hy and Hz from Ex
                hys, hzs = self.mt2dhyhz(self.freq[kf], dy, dz, sig, ex)
                
                # Extract electric field values at the bottom of air layers
                exs = ex[nza, :]
                # Interpolate electric and magnetic field values at observation points
                exr = np.interp(ry, yn, exs)
                hyr = np.interp(ry, yn, hys)

                Ex_all[kf, :] = ex
                
                # Calculate impedance, apparent resistivity and phase
                rhoxy[kf, :], phsxy[kf, :] = self.mt2dzxy(
                    self.freq[kf], exr, hyr)
            return rhoxy, phsxy, Ex_all
            
        elif mode == "TM":
            # TM mode calculation (remove air layers)
            dz = self.dz[nza:]
            sig = self.sig[nza:, :]
            for kf in range(0, self.nf):
                # Calculate magnetic field component Hx for TM mode
                hx = self.mt2dtm(self.freq[kf], dy, dz, sig, sig_diff, n_add)
                # Calculate electric field components Ey and Ez from Hx
                eys, ezs = self.mt2deyez(self.freq[kf], dy, dz, sig, hx)
                # Extract magnetic field values at the surface
                hxs = hx[0, :]
                # Interpolate magnetic and electric field values at observation points
                hxr = np.interp(ry, yn, hxs)
                eyr = np.interp(ry, yn, eys)

                Hx_all[kf, :] = hx
                
                # Calculate impedance, apparent resistivity and phase
                rhoyx[kf, :], phsyx[kf, :] = self.mt2dzyx(
                    self.freq[kf], hxr, eyr)
            return rhoyx, phsyx, Hx_all
            
        elif mode == "TETM":
            # First calculate TE mode
            sig1 = torch.tensor(sig, dtype=torch.float32, requires_grad=True, device=self.device)
            sig_diff1 = torch.tensor(sig_diff, dtype=torch.float32, requires_grad=True, device=self.device)

            for kf in range(0, self.nf):
                ex = self._mt2dte(self.freq[kf], dy, dz, sig1, sig_diff1, n_add)
                ex1 = self.mt2dte(self.freq[kf], dy, dz, sig, sig_diff, n_add)
                ex_real = ex.real
                ex_imag = ex.imag # shape (83,83,2), dtype=float32
                
                grad_real = torch.autograd.grad(
                            outputs=ex_real,       # Target: real/imag part of y (scalar field)
                            inputs=sig1,                   # Differentiation target: x
                            grad_outputs=torch.ones_like(ex_real),  # Gradient weight all 1 (sum scalar field then differentiate)
                            create_graph=False,          # Retain graph if backprop needed later
                            retain_graph=True,          # Keep graph for multiple differentiations
                            only_inputs=True,            # Return gradient for x only
                            allow_unused=True       

                        )[0]  # grad is tuple, take 0th element
                grad_imag = torch.autograd.grad(
                            outputs=ex_imag,       
                            inputs=sig1,                   
                            grad_outputs=torch.ones_like(ex_imag),  
                            create_graph=False,          
                            retain_graph=True,          
                            only_inputs=True            
                            
                        )[0]  
                # Combine complex gradient and store               
                grad_complex = grad_real + 1j * grad_imag 
                grad_te[kf,:,:] = grad_complex.cpu().detach().numpy()

                # Calculate magnetic field components Hy and Hz from Ex
                ex = ex.cpu().detach().numpy()
                hys, hzs = self.mt2dhyhz(self.freq[kf], dy, dz, sig, ex)
                exs = ex[nza, :]
                exr = np.interp(ry, yn, exs)
                hyr = np.interp(ry, yn, hys)
                Ex_all[kf, :,:] = ex
                rhoxy[kf, :], phsxy[kf, :] = self.mt2dzxy(
                    self.freq[kf], exr, hyr)
            
            # Then calculate TM mode (remove air layers)
            dz = self.dz[nza:]
            sig = self.sig[nza:, :]
            sig_diff = self.sig_diff[nza:, :]

            sig1 = torch.tensor(sig, dtype=torch.float32, requires_grad=True, device=self.device)
            sig_diff1 = torch.tensor(sig_diff, dtype=torch.float32, requires_grad=True, device=self.device)

            for kf in range(0, self.nf):
                hx = self._mt2dtm(self.freq[kf], dy, dz, sig1, sig_diff1, n_add)
                hx_real = hx.real
                hx_imag = hx.imag 
                grad_real = torch.autograd.grad(
                            outputs=hx_real,       
                            inputs=sig1,                   
                            grad_outputs=torch.ones_like(hx_real),  
                            create_graph=False,          
                            retain_graph=True,          
                            only_inputs=True,            
                            allow_unused=True
                        )[0]  
                grad_imag = torch.autograd.grad(
                            outputs=hx_imag,       
                            inputs=sig1,                   
                            grad_outputs=torch.ones_like(hx_imag),  
                            create_graph=False,          
                            retain_graph=True,          
                            only_inputs=True,            
                            allow_unused=True
                        )[0]  

                # Combine complex gradient and store               
                grad_complex = grad_real + 1j * grad_imag 
                grad_tm[kf,:,:] = grad_complex.cpu().detach().numpy()

                hx = hx.cpu().detach().numpy()
                eys, ezs = self.mt2deyez(self.freq[kf], dy, dz, sig, hx)
                hxs = hx[0,:]
                hxr = np.interp(ry, yn, hxs)
                eyr = np.interp(ry, yn, eys)

                Hx_all[kf, :,:] = hx

                rhoyx[kf, :], phsyx[kf, :] = self.mt2dzyx(
                    self.freq[kf], hxr, eyr)
            
            return rhoxy, phsxy, rhoyx, phsyx, Ex_all, Hx_all, grad_te, grad_tm

    def _mt2dte(self, freq, dy, dz, sig, sig_diff, n_add):
        '''
        Calculate secondary electric field for TE mode (Tensor version)
        Aligned with NumPy version (mt2dte) calculation rules
        
        Parameters:
        freq : frequency
        dy, dz : y and z direction grid spacing
        sig : total conductivity
        sig_diff : anomalous conductivity
        n_add : 1D field interpolation points
        
        Returns:
        ex0d : total electric field (primary + secondary field)
        '''
        # 1. Convert to tensor and unify device/data type
        device = self.device
        fdtype = torch.float32  # Unified float type
        cdtype = torch.complex64  # Unified complex type
        
        # Convert basic parameters to tensors
        freq = torch.tensor(freq, dtype=fdtype, device=device)
        dy = torch.tensor(dy, dtype=fdtype, device=device)
        dz = torch.tensor(dz, dtype=fdtype, device=device)
        miu = self.miu.to(fdtype).to(device) if isinstance(self.miu, torch.Tensor) else torch.tensor(self.miu, dtype=fdtype, device=device)

        omega = (2.0 * torch.pi * freq).to(fdtype).to(device)  # Angular frequency
        ny = self.ny - 1  # Y-direction grid count
        nz = self.nz - 1  # Z-direction grid count
        
        # 1. Build system matrix
        # Create 2D grid of spacing (replace np.meshgrid)
        dy0, dz0 = torch.meshgrid(dy, dz, indexing='ij')
        dy0 = dy0.permute(1, 0)  # Adjust dimension order to match NumPy default
        dz0 = dz0.permute(1, 0)
        
        # Calculate coefficients for central difference
        dyc = (dy0[:nz-1, :ny-1] + dy0[:nz-1, 1:ny]) / 2.0
        dzc = (dz0[:nz-1, :ny-1] + dz0[1:nz, :ny-1]) / 2.0

        # Optimized: merge slicing to reduce intermediate tensors
        dy_slice = dy0[:nz-1, :ny-1]
        dy_slice1 = dy0[:nz-1, 1:ny]
        dz_slice = dz0[:nz-1, :ny-1]
        dz_slice1 = dz0[1:nz, :ny-1]
        
        w1 = dy_slice * dz_slice
        w2 = dy_slice1 * dz_slice
        w3 = dy_slice * dz_slice1
        w4 = dy_slice1 * dz_slice1
        area = (w1 + w2 + w3 + w4) / 4.0  # Cell area
        
        # Calculate conductivity at cell center (weighted average)
        sigc = (sig[:nz-1, :ny-1] * w1 + sig[:nz-1, 1:ny] * w2 + 
                sig[1:nz, :ny-1] * w3 + sig[1:nz, :ny-1] * w4) / (area * 4.0)
        
        # Construct diagonal elements
        val = dzc / dy0[:nz-1, :ny-1] + dzc / dy0[:nz-1, 1:ny] + dyc / dz0[:nz-1, :ny-1] + dyc / dz0[1:nz, :ny-1]
        II = torch.tensor(1j, dtype=torch.complex64, device=device)
        mtx1 = (II * omega * miu * sigc * area).to(cdtype) - val.to(cdtype)
        mtx1 = mtx1.t().flatten()  # Flatten column-major (replace numpy flatten('F'))
        
        # Construct upper/lower diagonals (z-direction neighbors)
        mtx20 = dyc[1:nz-1, :ny-1] / dz0[1:nz-1, :ny-1]
        mtx2 = torch.cat((mtx20, torch.zeros((1, ny-1), device=device, dtype=fdtype)), dim=0)
        mtx2 = mtx2.t().flatten()[:-1].to(cdtype)  # Remove last element
        
        # Construct left/right diagonals (y-direction neighbors)
        mtx3 = dzc[:nz-1, 1:ny-1] / dy0[:nz-1, 1:ny-1].to(cdtype)
        mtx3 = mtx3.t().flatten().to(cdtype)
        k2 = nz  # Index difference for y-direction neighbors
        
        # Assemble matrix (replace scipy sparse matrix)
        n_total = (nz-1) * (ny-1)
        A_mat = torch.diag(mtx1)
        if mtx2.numel() > 0 and mtx2.numel() == n_total - 1:
            A_mat = A_mat + torch.diag(mtx2, diagonal=-1) + torch.diag(mtx2, diagonal=1)
        if mtx3.numel() > 0 and mtx3.numel() == n_total - k2:
            A_mat = A_mat + torch.diag(mtx3, diagonal=1-k2) + torch.diag(mtx3, diagonal=k2-1)
        
        # 2. Calculate primary field (background field)
        ex1d, _ = self._mt1dte(freq, dz, (sig - sig_diff)[:, 0], n_add)
        ex1d = ex1d.to(cdtype).unsqueeze(1).expand(-1, ny + 1)
        
        # 3. Calculate right-hand side (source term for anomalous field equation)
        sigc_diff = (sig_diff[:nz-1, :ny-1] * w1 + sig_diff[:nz-1, 1:ny] * w2 + 
                    sig_diff[1:nz, :ny-1] * w3 + sig_diff[1:nz, :ny-1] * w4) / (area * 4.0)
        
        coef = (II * omega * miu * sigc_diff * area).to(cdtype)  
        rhs = (-coef * ex1d[1:nz, 1:ny]).t().flatten().unsqueeze(1)
        
        # 4. Solve linear system to get secondary field
        ex2d_flat = torch.linalg.solve(A_mat, rhs).squeeze(1)
        ex2d = ex2d_flat.view(ny-1, nz-1).t()  # Reshape and transpose to match column-major
        
        # 5. Reconstruct total electric field (primary + secondary)
        ex0d = ex1d.clone()
        ex0d[1:nz, 1:ny] = ex1d[1:nz, 1:ny] + ex2d
        
        return ex0d.contiguous()

    def mt2dte(self, freq, dy, dz, sig, sig_diff, n_add):
        '''
        Calculate secondary electric field for TE mode
        
        Parameters:
        freq : frequency
        dy, dz : y and z direction grid spacing
        sig : total conductivity
        sig_diff : anomalous conductivity
        n_add : 1D field interpolation points
        
        Returns:
        ex0d : total electric field (primary + secondary field)
        '''
        omega = 2.0 * np.pi * freq  # Angular frequency
        ny = self.ny - 1  # Y-direction grid count
        nz = self.nz - 1  # Z-direction grid count
        
        # 1. Build system matrix
        # Create 2D grid of spacing
        dy0, dz0 = np.meshgrid(dy, dz)
        
        # Calculate coefficients for central difference
        dyc = (dy0[0:nz-1, 0:ny-1] + dy0[0:nz-1, 1:ny]) / 2.0
        dzc = (dz0[0:nz-1, 0:ny-1] + dz0[1:nz, 0:ny-1]) / 2.0
        
        # Calculate weights at different positions
        w1 = dy0[0:nz-1, 0:ny-1] * dz0[0:nz-1, 0:ny-1]
        w2 = dy0[0:nz-1, 1:ny] * dz0[0:nz-1, 0:ny-1]
        w3 = dy0[0:nz-1, 0:ny-1] * dz0[1:nz, 0:ny-1]
        w4 = dy0[0:nz-1, 1:ny] * dz0[1:nz, 0:ny-1]
        
        # Cell area
        area = (w1 + w2 + w3 + w4) / 4.0
        
        # Calculate conductivity at cell center (weighted average)
        sigc = (sig[0:nz-1, 0:ny-1] * w1 + sig[0:nz-1, 1:ny] * w2 + 
                sig[1:nz, :ny-1] * w3 + sig[1:nz, 1:ny] * w4) / (area * 4.0)
        
        # Construct diagonal elements
        val = dzc / dy0[0:nz-1, 0:ny-1] + dzc / dy0[0:nz-1, 1:ny] + \
              dyc / dz0[0:nz-1, 0:ny-1] + dyc / dz0[1:nz, 0:ny-1]
        mtx1 = self.II * omega * self.miu * sigc * area - val
        mtx1 = mtx1.flatten('F')  # Flatten column-major to 1D array
        
        # Construct upper/lower diagonals (z-direction neighbors)
        mtx20 = dyc[1:nz-1, 0:ny-1] / dz0[1:nz-1, 0:ny-1]
        mtx2 = np.concatenate((mtx20, np.zeros((1, ny-1))), 0)
        mtx2 = mtx2.flatten('F')[:-1]  # Remove last element
        
        # Construct left/right diagonals (y-direction neighbors)
        mtx3 = dzc[0:nz-1, 1:ny-1] / dy0[0:nz-1, 1:ny-1]
        mtx3 = mtx3.flatten('F')
        k2 = nz  # Index difference for y-direction neighbors
        
        # Assemble sparse matrix
        mtxA = scipa.diags(mtx1, format='csc') + \
               scipa.diags(mtx2, -1, format='csc') + scipa.diags(mtx2, 1, format='csc') + \
               scipa.diags(mtx3, 1 - k2, format='csc') + scipa.diags(mtx3, k2 - 1, format='csc')
        
        # 2. Calculate primary field (background field)
        ex1d, _ = self.mt1dte(freq, dz, (sig - sig_diff)[:, 0], n_add)
        ex1d = ex1d.reshape(-1, 1) * np.ones((nz + 1, ny + 1))  # Extend to 2D
        
        # 3. Calculate right-hand side (source term for anomalous field equation)
        # Calculate weighted average of anomalous conductivity
        sigc_diff = (sig_diff[0:nz-1, 0:ny-1] * w1 + sig_diff[0:nz-1, 1:ny] * w2 + 
                    sig_diff[1:nz, :ny-1] * w3 + sig_diff[1:nz, 1:ny] * w4) / (area * 4.0)
        
        coef = self.II * omega * self.miu * sigc_diff * area
        rhs = coef * ex1d[1:nz, 1:ny]  # Source term
        rhs = -rhs  # Equation transformation
        rhs = rhs.reshape((ny-1) * (nz-1), 1, order='F')  # Flatten column-major
        
        # 4. Solve linear system to get secondary field
        ex, _ = self.equation_solve(mtxA, rhs)
        
        # 5. Reconstruct total electric field (primary + secondary)
        ex2d = ex.reshape(nz-1, ny-1, order='F')  # Reshape to 2D
        ex0d = ex1d  # Initialize total field (equal to primary field)
        ex0d[1:nz, 1:ny] = ex1d[1:nz, 1:ny] + ex2d  # Add secondary field
        
        return ex0d

    def _mt2dtm(self, freq, dy, dz, sig, sig_diff, n_add):
        """TM mode secondary magnetic field calculation (Tensor version, differentiable, dtype-safe)"""

        device = self.device
        fdtype = sig.dtype  # float32
        cdtype = torch.complex64

        # --- Standardize input tensors ---
        freq = torch.as_tensor(freq, dtype=fdtype, device=device)
        dy   = torch.as_tensor(dy,   dtype=fdtype, device=device)
        dz   = torch.as_tensor(dz,   dtype=fdtype, device=device)
        sig_diff = torch.as_tensor(sig_diff, dtype=fdtype, device=device)

        omega = (2.0 * torch.pi * freq).to(fdtype)
        II = torch.tensor(1j, dtype=cdtype, device=device)
        miu = self.miu.to(fdtype).to(device) if isinstance(self.miu, torch.Tensor) else torch.tensor(self.miu, dtype=fdtype, device=device)

        ny = len(dy)  # Y-direction grid count
        nz = len(dz)  # Z-direction grid count
        
        # 1. Build grid
        dy0, dz0 = torch.meshgrid(dy, dz, indexing='ij')   # (ny, nz)
        dy0 = dy0.permute(1, 0)  # (nz, ny)
        dz0 = dz0.permute(1, 0)  # (nz, ny)

        # Difference coefficients
        dyc = (dy0[:nz-1, :ny-1] + dy0[:nz-1, 1:ny]) / 2.0
        dzc = (dz0[:nz-1, :ny-1] + dz0[1:nz, :ny-1]) / 2.0

        # Weights
        w1 = 2 * dz0[:nz-1, :ny-1]
        w2 = 2 * dz0[1:nz, :ny-1]
        w3 = 2 * dy0[:nz-1, :ny-1]
        w4 = 2 * dy0[:nz-1, 1:ny]

        area = (w1 + w2 + w3 + w4) / 4.0

        # Difference coefficients A,B,C,D
        A = (1.0/sig[:nz-1,:ny-1] * dy0[:nz-1,:ny-1] +
            1.0/sig[:nz-1,1:ny]  * dy0[:nz-1,1:ny]) / w1

        B = (1.0/sig[1:nz,:ny-1] * dy0[:nz-1,:ny-1] +
            1.0/sig[1:nz,1:ny]  * dy0[:nz-1,1:ny]) / w2

        C = (1.0/sig[:nz-1,:ny-1] * dz0[:nz-1,:ny-1] +
            1.0/sig[1:nz,:ny-1]  * dz0[1:nz,:ny-1]) / w3

        D = (1.0/sig[:nz-1,1:ny] * dz0[:nz-1,:ny-1] +
            1.0/sig[1:nz,1:ny]  * dz0[1:nz,:ny-1]) / w4

        # Diagonal elements
        diag = (II * omega * miu * dyc * dzc).to(cdtype) - (A + B + C + D).to(cdtype)
        mtx1 = diag.t().flatten()  

        # Upper/lower diagonals (z-direction)
        if nz-1 > 1:
            mtx20 = B[:nz-2, :ny-1]
            mtx2 = torch.cat([mtx20, torch.zeros((1, ny-1), device=device, dtype=fdtype)], dim=0)
            mtx2 = mtx2.t().flatten()[:-1].to(cdtype)
        else:
            mtx2 = torch.tensor([], device=device, dtype=cdtype)

        # Left/right diagonals (y-direction)
        if ny-1 > 1:
            mtx3 = D[:nz-1, :ny-2].t().flatten().to(cdtype)
        else:
            mtx3 = torch.tensor([], device=device, dtype=cdtype)

        k2 = nz
        n_total = (nz-1) * (ny-1)

        # Assemble matrix
        A_mat = torch.diag(mtx1)
        if mtx2.numel() > 0 and mtx2.numel() == n_total - 1:
            A_mat = A_mat + torch.diag(mtx2, diagonal=-1) + torch.diag(mtx2, diagonal=1)
        if mtx3.numel() > 0 and mtx3.numel() == n_total - k2:
            A_mat = A_mat + torch.diag(mtx3, diagonal=1-k2) + torch.diag(mtx3, diagonal=k2-1)

        # 2. Primary field (background field)
        ey1d, hx1d = self._mt1dtm(freq, dz, (sig - sig_diff)[:, 0], n_add)
        ey1d = ey1d.to(cdtype).unsqueeze(1).expand(-1, ny+1)
        hx1d = hx1d.to(cdtype).unsqueeze(1).expand(-1, ny+1)

        # 3. Right-hand side
        A1 = dy0[:nz-1,:ny-1] * dz0[:nz-1,:ny-1]
        A2 = dy0[:nz-1,1:ny]  * dz0[:nz-1,:ny-1]
        A3 = dy0[:nz-1,:ny-1] * dz0[1:nz,:ny-1]
        A4 = dy0[:nz-1,1:ny]  * dz0[1:nz,:ny-1]
        area = (A1 + A2 + A3 + A4) / 4.0

        sig_scale = sig_diff / sig
        sigc_diff = (sig_scale[:nz-1,:ny-1]*A1 + sig_scale[:nz-1,1:ny]*A2 +
                    sig_scale[1:nz,:ny-1]*A3 + sig_scale[1:nz,1:ny]*A4) / (area * 4.0)

        coef = (II * omega * miu * sigc_diff * area).to(cdtype)
        rhs = (coef * hx1d[1:nz,1:ny]).t().flatten().unsqueeze(1)

        # Additional derivative term
        sigc_t = (sig_scale[:nz-1,:ny-1]*dy0[:nz-1,:ny-1] + sig_scale[:nz-1,1:ny]*dy0[:nz-1,1:ny]) / (dy0[:nz-1,:ny-1] + dy0[:nz-1,1:ny])
        sigc_b = (sig_scale[1:nz,:ny-1]*dy0[1:nz,:ny-1] + sig_scale[1:nz,1:ny]*dy0[1:nz,1:ny]) / (dy0[1:nz,:ny-1] + dy0[1:nz,1:ny])
        ey_d = (sigc_b - sigc_t) / ((dz0[:nz-1,:ny-1] + dz0[1:nz,:ny-1])/2.0) * area * ey1d[1:nz,1:ny]

        rhs = -(rhs - ey_d.t().flatten().unsqueeze(1))

        # 4. Solve linear system
        hx2d_flat = torch.linalg.solve(A_mat, rhs).squeeze(1)
        hx2d = hx2d_flat.view(ny-1, nz-1).t()

        # 5. Reconstruct total field
        hx0d = hx1d.clone()
        hx0d[1:nz,1:ny] = hx1d[1:nz,1:ny] + hx2d

        return hx0d.contiguous()

    def mt2dtm(self, freq, dy, dz, sig, sig_diff, n_add):
        '''
        Calculate secondary magnetic field for TM mode
        
        Parameters:
        freq : frequency
        dy, dz : y and z direction grid spacing
        sig : total conductivity
        sig_diff : anomalous conductivity
        n_add : 1D field interpolation points
        
        Returns:
        hx0d : total magnetic field (primary + secondary field)
        '''
        omega = 2.0 * np.pi * freq  # Angular frequency
        ny = len(dy)  # Y-direction grid count
        nz = len(dz)  # Z-direction grid count
               

        # 1. Build system matrix
        # Create 2D grid of spacing
        dy0, dz0 = np.meshgrid(dy, dz)
        
        # Calculate coefficients for central difference
        dyc = (dy0[0:nz-1, 0:ny-1] + dy0[0:nz-1, 1:ny]) / 2.0
        dzc = (dz0[0:nz-1, 0:ny-1] + dz0[1:nz, 0:ny-1]) / 2.0
        
        # Calculate weights
        w1 = 2 * dz0[0:nz-1, 0:ny-1]
        w2 = 2 * dz0[1:nz, 0:ny-1]
        w3 = 2 * dy0[0:nz-1, 0:ny-1]
        w4 = 2 * dy0[0:nz-1, 1:ny]
        
        # Cell area
        area = (w1 + w2 + w3 + w4) / 4.0
        
        # Calculate difference coefficients
        A = (1.0/sig[0:nz-1,0:ny-1] * dy0[0:nz-1,0:ny-1] + 
             1.0/sig[0:nz-1,1:ny] * dy0[0:nz-1,1:ny]) / w1
        B = (1.0/sig[1:nz,0:ny-1] * dy0[0:nz-1,0:ny-1] + 
             1.0/sig[1:nz,1:ny] * dy0[0:nz-1,1:ny]) / w2
        C = (1.0/sig[0:nz-1,0:ny-1] * dz0[0:nz-1,0:ny-1] + 
             1.0/sig[1:nz,0:ny-1] * dz0[1:nz,0:ny-1]) / w3
        D = (1.0/sig[0:nz-1,1:ny] * dz0[0:nz-1,0:ny-1] + 
             1.0/sig[1:nz,1:ny] * dz0[1:nz,0:ny-1]) / w4
        
        # Construct diagonal elements
        mtx1 = self.II * omega * self.miu * dyc * dzc - A - B - C - D
        mtx1 = mtx1.flatten('F')  # Flatten column-major
        
        # Construct upper/lower diagonals (z-direction neighbors)
        mtx20 = B[0:nz-2, 0:ny-1]
        mtx2 = np.concatenate((mtx20, np.zeros((1, ny-1))), 0)
        mtx2 = mtx2.flatten('F')[:-1]
        
        # Construct left/right diagonals (y-direction neighbors)
        mtx3 = D[0:nz-1, 0:ny-2]
        mtx3 = mtx3.flatten('F')
        k2 = nz  # Index difference for y-direction neighbors
        
        # Assemble sparse matrix
        mtxA = scipa.diags(mtx1, format='csc') + \
               scipa.diags(mtx2, -1, format='csc') + scipa.diags(mtx2, 1, format='csc') + \
               scipa.diags(mtx3, 1 - k2, format='csc') + scipa.diags(mtx3, k2 - 1, format='csc')
        
        # 2. Calculate primary field (background field)
        ey1d, hx1d = self.mt1dtm(freq, dz, (sig - sig_diff)[:, 0], n_add)
        ey1d = ey1d.reshape(-1, 1) * np.ones((nz + 1, ny + 1))  # Extend to 2D
        hx1d = hx1d.reshape(-1, 1) * np.ones((nz + 1, ny + 1))  # Extend to 2D
        
        # 3. Calculate right-hand side (source term for anomalous field equation)
        A1 = dy0[0:nz-1, 0:ny-1] * dz0[0:nz-1, 0:ny-1]
        A2 = dy0[0:nz-1, 1:ny] * dz0[0:nz-1, 0:ny-1]
        A3 = dy0[0:nz-1, 0:ny-1] * dz0[1:nz, 0:ny-1]
        A4 = dy0[0:nz-1, 1:ny] * dz0[1:nz, 0:ny-1]
        area = (A1 + A2 + A3 + A4) / 4.0
        
        # Calculate weighted average of anomalous conductivity ratio
        sig_scale = sig_diff / sig
        sigc_diff = (sig_scale[0:nz-1,0:ny-1]*A1 + sig_scale[0:nz-1,1:ny]*A2 + 
                    sig_scale[1:nz,0:ny-1]*A3 + sig_scale[1:nz,1:ny]*A4) / (area * 4.0)
        
        coef = self.II * omega * self.miu * sigc_diff * area
        rhs = coef * hx1d[1:nz, 1:ny]
        
        # Add derivative term contribution
        sigc_t = (sig_scale[0:nz-1,0:ny-1] * dy0[0:nz-1,0:ny-1] + 
                 sig_scale[0:nz-1,1:ny] * dy0[0:nz-1,1:ny]) / (dy0[0:nz-1,0:ny-1] + dy0[0:nz-1,1:ny])
        sigc_b = (sig_scale[1:nz,0:ny-1] * dy0[1:nz,0:ny-1] + 
                 sig_scale[1:nz,1:ny] * dy0[1:nz,1:ny]) / (dy0[1:nz,0:ny-1] + dy0[1:nz,1:ny])
        ey_d = (sigc_b - sigc_t) / ((dz0[0:nz-1,0:ny-1] + dz0[1:nz,0:ny-1])/2.0) * area * ey1d[1:nz,1:ny]
        rhs = rhs - ey_d
        
        rhs = -rhs  # Equation transformation
        rhs = rhs.reshape((ny-1) * (nz-1), 1, order='F')  # Flatten column-major
        
        # 4. Solve linear system to get secondary field
        hx, _ = self.equation_solve(mtxA, rhs)
        
        # 5. Reconstruct total magnetic field (primary + secondary)
        hx2d = hx.reshape(nz-1, ny-1, order='F')  # Reshape to 2D
        hx0d = hx1d  # Initialize total field (equal to primary field)
        hx0d[1:nz, 1:ny] = hx1d[1:nz, 1:ny] + hx2d  # Add secondary field
        
        return hx0d

    def mt2dhyhz(self, freq, dy, dz, sig, ex):
        """
        Calculate magnetic fields Hy and Hz from electric field Ex (TE mode)
        
        Parameters:
        freq : frequency
        dy, dz : y and z direction grid spacing
        sig : conductivity
        ex : electric field component Ex
        
        Returns:
        hys : y-direction magnetic field component Hy
        hzs : z-direction magnetic field component Hz
        """
        omega = 2.0 * np.pi * freq
        ny = np.size(dy)  # Y-direction grid count
        
        # 1. Calculate Hy
        hys = np.zeros((ny + 1), dtype=complex)
        
        # 1.1 Calculate Hy at top-left corner
        kk = self.nza  # Index at bottom of air layers
        delz = dz[kk]
        sigc = sig[kk, 0]
        c0 = -1.0 / (self.II * omega * self.miu * delz) + (3.0 / 8.0) * sigc * delz
        c1 = 1.0 / (self.II * omega * self.miu * delz) + (1.0 / 8.0) * sigc * delz
        hys[0] = c0 * ex[kk, 0] + c1 * ex[kk + 1, 0]
        
        # 1.2 Calculate Hy at top-right corner
        sigc = sig[kk, ny - 1]
        c0 = -1.0 / (self.II * omega * self.miu * delz) + (3.0 / 8.0) * sigc * delz
        c1 = 1.0 / (self.II * omega * self.miu * delz) + (1.0 / 8.0) * sigc * delz
        hys[ny] = c0 * ex[kk, ny] + c1 * ex[kk + 1, ny]
        
        # 1.3 Calculate Hy at other positions
        dyj = dy[0:ny-1] + dy[1:ny]
        sigc = (sig[kk, 0:ny-1] * dy[0:ny-1] + sig[kk, 1:ny] * dy[1:ny]) / dyj
        cc = delz / (4.0 * self.II * omega * self.miu * dyj)
        c0 = -1.0/(self.II*omega*self.miu*delz) + (3.0/8.0)*sigc*delz - cc*3.0*(1.0/dy[1:ny]+1.0/dy[0:ny-1])
        c1 = 1.0/(self.II*omega*self.miu*delz) + (1.0/8.0)*sigc*delz - cc*1.0*(1.0/dy[1:ny]+1.0/dy[0:ny-1])
        c0l = 3.0 * cc / dy[0:ny-1]
        c0r = 3.0 * cc / dy[1:ny]
        c1l = 1.0 * cc / dy[0:ny-1]
        c1r = 1.0 * cc / dy[1:ny]
        
        hys[1:ny] = c0l * ex[kk, 0:ny-1] + c0 * ex[kk, 1:ny] + c0r * ex[kk, 2:ny+1] + \
                    c1l * ex[kk+1, 0:ny-1] + c1 * ex[kk+1, 1:ny] + c1r * ex[kk+1, 2:ny+1]
        
        # 2. Calculate Hz
        hzs = np.zeros((ny + 1), dtype=complex)
        
        # 2.1 Calculate Hz at left/right boundaries
        hzs[0] = -1.0 / (self.II * omega * self.miu) * (ex[kk, 1] - ex[kk, 0]) / dy[0]
        hzs[ny] = -1.0 / (self.II * omega * self.miu) * (ex[kk, ny] - ex[kk, ny-1]) / dy[ny-1]
        
        # 2.2 Calculate Hz at other positions
        hzs[1:ny] = -1.0 / (self.II * omega * self.miu) * (ex[kk, 2:ny+1] - ex[kk, 0:ny-1]) / (dy[0:ny-1] + dy[1:ny])
        
        return hys, hzs

    def mt2deyez(self, freq, dy, dz, sig, hx):
        """
        Calculate electric fields Ey and Ez from magnetic field Hx (TM mode)
        
        Parameters:
        freq : frequency
        dy, dz : y and z direction grid spacing
        sig : conductivity
        hx : magnetic field component Hx
        
        Returns:
        eys : y-direction electric field component Ey
        ezs : z-direction electric field component Ez
        """
        omega = 2.0 * np.pi * freq
        ny = np.size(dy)  # Y-direction grid count
        
        # 1. Calculate Ey
        eys = np.zeros((ny + 1), dtype=complex)
        
        # 1.1 Calculate Ey at top-left corner (no air layers, start from surface)
        kk = 0
        delz = dz[kk]
        sigc = sig[kk, 0]
        temp_beta = self.II * omega * self.miu * delz
        temp_1 = sigc * delz
        c0 = -1.0 / temp_1 + (3.0 / 8.0) * temp_beta
        c1 = 1.0 / temp_1 + (1.0 / 8.0) * temp_beta
        eys[0] = c0 * hx[kk, 0] + c1 * hx[kk + 1, 0]
        
        # 1.2 Calculate Ey at top-right corner
        sigc = sig[kk, ny - 1]
        temp_1 = sigc * delz
        c0 = -1.0 / temp_1 + (3.0 / 8.0) * temp_beta
        c1 = 1.0 / temp_1 + (1.0 / 8.0) * temp_beta
        eys[ny] = c0 * hx[kk, ny] + c1 * hx[kk + 1, ny]
        
        # 1.3 Calculate Ey at other positions
        dyj = (dy[0:ny-1] + dy[1:ny]) / 2.0
        tao = 1.0 / sig[kk, 0:ny]
        taoc = (tao[0:ny-1] * dy[0:ny-1] + tao[1:ny] * dy[1:ny]) / (2 * dyj)
        temp_1 = self.II * omega * self.miu * delz
        temp_2 = taoc / delz
        temp_3 = delz / dyj
        temp_4 = tao / dy
        
        c0 = (3.0 / 8.0) * temp_1 - temp_2
        c1 = (1.0 / 8.0) * temp_1 + temp_2 - (1.0 / 8.0) * temp_3 * (temp_4[0:ny-1] + temp_4[1:ny])
        c1l = (1.0 / 8.0) * temp_3 * temp_4[0:ny-1]
        c1r = (1.0 / 8.0) * temp_3 * temp_4[1:ny]
        
        eys[1:ny] = c0 * hx[kk, 1:ny] + c1l * hx[kk+1, 0:ny-1] + c1 * hx[kk+1, 1:ny] + c1r * hx[kk+1, 2:ny+1]
        
        # 2. Initialize Ez (calculation not implemented here)
        ezs = np.zeros((ny + 1), dtype=complex)
        
        return eys, ezs
        
    def mt2dzxy(self, freq, exr, hyr):
        """
        Calculate impedance Zxy, apparent resistivity and phase for TE mode
        
        Parameters:
        freq : frequency
        exr : electric field Ex at observation points
        hyr : magnetic field Hy at observation points
        
        Returns:
        rhote : TE mode apparent resistivity
        phste : TE mode phase
        """
        omega = 2.0 * np.pi * freq
        
        # Calculate impedance (Zxy = Ex / Hy)
        zxy = np.array(exr / hyr, dtype=complex)
        
        # Calculate apparent resistivity (rho = |Z|^2 / (ωμ))
        rhote = abs(zxy) ** 2 / (omega * self.miu)
        
        # Calculate phase (convert radians to degrees)
        phste = np.arctan2(zxy.imag, zxy.real) * 180.0 / np.pi
        
        return rhote, phste

    def mt2dzyx(self, freq, hxr, eyr):
        """
        Calculate impedance Zyx, apparent resistivity and phase for TM mode
        
        Parameters:
        freq : frequency
        hxr : magnetic field Hx at observation points
        eyr : electric field Ey at observation points
        
        Returns:
        rhotm : TM mode apparent resistivity
        phstm : TM mode phase
        """
        omega = 2.0 * np.pi * freq
        
        # Calculate impedance (Zyx = Ey / Hx)
        zyx = np.array(eyr / hxr, dtype=complex)
        
        # Calculate apparent resistivity
        rhotm = abs(zyx) ** 2 / (omega * self.miu)
        
        # Calculate phase
        phstm = np.arctan2(zyx.imag, zyx.real) * 180.0 / np.pi
        
        return rhotm, phstm

    def _mt1dte(self, freq, dz0, sig0, n_add):
        """
        Calculate 1D electric and magnetic fields for TE mode (used as primary field for 2D calculation)
        
        Parameters:
        freq : frequency
        dz0 : original z-direction grid spacing
        sig0 : 1D model conductivity
        n_add : interpolation points per original grid
        
        Returns:
        ex : interpolated electric field Ex
        hy : interpolated magnetic field Hy
        """
        # Convert inputs to torch tensor
        device = sig0.device
        fdtype = sig0.dtype if isinstance(sig0, torch.Tensor) else torch.float32
        cdtype = torch.complex64 if fdtype == torch.float32 else torch.complex128
        
        omega = (2.0 * torch.pi * freq).to(fdtype).to(device)
        II = torch.tensor(1j, dtype=torch.complex64, device=device)
        miu = self.miu if isinstance(self.miu, torch.Tensor) else torch.tensor(self.miu, dtype=fdtype, device=device)
        if isinstance(miu, torch.Tensor):
            miu = miu.to(fdtype).to(device)
        
        # Refine grid to improve 1D field calculation accuracy
        dz = torch.repeat_interleave(dz0 / n_add, n_add)
        sig = torch.repeat_interleave(sig0, n_add)
        nz = sig.numel()
        
        # Add an extra layer for bottom boundary condition
        sig = torch.cat((sig, sig[-1].unsqueeze(0)))
        boundary_dz = torch.sqrt(2.0 / (sig[-1] * omega * miu)).to(fdtype)
        dz = torch.cat((dz, boundary_dz.unsqueeze(0)))
        
        # Build 1D TE mode system matrix
        sig_top = sig[:nz]
        sig_bot = sig[1:nz+1]
        dz_top = dz[:nz]
        dz_bot = dz[1:nz+1]
        
        diagA = (
            II * omega * miu * (sig_top * dz_top + sig_bot * dz_bot)
            - (2.0 / dz_top + 2.0 / dz_bot).to(cdtype)
        ).to(cdtype) 
        offdiagA = 2.0 / dz[1:nz]
        offdiagA = offdiagA.to(cdtype)
        
        # Assemble matrix (replace scipy sparse matrix)
        n = diagA.numel()
        mtxA = torch.diag(diagA)
        if offdiagA.numel() > 0:
            mtxA += torch.diag(offdiagA, diagonal=1)
            mtxA += torch.diag(offdiagA, diagonal=-1)
        
        # Right-hand side (boundary conditions: Ex=1 at top, Ex=0 at bottom)
        rhs = torch.zeros((nz, 1), dtype=cdtype, device=device)
        rhs[0] = (-2.0 / dz[0]).to(cdtype)  
        
        # Solve linear system (replace scipy splu)
        ex0 = torch.linalg.solve(mtxA, rhs)
        
        # Reconstruct electric field (include top boundary condition)
        ex = torch.cat((torch.tensor([1.0], dtype=cdtype, device=device), ex0.squeeze()))
        
        # Calculate magnetic field Hy (from gradient of Ex)
        hy0 = (ex[1:] - ex[:-1]) / dz[:-1].to(cdtype) / II / omega.to(cdtype) / miu.to(cdtype)
        hy = torch.cat((hy0, hy0[-1:]))
        
        # Extract results at original grid points (downsample)
        idx = torch.arange(sig0.numel() + 1, device=device)*n_add
        return ex[idx].contiguous(), hy[idx].contiguous()

    def mt1dte(self, freq, dz0, sig0, n_add):
        """
        Calculate 1D electric and magnetic fields for TE mode (used as primary field for 2D calculation)
        
        Parameters:
        freq : frequency
        dz0 : original z-direction grid spacing
        sig0 : 1D model conductivity
        n_add : interpolation points per original grid
        
        Returns:
        ex : interpolated electric field Ex
        hy : interpolated magnetic field Hy
        """
        omega = 2.0 * np.pi * freq
        
        # Refine grid to improve 1D field calculation accuracy
        dz = np.array([dz0[i] / n_add * np.ones(n_add) for i in range(np.size(dz0))]).flatten()
        sig = np.array([sig0[i] * np.ones(n_add) for i in range(np.size(sig0))]).flatten()
        nz = np.size(sig)
        
        # Add an extra layer for bottom boundary condition
        sig = np.hstack((sig, sig[nz-1]))
        dz =  np.hstack((dz, np.array(np.sqrt(2.0 / (sig[nz] * omega * self.miu)), dtype=float)))
        
        # Build 1D TE mode system matrix
        diagA = self.II * omega * self.miu * (sig[0:nz] * dz[0:nz] + sig[1:nz+1] * dz[1:nz+1]) - \
                2.0 / dz[0:nz] - 2.0 / dz[1:nz+1]
        offdiagA = 2.0 / dz[1:nz]
        
        # Assemble sparse matrix
        mtxA = scipa.diags(diagA, format='csc') + \
               scipa.diags(offdiagA, 1, format='csc') + scipa.diags(offdiagA, -1, format='csc')
        
        # Right-hand side (boundary conditions: Ex=1 at top, Ex=0 at bottom)
        rhs = np.zeros((nz, 1), dtype=float)
        rhs[0] = -2.0 / dz[0]
        
        # Solve linear system
        lup = scilg.splu(mtxA)
        ex0 = lup.solve(rhs)
        
        # Reconstruct electric field (include top boundary condition)
        ex = np.array(np.concatenate(([1.0], ex0.reshape(-1))), dtype=complex)
        
        # Calculate magnetic field Hy (from gradient of Ex)
        hy0 = (ex[1:] - ex[:-1]) / dz[:-1] / self.II / omega / self.miu
        hy = np.concatenate((hy0, hy0[-1:]))
        
        # Extract results at original grid points (downsample)
        idx = np.arange(np.size(sig0) + 1) * n_add
        
        return ex[idx], hy[idx]

    def _mt1dtm(self, freq, dz0, sig0, n_add):
        """
        1D TM mode forward modeling (Tensor version, dtype-safe)
        - freq: scalar tensor (real)
        - dz0: 1D tensor of original layer thicknesses (float)
        - sig0: 1D tensor of original layer conductivities (float)
        - n_add: int, refinement factor
        Returns:
        ey: complex64 tensor of E_y on original node points (len(sig0)+1)
        hx: complex64 tensor of H_x on original node points (len(sig0)+1)
        """
        # 1. Convert inputs to torch tensor and unify device/data type
        device = sig0.device if isinstance(sig0, torch.Tensor) else self.device
        fdtype = sig0.dtype if isinstance(sig0, torch.Tensor) else torch.float32
        cdtype = torch.complex64 if fdtype == torch.float32 else torch.complex128
        
        # Basic parameter conversion
        freq = torch.as_tensor(freq, dtype=fdtype, device=device)
        dz0 = torch.as_tensor(dz0, dtype=fdtype, device=device) if not isinstance(dz0, torch.Tensor) else dz0.to(fdtype).to(device)
        sig0 = torch.as_tensor(sig0, dtype=fdtype, device=device) if not isinstance(sig0, torch.Tensor) else sig0.to(fdtype).to(device)
        
        # Permeability processing (constant tensor)
        miu = self.miu if isinstance(self.miu, torch.Tensor) else torch.tensor(self.miu, dtype=fdtype, device=device)
        if isinstance(miu, torch.Tensor):
            miu = miu.to(fdtype).to(device)
        miu.requires_grad = False
        
        omega = (2.0 * torch.pi * freq).to(fdtype).to(device)
        II = torch.tensor(1j, dtype=cdtype, device=device)

        # 2. Refine grid
        dz = torch.repeat_interleave(dz0 / n_add, n_add)
        sig = torch.repeat_interleave(sig0, n_add)
        nz = sig.numel()

        # 3. Add bottom boundary condition
        sig = torch.cat((sig, sig[-1].unsqueeze(0)), dim=0)
        boundary_dz = torch.sqrt(2.0 / (sig[-1] * omega * miu)).to(fdtype)
        dz = torch.cat((dz, boundary_dz.unsqueeze(0)), dim=0)

        # 4. Build 1D TM mode system matrix
        dz_top = dz[:nz]
        dz_bot = dz[1:nz+1]
        sig_top = sig[:nz]
        sig_bot = sig[1:nz+1]
        
        diagA = (
            II * omega * miu * (dz_top + dz_bot)
            - (2.0 / (dz_top * sig_top) + 2.0 / (dz_bot * sig_bot)).to(cdtype)
        ).to(cdtype)
        
        offdiagA = (2.0 / (dz[1:nz] * sig[1:nz])).to(cdtype) if nz > 1 else torch.tensor([], dtype=cdtype, device=device)

        # Assemble matrix
        n = diagA.numel()
        mtxA = torch.diag(diagA)
        if offdiagA.numel() > 0 and offdiagA.numel() == n - 1:
            mtxA += torch.diag(offdiagA, diagonal=1)
            mtxA += torch.diag(offdiagA, diagonal=-1)

        # 5. Right-hand side (apply boundary conditions)
        rhs = torch.zeros((nz, 1), dtype=cdtype, device=device)
        if dz[0] > 1e-12 and sig[0] > 1e-12:
            rhs[0] = (-2.0 / (dz[0] * sig[0])).to(cdtype)

        # 6. Solve linear system
        hx0 = torch.linalg.solve(mtxA, rhs).squeeze(1)

        # 7. Reconstruct magnetic field Hx (include top boundary condition)
        hx_boundary = torch.tensor([1.0+0j], dtype=cdtype, device=device)
        hx = torch.cat((hx_boundary, hx0), dim=0)

        # 8. Calculate electric field Ey (from gradient of Hx)
        if hx.numel() > 1 and dz.numel() >= 1:
            dz_complex = dz[:-1].to(cdtype)
            sig_complex = sig[:-1].to(cdtype)
            ey0 = (hx[1:] - hx[:-1]) / dz_complex / sig_complex
            ey = torch.cat((ey0, ey0[-1:].clone()), dim=0)
        else:
            ey = torch.zeros_like(hx, dtype=cdtype, device=device)

        # 9. Extract results at original grid points (downsample)
        orig_nodes = sig0.numel() + 1
        idx = torch.arange(0, orig_nodes, device=device) * n_add
        idx = idx[idx < ey.numel()].to(torch.long)

        ey_out = ey[idx].contiguous()
        hx_out = hx[idx].contiguous()

        return ey_out, hx_out

    def mt1dtm(self, freq, dz0, sig0, n_add):
        """
        Calculate 1D electric and magnetic fields for TM mode (used as primary field for 2D calculation)
        
        Parameters:
        freq : frequency
        dz0 : original z-direction grid spacing
        sig0 : 1D model conductivity
        n_add : interpolation points per original grid
        
        Returns:
        ey : interpolated electric field Ey
        hx : interpolated magnetic field Hx
        """
        omega = 2.0 * np.pi * freq
        
        # Refine grid to improve 1D field calculation accuracy
        dz = np.array([dz0[i] / n_add * np.ones(n_add) for i in range(np.size(dz0))]).flatten()
        sig = np.array([sig0[i] * np.ones(n_add) for i in range(np.size(sig0))]).flatten()
        nz = len(sig)
        
        # Add an extra layer for bottom boundary condition
        sig = np.hstack((sig, sig[nz-1]))
        dz = np.hstack((dz, np.array(np.sqrt(2.0 / (sig[nz] * omega * self.miu)), dtype=float)))
        
        # Build 1D TM mode system matrix
        diagA = self.II * omega * self.miu * (dz[0:nz] + dz[1:nz+1]) - \
                2.0 / (dz[0:nz] * sig[0:nz]) - 2.0 / (dz[1:nz+1] * sig[1:nz+1])
        offdiagA = 2.0 / (dz[1:nz] * sig[1:nz])
        
        # Assemble sparse matrix
        mtxA = scipa.diags(diagA, format='csc') + \
               scipa.diags(offdiagA, -1, format='csc') + scipa.diags(offdiagA, 1, format='csc')
        
        # Right-hand side (apply boundary conditions)
        rhs = np.zeros((nz, 1))
        rhs[0] = -2.0 / (dz[0] * sig[0])
        
        # Solve linear system
        lup = scilg.splu(mtxA)
        hx0 = lup.solve(rhs)
        
        # Reconstruct magnetic field (include top boundary condition)
        hx = np.concatenate(([complex(1, 0)], hx0.reshape(-1)))
        
        # Calculate electric field Ey (from gradient of Hx)
        ey0 = (hx[1:] - hx[:-1]) / dz[:-1] / sig[:-1]
        ey = np.concatenate((ey0, ey0[-1:]))
        
        # Extract results at original grid points (downsample)
        idx = np.arange(np.size(sig0) + 1) * n_add
        
        return ey[idx], hx[idx]

    def _equation_solve(self, mtxA, rhs):
        '''
        Solve linear system Ax = b (PyTorch tensor version)
        
        Parameters:
        mtxA : coefficient matrix A (PyTorch sparse/dense tensor)
        rhs : right-hand side b (PyTorch tensor)
        
        Returns:
        x : solution of the system (PyTorch tensor)
        0 : indicates successful solution
        '''
        device = mtxA.device
        dtype = mtxA.dtype
        
        # Ensure rhs is column vector
        if rhs.dim() == 1:
            rhs = rhs.unsqueeze(1)
        
        # Select solver based on matrix type
        if mtxA.is_sparse:
            # Iterative GMRES for sparse matrix
            from torch.sparse.linalg import gmres
            def A_func(x):
                return torch.sparse.mm(mtxA, x)
            
            x, _ = gmres(A_func, rhs, tol=1e-6, maxiter=1000)
        else:
            # Direct solve for dense matrix
            x = torch.linalg.solve(mtxA, rhs)
        
        return x.flatten(), 0
    
    def equation_solve(self, mtxA, rhs):
        '''
        Solve linear system Ax = b
        
        Parameters:
        mtxA : coefficient matrix A
        rhs : right-hand side b
        
        Returns:
        x : solution of the system
        0 : indicates successful solution
        '''
        # Solve sparse linear system using LU decomposition
        lup = scilg.splu(mtxA)
        ex = lup.solve(rhs)
        
        return ex, 0
        
def save_model(model_name, zn, yn, freq, ry, sig_log, rhoxy, phsxy, zxy, rhoyx, phsyx, zyx):
    '''
    Save model and forward results to .mat file
    
    Parameters:
    model_name : save file name
    zn, yn : z and y direction node positions
    freq : frequency array
    ry : observation point positions
    sig_log : log conductivity model
    rhoxy, phsxy, zxy : TE mode results
    rhoyx, phsyx, zyx : TM mode results
    '''
    scio.savemat(model_name, {'zn': zn, 'yn': yn, 'freq': freq, 'obs': ry, 'sig': sig_log,
                            'rhoxy': rhoxy, 'phsxy': phsxy, 'zxy': zxy,
                            'rhoyx': rhoyx, 'phsyx': phsyx, 'zyx': zyx})

def func_remote(nza, zn, yn, freq, ry, sig, n_sample, mode="TETM", np_dtype=np.float16,use_parallel=False,num_cpus=1,num_gpus=0,ray=None):
    """
    Parallel computation of MT forward results for multiple models
    
    Parameters:
    nza : number of air layers
    zn, yn : grid node positions
    freq : frequency array
    ry : observation point positions
    sig : conductivity model set, shape (n_sample, nz, ny)
    n_sample : number of models
    mode : calculation mode
    np_dtype : data type
    
    Returns:
    Forward results for multiple models (apparent resistivity, phase and impedance)
    """
    n_freq = np.size(freq)
    n_ry = len(ry)
    
    # Initialize result arrays
    rhoxy = np.zeros((n_sample, n_freq, n_ry), dtype=np_dtype)
    phsxy = np.zeros((n_sample, n_freq, n_ry), dtype=np_dtype)
    rhoyx = np.zeros((n_sample, n_freq, n_ry), dtype=np_dtype)
    phsyx = np.zeros((n_sample, n_freq, n_ry), dtype=np_dtype)

    Ex = np.zeros((n_sample, n_freq, len(zn), len(yn)),dtype=np.complex64)
    Hx = np.zeros((n_sample, n_freq, len(zn)-nza, len(yn)), dtype=np.complex64)
    grad_te = np.zeros((n_sample, n_freq, len(zn)-1, len(yn)-1), dtype=np.complex64)
    grad_tm = np.zeros((n_sample, n_freq, len(zn)-nza-1, len(yn)-1),dtype=np.complex64)

    # Submit parallel tasks
    # Use parallel or serial based on use_parallel flag
    if use_parallel:
        MT2DFDRemote = ray.remote(MT2DFD)
        result = []
        for ii in range(n_sample):
            model = MT2DFDRemote.remote(nza, zn, yn, freq, ry, sig[ii, :, :])
            result.append(model.mt2d.remote(mode))
        temp0 = ray.get(result)
        
    else:
        result = []
        for ii in range(n_sample):
            model = MT2DFD(nza, zn, yn, freq, ry, sig[ii, :, :])
            result.append(model.mt2d(mode))
        temp0 = result
    for ii in range(len(temp0)):
        temp = temp0[ii]
        (rhoxy[ii, :, :], phsxy[ii, :, :],
            rhoyx[ii, :, :], phsyx[ii, :, :],
            Ex[ii, ...],Hx[ii, ...],grad_te[ii, ...], grad_tm[ii,] )= temp
    
    return rhoxy, phsxy, rhoyx, phsyx, Ex, Hx, grad_te, grad_tm
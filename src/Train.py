# ============================== 1. Import Modules (Grouped by Category) ==============================
# System Basic Libraries
import os
import sys
import yaml
from typing import Optional, List
from timeit import default_timer
import datetime
# Numerical Computing & Deep Learning Libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cmath as cm  # For complex number operations
# Custom Utility Libraries (MT Forward Modeling Related)
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.extend([current_path, project_root])

# Only import the ADFNO network
from utils.ADFNO import ADFNO
from utils.load_data import get_loader  # Data loading and preprocessing


# Global Configuration (Avoid duplicate definitions)
torch.set_default_dtype(torch.float32)  # Unify floating-point precision
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================== 2. Automatic Weighted Multi-Task Loss Class ==============================
class AutomaticWeightedLoss(nn.Module):
    """
    Automatically weighted multi-task loss (Data loss + Gradient loss)
    Formula: L_total = Σ [ 1/(2(w²+ε)) · L_i + log(1+w²) ]
    """
    def __init__(
        self,
        num_losses: int = 2,
        init_weights: Optional[List[float]] = None,
        device: Optional[torch.device] = None,
        eps: float = 0.01,  # ε should not be too small, prevent numerical explosion when w→0
    ):
        super().__init__()
        self.eps = eps
        self.device = device or DEVICE

        if init_weights is not None:
            assert len(init_weights) == num_losses, "Number of initial weights must match number of loss terms"
            weight_param = torch.tensor(init_weights, device=self.device)
        else:
            weight_param = torch.ones(num_losses, device=self.device)

        self.weight = nn.Parameter(weight_param)
        # Clamp the range of w for numerical stability
        self.weight_min = eps   # Lower bound: same as ε, prevent 1/(w²+ε) from being too large
        self.weight_max = 10.0  # Upper bound: prevent w from being too large (task is ignored)

    def forward(self, *losses: torch.Tensor) -> torch.Tensor:
        assert len(losses) == len(self.weight), "Number of loss terms does not match number of weights"

        # Accumulate losses, use 0.0 to preserve gradient chain (not torch.tensor(0.0))
        total_loss = 0.0
        for idx, single_loss in enumerate(losses):
            clamped_w = torch.clamp(self.weight[idx], min=self.weight_min, max=self.weight_max)
            total_loss = total_loss + 0.5 / (clamped_w ** 2 + self.eps) * single_loss + torch.log(1 + clamped_w ** 2)

        return total_loss


# ============================== 3. TE-mode Magnetotelluric Forward Model Class ==============================
class LpLoss1(object):
    """Lp Loss Class (Relative error calculation)"""
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]
        h = 1.0 / (x.size()[1] - 1.0)
        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
        if self.reduction:
            return torch.mean(all_norms) if self.size_average else torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            return torch.mean(diff_norms/y_norms) if self.size_average else torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class TEElectromagneticModel_ADFNO(nn.Module):
    """
    TE-mode Magnetotelluric Forward Model (ADFNO Specialized Version)
    Default loss configuration: data+grad
    """
    def __init__(
        self,
        train_file: str,
        test_file: str,
        n_train: int,
        n_test: int,
        batch_size: int,
        f_idx: List[int],
        loss_config: str = "data+grad",  # Default loss configuration
        device: torch.device = DEVICE,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        layer_num: int = 4,
        last_size: int = 128,
        act_fno: nn.Module = nn.GELU(),
        init_func: callable = None,
        tf_lr: float = 1e-3,
        weight_decay: float = 1e-4,
        step_size: int = 50,
        gamma: float = 0.5,
        freq_sample: int = 3,
        second_ep: int = 100,
        print_interval: int = 20,
    ):
        super().__init__()
        self.device = device
        self.ii = complex(0, 1)
        self.batch_size = batch_size
        self.n_out = 4
        self.miu = 4.0e-7 * torch.pi
        self.nza = 10
        self.print_interval = print_interval

        # Validate loss configuration
        assert loss_config in ["data", "data+grad", "grad"], \
            f"Invalid loss configuration: {loss_config}, must be 'data', 'grad' or 'data+grad'"
        self.loss_config = loss_config

        # Load data
        self._load_data(train_file, test_file, n_train, n_test, batch_size, f_idx)
        # Build ADFNO network
        self.model = ADFNO().to(self.device)
        # Initialize optimizer and loss
        self._init_optimizer(tf_lr, weight_decay, step_size, gamma, freq_sample, second_ep)

    def _load_data(self, train_file, test_file, n_train, n_test, batch_size, f_idx):
        """Data Loading"""
        (self.freq, self.nza, self.zn, self.yn, self.dz, self.dy, self.ry,
         self.train_loader, self.test_loader, self.x_normalizer, self.y_normalizer, loc_train) = get_loader(
            train_file=train_file,
            test_file=test_file,
            n_train=n_train,
            n_test=n_test,
            batch_size=batch_size,
            f_idx=f_idx
        )
        self.f_idx = f_idx
        self.n_freq_train = f_idx[1]
        self.n_freq_test = f_idx[3]
        self.n_train = n_train
        self.n_test = n_test
        self.dz = self.dz.to(self.device)
        self.dy = self.dy.to(self.device)
        self.loc_train = loc_train.to(self.device)

    def _init_optimizer(self, tf_lr, weight_decay, step_size, gamma, freq_sample, second_ep):
        """Optimizer Initialization"""
        self.lp_loss = LpLoss1(size_average=False)

        # Only use automatic weighting for combined loss mode
        if self.loss_config == "data+grad":
            self.auto_weight_loss = AutomaticWeightedLoss(
                num_losses=2, init_weights=[1.0, 1.0], device=self.device
            )

        # Optimizer params: model weights use weight_decay, loss weights do not (avoid weights tending to 0)
        params = [{"params": self.model.parameters(), "weight_decay": weight_decay}]
        if self.loss_config == "data+grad":
            # Separate setting for loss weights, no weight_decay applied
            params.append({"params": self.auto_weight_loss.parameters(), "weight_decay": 0.0})
        self.optimizer = optim.AdamW(params=params, lr=tf_lr)
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-6)
        self.freq_sample = freq_sample
        self.second_ep = second_ep

    def _forward_ADFNO(self, x, grid, freq):
        """Forward pass of the ADFNO network"""
        pred_grad, pred_y = self.model(x, grid, freq)
        return pred_grad, pred_y

    def _calculate_loss_ADFNO(self, pred_y, y, pred_grad, grad_tetm, x, freq, u_bc):
        """Loss Calculation for ADFNO"""

        data_loss = self.lp_loss(pred_y, y)
        grad_loss = self.lp_loss(pred_grad, grad_tetm)  # Use relative error uniformly for consistent magnitude
        return self.auto_weight_loss(data_loss, grad_loss)

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_train_loss = 0.0

        for batch_idx, (x, y, freq, u_bc, grad_tetm, grid) in enumerate(self.train_loader):
            # 1. Move data to device
            batch_data = [x, y, freq, u_bc, grad_tetm, grid]
            x, y, freq, u_bc, grad_tetm, grid = [item.to(self.device) for item in batch_data]

            # 2. Clear gradients
            self.optimizer.zero_grad()
            x.requires_grad_(True)
            # 3. ADFNO forward pass
            pred_grad, pred_y = self._forward_ADFNO(x, grid, freq)

            # 4. Calculate loss
            total_loss = self._calculate_loss_ADFNO(pred_y, y, pred_grad, grad_tetm, x, freq, u_bc)

            # 5. Backpropagation and parameter update
            total_loss.backward()
            self.optimizer.step()
            total_train_loss += total_loss.item()
            if self.print_interval > 0 and batch_idx % self.print_interval == 0:
                print(f"batch_idx [{batch_idx}/{len(self.train_loader)}], Loss: {total_loss.item():.4f}")
        self.lr_scheduler.step()
        return total_train_loss / len(self.train_loader)

    def _evaluate(self) -> float:
        self.model.eval()
        total_test_loss = 0.0

        with torch.no_grad():
            for batch_idx, (x, y, freq, u_bc, grad_tetm, grid) in enumerate(self.test_loader):
                # Move data to device
                batch_data = [x, y, freq, u_bc, grad_tetm, grid]
                x, y, freq, u_bc, grad_tetm, grid = [item.to(self.device) for item in batch_data]
                # ADFNO forward pass
                pred_grad, pred_y = self._forward_ADFNO(x, grid, freq)

                # Evaluate loss (use data-only loss logic for testing)
                original_loss_config = self.loss_config
                test_loss = self._calculate_loss_ADFNO(pred_y, y, pred_grad, grad_tetm, x, freq, u_bc)
                self.loss_config = original_loss_config

                total_test_loss += test_loss.item()

        return total_test_loss / len(self.test_loader)

    def train_model(self, tf_epochs, thre_epoch, patience, save_step, save_mode, model_path, model_path_temp, log_file):
        """Main Training Loop"""
        best_test_loss = np.inf
        early_stop_counter = 0
        temp_model_path = None

        for epoch in range(tf_epochs):
            start_time = default_timer()
            avg_train_loss = self._train_one_epoch(epoch)
            avg_test_loss = self._evaluate()

            # Save temporary model
            if epoch % save_step == 0:
                if temp_model_path and os.path.exists(temp_model_path):
                    os.remove(temp_model_path)
                curtime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_model_path = f"{model_path_temp}_{curtime}_epoch_{epoch}.pkl"
                torch.save(self.model.state_dict(), temp_model_path)

            # Early stopping check
            if epoch > thre_epoch:
                if avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    early_stop_counter = 0
                    self._save_best_model(model_path, save_mode)
                else:
                    early_stop_counter += 1
                if early_stop_counter > patience:
                    print(f"[Early Stop Triggered] Epoch {epoch}: Test loss increased for {patience} consecutive epochs")
                    print(f"# Early stop at epoch {epoch}", file=log_file)
                    break

            # Logging
            end_time = default_timer()
            time_cost = end_time - start_time

            # Get automatic weighted weights (only for data+grad mode)
            weight_info = ""
            if self.loss_config == "data+grad":
                w_data = self.auto_weight_loss.weight[0].item()
                w_grad = self.auto_weight_loss.weight[1].item()
                weight_info = f" | w_data={w_data:.4f}, w_grad={w_grad:.4f}"

            log_msg = (
                f"[Epoch {epoch}] "
                f"Time: {time_cost:.2f}s | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Test Loss: {avg_test_loss:.6f} | "
                f"Best Test Loss: {best_test_loss:.6f}{weight_info} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            print(log_msg)
            log_file_msg = f"{epoch}, {time_cost:.2f}, {avg_train_loss:.6f}, {avg_test_loss:.6f}"
            if self.loss_config == "data+grad":
                log_file_msg += f", {w_data:.4f}, {w_grad:.4f}"
            print(log_file_msg, file=log_file)
            log_file.flush()

    def _save_best_model(self, model_path: str, save_mode: str):
        """Save Best Model"""
        if save_mode == "state_dict":
            torch.save(self.model.state_dict(), f"{model_path}.pkl")
        elif save_mode == "full":
            torch.save(self.model, f"{model_path}.pt")
        else:
            raise ValueError(f"Unsupported save mode: {save_mode}")


# ============================== 4. Main Function (Entry Logic) ==============================
def load_config(config_item: str) -> dict:
    """Load Configuration File"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.full_load(f)

    if config_item not in config:
        raise KeyError(f"No '{config_item}' section in config file, please check configuration")

    return config[config_item]


def create_output_dirs(model_path: str, model_path_temp: str, log_path: str):
    """Create Output Directories"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path_temp), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)


def train_ADFNO(config_item: str, loss_config: str = "data+grad"):
    """Train ADFNO Network"""
    total_start_time = default_timer()
    config = load_config(config_item)

    # Parse configuration parameters
    cuda_id = f"cuda:{config['cuda_id']}"
    device = torch.device(cuda_id if torch.cuda.is_available() else "cpu")
    # Use absolute path of project root
    train_file = os.path.join(project_root, config["TRAIN_PATH"])
    test_file = os.path.join(project_root, config["VAL_PATH"])
    f_idx = config["f_idx"]
    n_train = config["n_train"]
    n_test = config["n_test"]
    batch_size = config["batch_size"]
    modes = config["modes"]
    width = config["width"]
    layer_num = config["layer_num"]
    last_size = config["last_size"]
    act_fno = config["act_fno"]
    init_func = config["init_func"]
    tf_epochs = config["tf_epochs"]
    tf_lr = config["tf_lr"]
    weight_decay = config["weight_decay"]
    step_size = config.get("step_size", tf_epochs//10)
    gamma = config["gamma"]
    patience = config["patience"]
    thre_epoch = config["thre_epoch"]
    freq_sample = config["freq_sample"]
    save_mode = config["save_mode"]
    save_step = config["save_step"]
    print_interval = config["print_interval"]
    
    # Append loss config to paths (ensure uniqueness)
    model_path = os.path.join(project_root, f"model/{config_item}_ADFNO_{loss_config}")
    model_path_temp = os.path.join(project_root, f"temp/{config_item}_ADFNO_{loss_config}")
    log_path = os.path.join(project_root, f"Log/{config_item}_ADFNO_{loss_config}.log")

    # Create directories
    create_output_dirs(model_path, model_path_temp, log_path)

    # Initialize log and start training
    with open(log_path, "w+", encoding="utf-8") as log_file:
        print("=" * 70)
        print(f"Start Training: Config = {config_item} | Network = ADFNO | Loss = {loss_config}")
        print(f"Device = {device}")
        print("=" * 70)
        print(f"# Training Start: {config_item}, Network=ADFNO, Loss={loss_config}", file=log_file)
        print(f"# Device: {device}", file=log_file)

        # Initialize model
        te_model = TEElectromagneticModel_ADFNO(
            train_file=train_file,
            test_file=test_file,
            n_train=n_train,
            n_test=n_test,
            batch_size=batch_size,
            f_idx=f_idx,
            loss_config=loss_config,
            device=device,
            modes1=modes,
            modes2=modes,
            width=width,
            layer_num=layer_num,
            last_size=last_size,
            act_fno=act_fno,
            init_func=init_func,
            tf_lr=tf_lr,
            weight_decay=weight_decay,
            step_size=step_size,
            gamma=gamma,
            freq_sample=freq_sample,
            print_interval=print_interval
        )

        # Start training
        te_model.train_model(
            tf_epochs=tf_epochs,
            thre_epoch=thre_epoch,
            patience=patience,
            save_step=save_step,
            save_mode=save_mode,
            model_path=model_path,
            model_path_temp=model_path_temp,
            log_file=log_file
        )

        # Log total time cost
        total_time = default_timer() - total_start_time
        print(f"\nTraining Finished: Total Time = {total_time:.2f}s")
        print(f"# Training Finished: Total Time = {total_time:.2f}s", file=log_file)


# ============================== 5. Program Entry ==============================
if __name__ == "__main__":
    # Get base config from command line
    try:
        config_item = sys.argv[1]
    except IndexError:
        config_item = "grid_30000"  # Default config
        print(f"No config specified, using default: {config_item}")

    print(f"\n{'='*50} Start Training ADFNO Network {'='*50}")
    train_ADFNO(config_item)
    print(f"{'='*50} ADFNO Network Training Completed {'='*50}\n")
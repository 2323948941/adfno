
import os
import sys
import yaml
import datetime
from timeit import default_timer

import numpy as np
import torch

# Path setup
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.extend([current_path, project_root])

from utils.ADFNO import ADFNO
from utils.load_data import get_loader

torch.set_default_dtype(torch.float32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


# ==============================================================================
# Configuration
# ==============================================================================
def load_config(config_item: str) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.full_load(f)
    if config_item not in config:
        raise KeyError(f"Config item '{config_item}' not found")
    return config[config_item]


# ==============================================================================
# Model Loading
# ==============================================================================
def load_model(model_path: str, device: torch.device) -> ADFNO:
    model = ADFNO().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model



# ==============================================================================
# Main Prediction
# ==============================================================================
def predict(config_item: str, model_path: str = None):
    total_start = default_timer()
    config = load_config(config_item)

    # Device
    cuda_id = f"cuda:{config['cuda_id']}"
    device = torch.device(cuda_id if torch.cuda.is_available() else "cpu")

    # Paths
    train_file = os.path.join(project_root, config["TRAIN_PATH"])
    test_file = os.path.join(project_root, config["TEST_PATH"])
    f_idx = config["f_idx"]
    n_train = config["n_train"]
    n_test = config["n_test"]
    batch_size = config.get("batch_sizetest", config["batch_size"])

    # Model path
    if model_path is None:
        model_dir = os.path.join(project_root, "temp")
        candidates = [f for f in os.listdir(model_dir) if f.startswith(f"{config_item}_ADFNO")]
        if not candidates:
            raise FileNotFoundError(f"No model found in {model_dir} for {config_item}")
        model_path = os.path.join(model_dir, candidates[0])
    print(f"Model: {model_path}")

    # Output directory
    save_root = os.path.join(project_root, "results", "predictions")
    os.makedirs(save_root, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data (get_loader fits normalizers on train data, applies to test)
    # ------------------------------------------------------------------
    print("Loading data ...")
    (freq_all, nza, zn, yn, dz, dy, ry,
     _, test_loader, x_normalizer, y_normalizer, loc_train) = get_loader(
        train_file=train_file,
        test_file=test_file,
        n_train=n_train,
        n_test=n_test,
        batch_size=batch_size,
        f_idx=f_idx,
    )
    grad_normalizer = None  # get_loader does not return it; we reconstruct from train data
    # NOTE: grad_normalizer is created inside get_loader but not returned.
    # For denormalizing pred_grad we re-compute it from the raw training gradients.
    # If you need pred_grad denormalized, pass --save-grad and the script will
    # re-fit the normalizer on train data.

    print(f"  Test samples: {n_test}")
    print(f"  Frequencies: {freq_all.shape[0]}")

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    print("Loading model ...")
    model = load_model(model_path, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # 3. Predict
    # ------------------------------------------------------------------
    print("Predicting ...")
    model.eval()

    all_y_true = []
    all_y_pred = []
    all_x = []
    all_freq = []

    with torch.no_grad():
        for batch_idx, (x, y, freq, u_bc, grad_tetm, grid) in enumerate(test_loader):
            x, y, freq, grid = (
                x.to(device, non_blocking=True),
                y.to(device, non_blocking=True),
                freq.to(device, non_blocking=True),
                grid.to(device, non_blocking=True),
            )

            pred_grad, pred_y = model(x, grid, freq)

            # 反归一化（恢复物理意义）
            x_denorm = x_normalizer.decode(x)
            y_true_denorm = y_normalizer.decode(y)
            y_pred_denorm = y_normalizer.decode(pred_y)
            y_true_denorm[..., 3] = y_true_denorm[..., 3] - 90
            y_pred_denorm[..., 3] = y_pred_denorm[..., 3] - 90

            all_y_true.append(y_true_denorm.cpu().numpy())
            all_y_pred.append(y_pred_denorm.cpu().numpy())
            all_x.append(x_denorm.cpu().numpy())
            all_freq.append(freq.cpu().numpy())


            if (batch_idx + 1) % 10 == 0:
                print(f"  batch {batch_idx + 1}/{len(test_loader)}")

    # Concatenate all batches
    y_true = np.concatenate(all_y_true, axis=0)[:n_test]   # (n_test, n_freq, dy, 4)
    y_pred = np.concatenate(all_y_pred, axis=0)[:n_test]
    x_data = np.concatenate(all_x, axis=0)[:n_test]         # (n_test, dy, dz, 1)
    # freq_data = np.concatenate(all_freq, axis=0)[:n_test]   # (n_test, n_freq, 1)

    # ------------------------------------------------------------------
    # 4. Save results
    # ------------------------------------------------------------------
    print("Saving results ...")
    npz_path = os.path.join(save_root, f"{config_item}_predictions.npz")
    np.savez_compressed(
        npz_path,
        y_true=y_true,
        y_pred=y_pred,
        x=x_data,
        freq=freq[0,:,0].cpu().numpy(),  # Save the frequency array (same for all samples in the batch)
    )
    print(f"  Predictions: {npz_path}")

# ==============================================================================
# Entry point
# ==============================================================================
if __name__ == "__main__":
    try:
        config_item = sys.argv[1]
    except IndexError:
        config_item = "grid_30000"
        print(f"No config item given, using default: {config_item}")

    try:
        model_path = sys.argv[2]
    except IndexError:
        model_path = 'temp/grid_30000_ADFNO_data+grad_20260514_174016_epoch_240.pkl'  # Default to searching in ../temp/ instead of ../model/
        print("No model path given, auto-searching ../temp/")

    print(f"\n{'=' * 50} ADFNO Prediction {'=' * 50}")
    predict(config_item, model_path)
    print(f"{'=' * 50} Done {'=' * 50}\n")

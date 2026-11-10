import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics import F1Score, JaccardIndex, Precision, Recall
from tqdm import tqdm

from dataset.avalanches import AvalancheDataset
from models.build import build_model, model_choices
from utils.morph import morph_close
from utils.runtime import get_device, log_param_count, setup_logging
from utils.visualization import save_test_quicklooks

# --------------------------------------------------------------------------- #
# CLI + paths
# --------------------------------------------------------------------------- #
setup_logging()

parser = argparse.ArgumentParser(description="Avalanche change-detection test")
parser.add_argument("--patch-size", default=128, type=int)
parser.add_argument("--dataset-root", type=str, default="/home/jovyan/nfs/mgatti/datasets/Avalanches/patches/",
                    help="Root directory of the dataset (without patch size).")
parser.add_argument("--model", type=str, choices=model_choices(),
                    help="Choose an architecture")
parser.add_argument("--model-size", type=str, choices=["tiny", "small", "base"], default="tiny",
                    help="Size to use: tiny, small, base")
parser.add_argument("--use-aux", action="store_true",
                    help="Enable Auxiliary data usage.")
parser.add_argument("--exp-root", type=str, default="exp",
                    help="Root directory containing experiment folders with best_model.pth.")
parser.add_argument("--model-ckpt", type=str, default=None,
                    help="Explicit checkpoint path (overrides --exp-root lookup).")
parser.add_argument("--plot-every", type=int, default=1,
                    help="Plot every Nth sample that has any change in GT")
parser.add_argument("--no-morph", dest="use_morph", action="store_false",
                    help="Disable morphological closing (dilate -> erode) ...")
parser.add_argument("--kernel-size", type=int, default=3,
                    help="Kernel size for morphological operations (must be odd)")
parser.add_argument("--iterations", type=int, default=1,
                    help="Number of dilation / erosion iterations")

args = parser.parse_args()

MODEL_SIZE = args.model_size
PATCH_SIZE = args.patch_size
USE_AUX = args.use_aux
PLOT_EVERY = args.plot_every

USE_MORPH = args.use_morph
KERNEL_SIZE = args.kernel_size
NUM_ITERS = args.iterations

DATASET_DIR = Path(args.dataset_root) / f"{PATCH_SIZE}"
if args.model_ckpt is not None:
    MODEL_CKPT = Path(args.model_ckpt)
else:
    MODEL_CKPT = Path(args.exp_root) / f"{args.model}_{PATCH_SIZE}{'_aux' if USE_AUX else ''}" / "best_model.pth"

OUTPUT_DIR = Path(f"output/test/{PATCH_SIZE}")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# --------------------------------------------------------------------------- #
# data + model
# --------------------------------------------------------------------------- #
device = get_device()
test_ds = AvalancheDataset(DATASET_DIR, ["Tromso_20241220"], apply_transform=False)
test_dl = DataLoader(test_ds, batch_size=32, num_workers=4)

TOTAL_TEST_PATCHES = len(test_ds)
logging.info(f"Total test patches (dataset): {TOTAL_TEST_PATCHES}")

model = build_model(
    args.model,
    patch_size=PATCH_SIZE,
    use_aux=USE_AUX,
    model_size=MODEL_SIZE,
    device=device,
)
log_param_count(model)

ckpt = torch.load(MODEL_CKPT, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
best_thr = ckpt.get("best_threshold", 0.5)
logging.info(f"Decision threshold (best_threshold from ckpt): {best_thr:.4f}")

metrics = {
    "recall": Recall(task="binary", num_classes=1, threshold=best_thr).to(device),
    "precision": Precision(task="binary", num_classes=1, threshold=best_thr).to(device),
    "f1": F1Score(task="binary",  num_classes=1, threshold=best_thr).to(device),
    "iou": JaccardIndex(task="binary", num_classes=1, threshold=best_thr).to(device),
}

# --------------------------------------------------------------------------- #
# plotting stats (used for quick-look de-normalization)
# --------------------------------------------------------------------------- #
STATS_PATH = DATASET_DIR / "stats.json"
stats = json.loads(Path(STATS_PATH).read_text())
IMG_MEAN_DB, IMG_STD_DB = map(torch.tensor, (stats["img_mean"], stats["img_std"]))
SENTINEL_Z_IMG = torch.tensor(stats["sentinel_z_img"])  # shape: (2,)


if __name__ == "__main__":
    global_idx = 0
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Testing", ncols=100):
            pre, post, aux = batch["pre"].to(device), batch["post"].to(device), batch["aux"].to(device)
            gt = batch["mask"].to(device)

            # ---- forward (with/without AUX) ----
            if not USE_AUX:
                logits = model(pre, post)
            else:
                logits = model(pre, post, aux)

            # ---- probabilities + threshold ----
            prb = torch.sigmoid(logits)           # (B,1,H,W)
            bin_mask = (prb > best_thr).float()

            # -------- optional morphology -------- #
            if USE_MORPH:
                bin_mask = morph_close(bin_mask, kernel_size=KERNEL_SIZE, iterations=NUM_ITERS)

            # -------- metric update -------- #
            for m in metrics.values():
                # torchmetrics will internally threshold again, but that's fine
                m.update(bin_mask, gt)

            for b in range(pre.size(0)):
                if PLOT_EVERY > 0 and (global_idx % PLOT_EVERY == 0) and torch.any(gt[b]):
                    save_test_quicklooks(
                        global_idx, pre[b], post[b], aux[b], gt[b], prb[b], bin_mask[b],
                        out_dir=OUTPUT_DIR,
                        img_mean_db=IMG_MEAN_DB,
                        img_std_db=IMG_STD_DB,
                        sentinel_z_img=SENTINEL_Z_IMG,
                    )
                global_idx += 1

    for n, m in metrics.items():
        logging.info(f"{n.capitalize()}: {m.compute().cpu().numpy():.4f}")

    logging.info(f"Total number of patches: {global_idx}")
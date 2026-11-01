# VGG6 on CIFAR-10: activation, optimizer, batch size, epochs and learning rate study

All experiments train a small VGG6 network on CIFAR-10 and are tracked with Weights & Biases (W&B) sweeps.

**Best configuration:** ReLU, Nesterov-SGD, lr 0.05, batch size 128, 20 epochs, momentum 0.9, weight decay 5e-4, seed 42
-> **best validation accuracy 83.40% (epoch 19), test top-1 accuracy 82.56%.**

W&B project: `swaminathan_vgg6_CIFAR-10` 
https://forge.coreweave.com/wandb/swaminathan47-indian-institute-of-madras/swaminathan_vgg6_CIFAR-10/workspace?nw=nwuserswaminathan47

GitHUB Link: 
https://github.com/GSN001/Swaminathan_VGG6_CIFAR-10.git

---

## 1. Repository structure

| File | Purpose |
|---|---|
| `data.py` | CIFAR-10 loading, fixed train/validation/test split, normalisation and augmentation |
| `model.py` | VGG6 with a configurable activation function (batch norm on by default) |
| `train.py` | Training script: argument parsing, optimizers, cosine LR schedule, W&B logging, checkpointing |
| `evaluate.py` | Evaluation function used during training, and a command-line tool that evaluates a saved checkpoint |
| `sweep_activation.yaml` | W&B sweep for : ReLU, Sigmoid, Tanh, SiLU, GELU |
| `sweep_optimizer.yaml` | W&B sweep for : SGD, Nesterov-SGD, Adam, Adagrad, RMSprop, Nadam |
| `sweep_lr.yaml`, `sweep_batch.yaml`, `sweep_epochs.yaml` | W&B sweeps for : lr, batch_size and epochs |
| `checkpoints/` | Saved best-validation checkpoints (the best model is included in the repo) |
| `sweep.py` | Run all Sweeps and create wandb plots in the configured project |

## 2. Environment

Experiments were run on a CUDA GPU (Google Colab) with Python 3.13.

| Package 
|---|---|
| Python | [ 3.13.+] |
| torch | 
| torchvision | 
| numpy | 
| wandb | 


Install:

```bash
pip install torch torchvision numpy wandb
```

Log in to W&B once (skip this if you use `--wandb_mode disabled`):

```bash
wandb login
```

## 3. Dataset preparation

`data.py` loads CIFAR-10 with `download=False`, so the extracted dataset must already exist under `--data_dir` (default `./data`). The folder must contain `cifar-10-batches-py/` directly:

```bash
mkdir -p data
cd data
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz
cd ..
# result: data/cifar-10-batches-py/data_batch_1 ... data_batch_5, test_batch, batches.meta
```

**Split:** the 50,000 training images are split once, using a fixed seed (`split_seed=42`), into **45,000 training** and **5,000 validation** images. The 10,000 official test images are used exactly once per run, on the best-on-validation checkpoint. The validation set uses the non-augmented transform.

**Transforms**

| Split | Transforms |
|---|---|
| Train | `RandomCrop(32, padding=4)`, `RandomHorizontalFlip()`, `ToTensor()`, `Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))`, `Cutout(n_holes=1, length=16)` |
| Validation / Test | `ToTensor()`, `Normalize` (same mean and std) |

## 4. Model

VGG6: two conv blocks, each `[Conv3x3 -> BatchNorm -> activation] x 2 -> MaxPool2x2`, with 64 then 128 channels, followed by global average pooling and one linear layer (128 -> 10). Weights use the template initialisation. The activation is selectable: `relu`, `sigmoid`, `tanh`, `silu`, `gelu`.

## 5. Training setup

- Loss: cross-entropy
- Learning rate schedule: cosine annealing over the full number of epochs (`CosineAnnealingLR`, `T_max = epochs`), stepped once per epoch
- Momentum 0.9 (SGD, Nesterov), weight decay 5e-4 (all optimizers)
- Default configuration for single-factor sweeps: ReLU, Nesterov-SGD, lr 0.05, batch size 128, 10 epochs
- With `--lr -1` (the default), each optimizer uses its own default learning rate:

| Optimizer | `--optimizer` | Default lr |
|---|---|---|
| SGD (with momentum) | `sgd` | 0.05 |
| Nesterov-SGD | `nesterov` | 0.05 |
| Adam | `adam` | 0.001 |
| Adagrad | `adagrad` | 0.01 |
| RMSprop | `rmsprop` | 0.001 |
| Nadam | `nadam` | 0.001 |

## 6. Seed configuration

The default seed is **42** (`--seed 42`). `set_seed()` in `train.py` seeds Python `random`, NumPy, PyTorch CPU and all CUDA devices, and sets `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`. The train/validation split uses its own fixed seed (`split_seed=42` in `data.py`), so every run sees the same split. With the same hardware, software versions and `--num_workers`, repeated runs of the same configuration gave identical per-epoch numbers. Results may differ slightly on different GPUs or library versions.

## 7. How to reproduce

### 7.1 Best model (Question 4)

```bash
python train.py --activation relu --optimizer nesterov --lr 0.05 --batch_size 128 --epochs 20 --seed 42
```

Expected: best validation accuracy 83.40% at epoch 19 and test accuracy 82.56%. The checkpoint is saved to `checkpoints/relu_nesterov_lr0.05_bs128_ep20.pt`.

Run without W&B (no login needed):

```bash
python train.py --activation relu --optimizer nesterov --lr 0.05 --batch_size 128 --epochs 20 --wandb_mode disabled
```

### 7.2 Evaluate a saved checkpoint

```bash
python evaluate.py --checkpoint checkpoints/relu_nesterov_lr0.05_bs128_ep20.pt --data_dir ./data
```

This prints the configuration the checkpoint was trained with, the epoch it was saved at, and validation and test top-1 accuracy.

### 7.3 All sweeps (Questions 2 and 3)

Create each sweep, then start an agent with the sweep ID that W&B prints:

```bash
python sweep.py
```

This runs for `sweep_optimizer.yaml`, `sweep_lr.yaml`, `sweep_batch.yaml` and `sweep_epochs.yaml`. All sweeps use `method: grid` and maximize `val_acc`. Each sweep varies one factor and holds the others at the default configuration (the default-configuration run appears in the activation, optimizer, learning-rate, batch-size and epoch sweeps).

| Sweep file | Varies | Values | Runs |
|---|---|---|---|
| `sweep_activation.yaml` | activation | relu, sigmoid, tanh, silu, gelu | 5 |
| `sweep_optimizer.yaml` | optimizer (default lr each) | sgd, nesterov, adam, adagrad, rmsprop, nadam | 6 |
| `sweep_lr.yaml` | learning rate | 0.01, 0.1 | 2 |
| `sweep_batch.yaml` | batch size | 64, 256 | 2 |
| `sweep_epochs.yaml` | epochs | 5, 20 | 2 |

The W&B parallel-coordinates, validation-accuracy-vs-step and loss/accuracy plots are generated from these sweeps in the W&B dashboard.


## 11. Author

G Swaminathan, CS25M546, IIT Madras
GitHub: (https://github.com/GSN001/Swaminathan_VGG6_CIFAR-10.git)
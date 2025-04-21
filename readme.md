

# 🧬 Cell Counting with U-Net and EfficientNet

This repository provides a modular pipeline for training deep learning models to perform **cell counting** from microscopy images using **density maps**. The setup includes:

- ✅ Preprocessing with dot-to-density conversion  
- ✅ Training with U-Net or EfficientNet backbones  
- ✅ Evaluation & visualization using Prefect & Weights & Biases (W&B)

---

## 📂 Structure

- `preprocess_cells.py` — Handles dataset preparation, including image loading, dot mask conversion to density maps, and custom augmentations like quarter shuffling.
- `train_cells.py` — Defines the models (manual U-Net and EfficientNet-based U-Net), training loop, and custom loss function.
- `cells_wandb.py` — Orchestrates the full workflow (preprocess, train, evaluate) using Prefect and logs metrics/plots to W&B.

---

## 🔁 Task Flow

Inside `cells_wandb.py`, the following pipeline is orchestrated using Prefect:

```text
preprocess_task 
      ↓ 
  train_task 
      ↓ 
error_analysis
      ↓
    Flow
```

Each component is a separate `@task`, and they are chained together inside the `main_flow`.

---

## 🚀 How to Run

First, install dependencies:

```bash
pip install -r requirements.txt
```

Then, run the full pipeline using:

```bash
python cells_wandb.py --smp --preprocess --train --error_analysis --num_epochs 5
```

You can also optionally control training behavior with:

- `--lr` (default: `1e-4`)
- `--weight_decay` (default: `1e-5`)
- `--summary` (prints model architecture)

---

## 🧪 Notes

- **Preprocessing** currently uses `gt=False` (no heavy augmentation) and `qt=True` (quarter shuffling enabled).
- The validation and training sets are split **according to the original dataset structure**.
- `preprocess_cells.py` handles everything — just provide the paths, and it prepares the loaders.
- Weights & Biases is integrated for visualization — ensure you're logged in via `wandb login`.

---

## 📈 Example Outputs

- 🎯 Training & validation loss per epoch  
- 📊 MAE (mean absolute error) tracking  
- 🖼️ Sample visualizations of inputs, dot masks, and predicted density maps (logged to W&B)


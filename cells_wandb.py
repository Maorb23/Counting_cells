#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from pathlib import Path
import wandb
import prefect
from prefect import task, flow
from preprocess_cells import CellCountingDataset
from train_cells import UNetCellCounter, UNetCellCounterEffNet, train_unet, mse_with_count_regularization
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------------
# Tasks
# -------------------------------------------------------


@task
def preprocess_task(train_images: str, train_labels: str, val_images: str, val_labels: str, batch_size: int = 8):
    

    logger = prefect.get_run_logger()
    train_image_paths = [os.path.join(train_images, f) for f in os.listdir(train_images) if f.endswith('.png')]
    train_label_paths = [os.path.join(train_labels, f) for f in os.listdir(train_labels) if f.endswith('.png')]
    val_image_paths = [os.path.join(val_images, f) for f in os.listdir(val_images) if f.endswith('.png')]
    val_label_paths = [os.path.join(val_labels, f) for f in os.listdir(val_labels) if f.endswith('.png')]


    train_dataset = CellCountingDataset(train_image_paths, train_label_paths, mode='train')
    val_dataset = CellCountingDataset(val_image_paths, val_label_paths, mode='val')
    train_loader = train_dataset.get_loader(batch_size= batch_size, shuffle=True)
    val_loader = val_dataset.get_loader(batch_size= batch_size, shuffle=False)
    fig = train_dataset.visualize_sample(0)
    wandb.log({"Sample Image": wandb.Image(fig)})
    return train_loader, val_loader




@task
def train_task(train_loader,val_loader,
               manual = False, smp = False, num_epochs=30, lr = 1e-4, weight_decay = 1e-5, summary = True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if manual:
        model = UNetCellCounter().to(device)
    elif smp:
        model = UNetCellCounterEffNet(backbone="efficientnet-b4", pretrained=True).to(device)
    else: 
        raise ValueError("Please specify a model type")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
    if summary:    
        from torchsummary import summary
        summary(model, input_size = (3,256,256))



    train_losses, val_losses, mae_list = train_unet(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=mse_with_count_regularization,
        device=device,
        num_epochs=num_epochs
    )
    return train_losses, val_losses, mae_list

import wandb
from bokeh.plotting import figure, output_file, save, show
from bokeh.layouts import column

@task
def error_analyze(train_losses, val_losses, mae_list, save_path="training_curves.html"):
    output_file(save_path)
    
    x = list(range(1, len(train_losses) + 1))

    # Loss plot
    p1 = figure(title="Training vs Validation Loss", x_axis_label='Epoch', y_axis_label='Loss',
                width=600, height=300)
    p1.line(x, train_losses, legend_label="Train Loss", line_color="blue", line_width=2)
    p1.line(x, val_losses, legend_label="Val Loss", line_color="green", line_width=2)
    p1.legend.location = "top_right"
    p1.legend.click_policy = "hide"

    # MAE plot
    p2 = figure(title="Validation MAE per Epoch", x_axis_label='Epoch', y_axis_label='MAE',
                width=600, height=300)
    p2.line(x, mae_list, legend_label="MAE", line_color="red", line_width=2)
    p2.legend.location = "top_right"

    # Combine and save
    layout = column(p1, p2)
    save(layout)
    show(layout)

    # 🧠 Log to Weights & Biases
    with open(save_path, "r") as f:
        wandb.log({"Training Curves (Bokeh HTML)": wandb.Html(f.read())})


# -------------------------------------------------------
# Flow
# -------------------------------------------------------

@flow(name="Cell training and error analysis")
def main_flow(train_images: str, train_labels: str, val_images: str, val_labels: str, batch_size: int = 8, 
              manual = False, smp = False, num_epochs=30, lr = 1e-4, weight_decay = 1e-5, summary = True,
              preprocess = False, train = False, error_analysis = False, eda_plots = False):
    """Flow to train the model and plot UMAP."""
    # Initialize Weights & Biases
    wandb.init(project="cell-counting", entity="maorblumberg-tel-aviv-university", settings=wandb.Settings(start_method="thread"))
    wandb.config.update({"batch_size": batch_size, "num_epochs": num_epochs, "lr": lr, "weight_decay": weight_decay})

    if preprocess:
        train_loader, val_loader = preprocess_task(train_images, train_labels, val_images, val_labels, batch_size)
    
    if train:
        train_losses, val_losses, mae_list = train_task(train_loader, val_loader, manual, smp, num_epochs, lr, weight_decay, summary)
    
    if error_analysis:
        error_analyze(train_losses, val_losses, mae_list)

    
    wandb.finish()

# -------------------------------------------------------
# CLI
# -------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scGPT training and UMAP visualization as a Prefect Flow.")
    parser.add_argument('--train_images', type=str, default='data/train_images', help='Path to the train imagse folder')
    parser.add_argument('--train_labels', type=str, default='data/train_labels', help='Path to the train labels folder')
    parser.add_argument('--val_images', type=str, default='data/val_images', help='Path to the val images folder')
    parser.add_argument('--val_labels', type=str, default='data/val_labels', help='Path to the val labels folder')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for DataLoader')
    parser.add_argument('--manual', action='store_true', help='Use manual model')
    parser.add_argument('--smp', action='store_true', help='Use SMP model')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--summary', action='store_true', help='Show model summary')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the data')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--error_analysis', action='store_true', help='Perform error analysis')
    args = parser.parse_args()

    # Call the flow
    main_flow(
        train_images=args.train_images,
        train_labels=args.train_labels,
        val_images=args.val_images,
        val_labels=args.val_labels,
        batch_size=args.batch_size,
        manual=args.manual,
        smp=args.smp,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        summary=args.summary,
        preprocess=args.preprocess,
        train=args.train,
        error_analysis=args.error_analysis
    )


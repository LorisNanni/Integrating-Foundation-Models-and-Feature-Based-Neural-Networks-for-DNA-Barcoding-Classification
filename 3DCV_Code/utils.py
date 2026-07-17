import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
from transformers import Trainer
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def generate_and_save_dot_images(folder_path:str, coords:dict[str, list[tuple[float, float]]], generate_images:bool, s_value: float) -> list[str]:
    '''
    Generates and saves dot images for given coordinates. In particular, the resulting images will have black background and white dots, representing the positions given by the provided coordinates.
    Arguments:
      folder_path : path of the folder where images will be saved
      coords : dictionary with coordinates
      generate_images : 1 to generate images, otherwise skip generation. If 0, images paths are still returned.
      s_value : dot size

    Returns:
      images_paths : list of paths to the generated images

    '''
    images_paths = []
    os.makedirs(folder_path, exist_ok=True)
    
    if generate_images:
        plt.style.use('dark_background')
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.axis("off")
        sc = ax.scatter([], [], c='white', s=s_value)

    for idx, positions in enumerate(coords.values()):
        path = ''.join([f"{folder_path}/img_{idx}_coords.png"])
        if generate_images:
            positions = np.array(positions)
            sc.set_offsets(positions)
            fig.canvas.draw()  # redraw the canvas
            fig.savefig(path, format='png', dpi=100)

        images_paths.append(path)

    if generate_images:
        plt.close(fig)

    return images_paths

def multi_collate_fn(examples):
    """
    Aggregates a batch of example dictionaries by stacking their
    pixel_values_xy, pixel_values_yz, and pixel_values_xz tensors and
    collecting their label values into a single tensor, returning a
    dictionary suitable for model input.
    """
    return {
        "pixel_values_xy": torch.stack([ex["pixel_values_xy"] for ex in examples]),
        "pixel_values_yz": torch.stack([ex["pixel_values_yz"] for ex in examples]),
        "pixel_values_xz": torch.stack([ex["pixel_values_xz"] for ex in examples]),
        "labels": torch.tensor([ex["label"] for ex in examples])
    }

def compute_metrics(eval_pred) -> dict:
    '''
    Compute accuracy, precision, recall and f1 score for the predictions of the model.
    Arguments:
      eval_pred : tuple (logits, labels) where logits are the raw predictions of the model and labels are the true labels of the samples.
    Returns:
      A dictionary with the computed metrics.
    '''
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def get_linear_halving_schedule(optimizer, num_warmup_steps, num_training_steps, steps_per_epoch):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            linear_factor = float(current_step) / float(max(1, num_warmup_steps))
        else:
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            linear_factor = max(0.0, 1.0 - progress)
            
        current_epoch = current_step / steps_per_epoch
        halving_factor = 0.5 ** (math.floor(current_epoch / 50))
        
        return linear_factor * halving_factor

    return LambdaLR(optimizer, lr_lambda)

class CustomSchedulerTrainer(Trainer):
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            train_dataloader = self.get_train_dataloader()
            steps_per_epoch = len(train_dataloader)
            
            num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
            
            self.lr_scheduler = get_linear_halving_schedule(
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                steps_per_epoch=steps_per_epoch
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler


# # 3D Chaos Game Representation projections using Vision Transformers for the classification
# ---


# #Imports and dependencies


# !pip install biopython

# #!git clone GITHUB_LINK

# # Mount drive
# from google.colab import drive
# drive.mount('/content/drive')

# # modify if cloned
# %cd /content/drive/MyDrive/3D_CGR_with_ViTs


import os
import random

import numpy as np

import matplotlib.pyplot as plt

from datasets import Dataset
from datasets import Image as DatasetImage

from transformers import AutoImageProcessor, AutoModel
from transformers import TrainingArguments, Trainer

from huggingface_hub import PyTorchModelHubMixin

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.transforms import InterpolationMode

from torchvision.transforms import (
    Compose,
    Normalize,
    ColorJitter,
    GaussianBlur,
    RandomApply,
    ToTensor,
    Resize,
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from cgr_helpers import seq_to_cgr
from dna_datasets import DNADataset

from model import ViTLateFusion
from utils import generate_and_save_dot_images, multi_collate_fn, compute_metrics, CustomSchedulerTrainer, get_linear_halving_schedule
import math

#model_name = "google/vit-large-patch16-224"
#model_name = "google/vit-base-patch16-384"
#model_name = "facebook/dinov2-small"
# The above methods were worse in performance (maybe due to a lower patch size)

#model_name = "brunoasm/vit_large_patch32_224"    # Same results as the used one

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D-CGR ViT model")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name to use (e.g.: fish, beetle, 'GNe50000 Filtered', ...)")
    parser.add_argument("--generate_images", action="store_true", help="Generate and save the dot images")
    parser.add_argument("--save_logits", action="store_true", help="Save the logits in memory on the test set")
    parser.add_argument("--data_path", type=str, default="data", help="Path to the data folder")
    parser.add_argument("--images_path", type=str, default="images", help="Path to the images folder")
    parser.add_argument("--model_name", type=str, default="timm/vit_large_patch32_224.orig_in21k", help="ViT model to use")
    parser.add_argument("--batch_size", type=int, default=4, help="Per device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    return parser.parse_args()

args = parse_args()
model_name = args.model_name
dataset_name = args.dataset_name

# Deal with the model differently, based on the one we use for the training
if model_name in ["timm/vit_large_patch32_224.orig_in21k", "brunoasm/vit_large_patch32_224"]:
    import timm   # Lazy import
    model = timm.create_model(model_name, pretrained=False)
    data_cfg = timm.data.resolve_model_data_config(model)
    image_mean = list(data_cfg['mean'])
    image_std  = list(data_cfg['std'])
    size       = data_cfg['input_size'][1]  # (C, H, W) → take H
    processor  = None
else:
    processor = AutoImageProcessor.from_pretrained(model_name)
    image_mean, image_std = processor.image_mean, processor.image_std
    if "shortest_edge" in processor.size:
        size = processor.size["shortest_edge"]
    else:
        size = processor.size["height"]

# Normalizes the image pixels by subtracting the mean and dividing by the std from the pretrained model configurations
normalize = Normalize(mean=image_mean, std=image_std)


# Compose: Combines a series of image transformations into one pipeline.
train_transforms = Compose(
    [
        Resize((size, size)),
        RandomApply([ColorJitter(brightness=0.03, contrast=0.03)], p=0.3),
        RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 0.4))], p=0.3),
        ToTensor(),
        normalize,
    ]
)
# Validation not used, as the majority of the datasets were already too small
val_transforms = Compose(
   [
       Resize((size, size)),
       ToTensor(),
       normalize,
   ]
)
test_transforms = Compose(
    [
        Resize((size, size)),
        ToTensor(),
        normalize,
    ]
)


def apply_train_transforms(examples):
    '''
    Applies the train_transforms to the images in the examples.

    Arguments:
      examples : A dictionary containing the images to be transformed.
      It should have the keys 'xy_image', 'yz_image', and 'xz_image',
      each containing a list of images corresponding to the three sides ofthe tetrahedron for each sequence.

    Returns:
      examples : The input examples dictionary with the transformed images added under the keys 'pixel_values_xy', 'pixel_values_yz', and 'pixel_values_xz'.
    '''

    xy_p, yz_p, xz_p = [], [], []

    for xy, yz, xz in zip(examples['xy_image'], examples['yz_image'], examples['xz_image']):
      xy_p.append(train_transforms(xy.convert("RGB")))
      yz_p.append(train_transforms(yz.convert("RGB")))
      xz_p.append(train_transforms(xz.convert("RGB")))

    examples['pixel_values_xy'] = xy_p
    examples['pixel_values_yz'] = yz_p
    examples['pixel_values_xz'] = xz_p

    return examples

def apply_test_transforms(examples):
    '''
    Applies the test_transforms to the images in the examples.
    The same random transformations are applied to all three images of the same sequence, to maintain the correspondence between them.

    Arguments:
      examples : A dictionary containing the images to be transformed.
                It should have the keys 'xy_image', 'yz_image', and 'xz_image',
                each containing a list of images corresponding to the three sides of the tetrahedron for each sequence.

    Returns:
      examples : The input examples dictionary with the transformed images added under the keys 'pixel_values_xy', 'pixel_values_yz', and 'pixel_values_xz'.
    '''
    xy_p, yz_p, xz_p = [], [], []

    for xy, yz, xz in zip(examples['xy_image'], examples['yz_image'], examples['xz_image']):
      seed = np.random.randint(2147483647) # Create a shared seed

      # Apply exactly the same random transform to all three
      random.seed(seed); torch.manual_seed(seed)
      xy_p.append(test_transforms(xy.convert("RGB")))

      random.seed(seed); torch.manual_seed(seed)
      yz_p.append(test_transforms(yz.convert("RGB")))

      random.seed(seed); torch.manual_seed(seed)
      xz_p.append(test_transforms(xz.convert("RGB")))

    examples['pixel_values_xy'] = xy_p
    examples['pixel_values_yz'] = yz_p
    examples['pixel_values_xz'] = xz_p

    return examples

# Using argparse values
generate_images = args.generate_images
save_logits     = args.save_logits
data_path       = args.data_path
images_path     = args.images_path


print("="*80)
print(f"Tuning the model {model_name} using the {dataset_name} dataset")
print("="*80)

simulated_type = ""
dataset_types = []

if dataset_name == "unseen":
    dataset_types.append("supervised_train")
    dataset_types.append("unseen")
elif dataset_name.startswith("GNe50000"):
    # Deal with GNe datasets
    try:
      simulated_type = dataset_name.split(" ")[1]
    except IndexError:
      raise IndexError("Invalid type! It is needed to specify *name of simulated datasets* *\"Filtered\" or \"Unfiltered\"* (e.g.: GNe50000 Filtered)")
    if not simulated_type  in ["Filtered", "Unfiltered"]:
        raise ValueError(f"{simulated_type} is not a valid option for simulated dataset type (*\"Filtered\" or \"Unfiltered\"*).")
    dataset_name = dataset_name.split(" ")[0]
    dataset_types.append("train")
    dataset_types.append("test")
else:
    dataset_types.append("Train")
    dataset_types.append("Test")

if dataset_name.startswith("GNe50000"):
  number_of_datasets = 100
elif dataset_name in ["beetle", "fish"]:
  number_of_datasets = 10
else:
  number_of_datasets = 1


# Main code:
# * Dataset definition
# * Images creation or retrieval
# * ViT fine-tuning
# * Results computation

print(torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))


accuracies = []     # Store all the accuracies, to print the mean value at the end

for fold_index in range(number_of_datasets):

  print("-" * 30)
  print("Fold number", fold_index)
  print("-" * 30)

  files_identifier = f"_{fold_index}" if dataset_name in ["GNe50000", "fish", "beetle"] else "" # Pure formattation purpose

  """Create images for both train and test set, given the dataset name
  ---
  """
  max_len = 0
  label_set = None

  train_datasets = []
  test_datasets = []

  train_stacked_dataset = []
  test_stacked_dataset = []

  full_dataset_train = None
  full_dataset_test = None

  for dataset_type in dataset_types:    # to process both train and test datasets

    indexes_trainset = None
    indexes_testset = None

    if dataset_name == "unseen":
        dataset_path = data_path + "/" + "CanadianInvertebrates-ML" + "/" + dataset_type + ".fas"
    elif dataset_name.startswith("GNe"):
        dataset_path = os.path.join(data_path,"SimulatedDATA", "forBLOG", simulated_type, "Simulated", f"{dataset_name}", f"{dataset_name}r{fold_index}_{dataset_type}.fasta")
    elif dataset_name in ["beetle", "fish"]:
        # Deal with the .mat file to execute the code using beetle and fish datasets (using 10-fold cross validation)
        import scipy.io     # Lazy import to avoid dependency if not using beetle or fish datasets
        mat_folds = scipy.io.loadmat(os.path.join(data_path, "splitFish.mat" if dataset_name == "fish" else "splitBeetle.mat"))

        dataset_path = os.path.join(data_path, f"{dataset_name}-seq.fas")

        indexes_trainset = mat_folds["trainPattern"][0][fold_index][0]
        indexes_testset = mat_folds["testPattern"][0][fold_index][0]

        print(f"Len of indexes {dataset_type}:", len(indexes_trainset) if dataset_type == "Train" else len(indexes_testset))
    else:
        dataset_path = data_path + "/" + dataset_name + dataset_type + ".fas"


    dataset = DNADataset(
        file_path=dataset_path,
        dataset_format=dataset_name,
        max_len=max_len,    # 0 for the train set, so it computes automatically the right value
        label_set=label_set, # None for the train set, so it computes the label set to follow
        choose_indexes_set=indexes_trainset if dataset_type in ["Train"] else indexes_testset,
        replace_ambiguous_bases_with_random=False
      )

    label_set = dataset.label_set

    labels = dataset.labels
    id_seq_map = dataset.id_seq_map
    max_len = dataset.max_len

    all_labels = [label_set[label] for label in labels]

    dataset_cg = [(name, seq_to_cgr(seq.upper())) for name, seq in id_seq_map.items()]


    """2D coordinates computation (retrieving them from the already computed 3D-CGR)"""

    items = dataset_cg  # contains tuples (sequence id, barcode 3d-cgr coordinates recarray)

    # For each sequence, retrieve the coordinates for each side of the tetrahedron
    # We will have 3 dictionaries: one for each side of the tetrahedron
    xy_coords = {}
    yz_coords = {}
    xz_coords = {}

    print("Retrieving 2D coordinates for the CGR projections...")
    print("Number of sequences to process:", len(items))

    for i in range(len(items)):   # i := iterator for the sequences; j := iterator for all the coordinates of the cgr for the sequence i
      x_coords = [items[i][1][j][0] for j in range(len(items[i][1]))]
      y_coords = [items[i][1][j][1] for j in range(len(items[i][1]))]
      z_coords = [items[i][1][j][2] for j in range(len(items[i][1]))]

      xy_cgr_points = [(x_coords[j], y_coords[j]) for j in range(len(x_coords))]
      yz_cgr_points = [(y_coords[j], z_coords[j]) for j in range(len(x_coords))]
      xz_cgr_points = [(x_coords[j], z_coords[j]) for j in range(len(x_coords))]

      xy_coords.update({items[i][0]: xy_cgr_points})
      yz_coords.update({items[i][0]: yz_cgr_points})
      xz_coords.update({items[i][0]: xz_cgr_points})

    s_value = max(1, min(20, 1500 / dataset.max_len))


    if generate_images: print("\nGenerating images:")
    if dataset_type == "Test":
      pre = generate_images
      # generate_images = False
      xy_images_paths = generate_and_save_dot_images(''.join([f"{images_path}/{dataset_name}{simulated_type}{files_identifier}/{dataset_type}_xy"]), xy_coords, generate_images, s_value)
      yz_images_paths = generate_and_save_dot_images(''.join([f"{images_path}/{dataset_name}{simulated_type}{files_identifier}/{dataset_type}_yz"]), yz_coords, generate_images, s_value)
      xz_images_paths = generate_and_save_dot_images(''.join([f"{images_path}/{dataset_name}{simulated_type}{files_identifier}/{dataset_type}_xz"]), xz_coords, generate_images, s_value)
      generate_images = pre
    else:
      pre = generate_images
      # generate_images = False
      xy_images_paths = generate_and_save_dot_images(''.join([f"{images_path}/{dataset_name}{simulated_type}{files_identifier}/{dataset_type}_xy"]), xy_coords, generate_images, s_value)
      yz_images_paths = generate_and_save_dot_images(''.join([f"{images_path}/{dataset_name}{simulated_type}{files_identifier}/{dataset_type}_yz"]), yz_coords, generate_images, s_value)
      xz_images_paths = generate_and_save_dot_images(''.join([f"{images_path}/{dataset_name}{simulated_type}{files_identifier}/{dataset_type}_xz"]), xz_coords, generate_images, s_value)
      generate_images = pre

    print("xy_images_paths size: ", len(xy_images_paths))
    print("yz_images_paths size: ", len(yz_images_paths))
    print("xz_images_paths size: ", len(xz_images_paths))


    id2label = {id:label for id, label in enumerate(label_set)}
    label2id = {label:id for id,label in id2label.items()}

    if not xy_coords.keys() == yz_coords.keys() == xz_coords.keys():
      raise ValueError("Keys of xy_coords, yz_coords, xz_coords do not match!")

    print("-------------------------------------------------------\n\n")

    ids = [label2id[label] for label in all_labels]

    dataset_xy = Dataset.from_dict({"image": xy_images_paths, "label": ids})
    dataset_xy = dataset_xy.cast_column("image", DatasetImage())
    dataset_yz = Dataset.from_dict({"image": yz_images_paths, "label": ids})
    dataset_yz = dataset_yz.cast_column("image", DatasetImage())
    dataset_xz = Dataset.from_dict({"image": xz_images_paths, "label": ids})
    dataset_xz = dataset_xz.cast_column("image", DatasetImage())

    if dataset_type in ["train", "Train", "supervised_train"]:
      train_datasets.extend([dataset_xy, dataset_yz, dataset_xz])
      full_dataset_train = Dataset.from_dict({"xy_image": xy_images_paths, "yz_image": yz_images_paths, "xz_image": xz_images_paths, "label": ids})
      full_dataset_train = full_dataset_train.cast_column("xy_image", DatasetImage())
      full_dataset_train = full_dataset_train.cast_column("yz_image", DatasetImage())
      full_dataset_train = full_dataset_train.cast_column("xz_image", DatasetImage())
    else:
      test_datasets.extend([dataset_xy, dataset_yz, dataset_xz])
      full_dataset_test = Dataset.from_dict({"xy_image": xy_images_paths, "yz_image": yz_images_paths, "xz_image": xz_images_paths, "label": ids})
      full_dataset_test = full_dataset_test.cast_column("xy_image", DatasetImage())
      full_dataset_test = full_dataset_test.cast_column("yz_image", DatasetImage())
      full_dataset_test = full_dataset_test.cast_column("xz_image", DatasetImage())

  if full_dataset_train is None or full_dataset_test is None:
    raise ValueError("full_dataset_train or full_dataset_test is None. Check the dataset_types and dataset_name variables.")
  print(len(full_dataset_train))
  print(len(full_dataset_test))

  full_dataset_train.set_transform(apply_train_transforms)
  full_dataset_test.set_transform(apply_test_transforms)

  multi_train_dl = DataLoader(full_dataset_train, collate_fn=multi_collate_fn)
  batch = next(iter(multi_train_dl))

  for k,v in batch.items():
    if isinstance(v, torch.Tensor):
      print(k, v.shape)

  learning_rate = args.learning_rate

  batch_size = args.batch_size
  gradient_accumulation_steps = args.gradient_accumulation_steps

  train_args = TrainingArguments(
    #output_dir = os.path.join("output-models", f"Vit_{dataset_name}{simulated_type}{files_identifier}"),
    save_total_limit=2,
    learning_rate=learning_rate,
    report_to="tensorboard",
    save_strategy="epoch",
    eval_strategy="epoch",
    warmup_ratio=0.1,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    load_best_model_at_end=False,
    logging_dir='logs',
    remove_unused_columns=False,
    logging_strategy="epoch",  # Logs training metrics (loss) at the end of each epoch
    lr_scheduler_type="linear", # This sets the base HuggingFace linear schedule

  )


  model = ViTLateFusion(model_name, len(id2label))

  from transformers import TrainerCallback

  class PrintEpochCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\n--- Epoch {int(state.epoch)} completed ---")
        print(f"LR: {kwargs['optimizer'].param_groups[0]['lr']:.6f}")

  trainer = Trainer (
    model,
    train_args,
    train_dataset=full_dataset_train,
    eval_dataset=full_dataset_test,
    data_collator=multi_collate_fn,
    processing_class=processor,
    compute_metrics=compute_metrics,
    callbacks=[PrintEpochCallback()], 

  )
  trainer.train()

# save model
checkpoint = {
    "model_state_dict": model.state_dict(),   
}
os.makedirs(f"trainer_output/Vit_{dataset_name}{simulated_type}{files_identifier}", exist_ok=True)   # Create the directory if it doesn't exist
torch.save(checkpoint, f"trainer_output/Vit_{dataset_name}{simulated_type}{files_identifier}/model.pt")


model.eval()

outputs = trainer.predict(full_dataset_test, ignore_keys=["features"])

accuracy = accuracy_score(outputs.label_ids, outputs.predictions.argmax(1))
print("-" * 30)
print(f"Resulting acc for step {fold_index}: {accuracy}")
print("-" * 30)
accuracies.append(accuracy)


print("finished!")

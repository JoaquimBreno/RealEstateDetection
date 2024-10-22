# main.py
import argparse
import json
import os
import datetime
from pathlib import Path
import sys
import math 
from tqdm.auto import tqdm
from distinctipy import distinctipy
import pandas as pd
import multiprocessing
from cjm_pytorch_utils.core import set_seed
from dataloader import CocoDataLoader
from loader import CocoDataset
import wandb

import torch
import torch.optim
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from cjm_pytorch_utils.core import get_torch_device, set_seed, denorm_img_tensor, move_data_to_device
from cjm_torchvision_tfms.core import ResizeMax, PadSquare, CustomRandomIoUCrop
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.amp import autocast


#######################################################################################

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "pretrained.h5")

#######################################################################################

def transforms(size):
    iou_crop = CustomRandomIoUCrop(min_scale=0.3, 
                                max_scale=1.0, 
                                min_aspect_ratio=0.5, 
                                max_aspect_ratio=2.0, 
                                sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                                trials=400, 
                                jitter_factor=0.25)
    # Create a `ResizeMax` object
    resize_max = ResizeMax(max_sz=size)

    # Create a `PadSquare` object
    pad_square = PadSquare(shift=True, fill=0)
    data_aug_tfms = transforms.Compose(
        transforms=[
            iou_crop,  # Assuming 'iou_crop' is properly defined elsewhere as a callable
            transforms.ColorJitter(
                brightness=(0.875, 1.125),
                contrast=(0.5, 1.5),
                saturation=(0.5, 1.5),
                hue=(-0.05, 0.05),
            ),
            transforms.RandomGrayscale(),
            transforms.RandomEqualize(),
            transforms.RandomPosterize(bits=3, p=0.5),
            transforms.RandomHorizontalFlip(p=0.75),  # Adjusted probability to 0.75
        ],
    )

    # Assuming 'resize_max' and 'pad_square' are custom defined functions
    # and 'train_sz' is a defined variable
    resize_pad_tfm = transforms.Compose([
        resize_max,  # Custom transform func
        pad_square,  # Custom transform func
        transforms.Resize([size] * 2, antialias=True),
    ])

    # Assuming the existence of custom transform classes or functions similar to prior definitions
    final_tfms = transforms.Compose([
        transforms.ToImage(),  # Assuming 'ToImage' transform exists
        transforms.ToDtype(torch.float32, scale=True),  # Assuming 'ToDtype' transform exists
        transforms.SanitizeBoundingBoxes(),  # Assuming 'SanitizeBoundingBoxes' transform exists
    ])

    # Define the transformed training and validation datasets
    train_tfms = transforms.Compose([data_aug_tfms, resize_pad_tfm, final_tfms])
    valid_tfms = transforms.Compose([resize_pad_tfm, final_tfms])
    
    return train_tfms,valid_tfms
def dataset_load(size, dataset_dir, device):
    train_tfms, valid_tfms = transforms(size)  
    dataset = CocoDataset()
    coco = dataset.load_coco(dataset_dir, 'train', return_coco=True)
    dataset.prepare()

    print("Train Image Count: {}".format(len(dataset.image_ids)))
    print("Train Class Count: {}".format(dataset.num_classes))
    for i, info in enumerate(dataset.class_info):
        print("{:3}. {:50}".format(i, info['name']))
    train_data = CocoDataLoader(coco,dataset_dir,'train', dataset.class_info, transform=train_tfms)
    
    dataset = CocoDataset()
    coco = dataset.load_coco(dataset_dir, 'val', return_coco=True)
    dataset.prepare()

    print("Val Image Count: {}".format(len(dataset.image_ids)))
    print("Val Class Count: {}".format(dataset.num_classes))
    for i, info in enumerate(dataset.class_info):
        print("{:3}. {:50}".format(i, info['name']))
    val_data = CocoDataLoader(coco,dataset_dir,'val', dataset.class_info, transform=train_tfms)
    
    num_workers = multiprocessing.cpu_count()//2
    bs = 4
    # Define parameters for DataLoader
    data_loader_params = {
        'batch_size': bs,  # Batch size for data loading
        'num_workers': num_workers,  # Number of subprocesses to use for data loading
        'persistent_workers': True,  # If True, the data loader will not shutdown the worker processes after a dataset has been consumed once. This allows to maintain the worker dataset instances alive.
        'pin_memory': 'cuda' in device,  # If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Useful when using GPU.
        'pin_memory_device': device if 'cuda' in device else '',  # Specifies the device where the data should be loaded. Commonly set to use the GPU.
        'collate_fn': lambda batch: tuple(zip(*batch)),
    }
    
    train_dataloader = DataLoader(train_data, **data_loader_params, shuffle=True)
    valid_dataloader = DataLoader(val_data, **data_loader_params)
    return train_dataloader, valid_dataloader


def run_epoch(model, dataloader, optimizer, lr_scheduler, device, scaler, epoch_id, is_training):
    """
    Function to run a single training or evaluation epoch.
    
    Args:
        model: A PyTorch model to train or evaluate.
        dataloader: A PyTorch DataLoader providing the data.
        optimizer: The optimizer to use for training the model.
        loss_func: The loss function used for training.
        device: The device (CPU or GPU) to run the model on.
        scaler: Gradient scaler for mixed-precision training.
        is_training: Boolean flag indicating whether the model is in training or evaluation mode.
    
    Returns:
        The average loss for the epoch.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU disponível. Treinamento iniciará na GPU.")
    else:
        print("GPU não disponível. Treinamento não será iniciado.")
    
    # Set the model to training mode
    model.train()
    
    epoch_loss = 0  # Initialize the total loss for this epoch
    progress_bar = tqdm(total=len(dataloader), desc="Train" if is_training else "Eval")  # Initialize a progress bar
    
    # Loop over the data
    for batch_id, (inputs, targets) in enumerate(dataloader):
        # Move inputs and targets to the specified device
        inputs = torch.stack(inputs).to(device)
        
        # Forward pass with Automatic Mixed Precision (AMP) context manager
        with autocast(torch.device(device).type):
            if is_training:
                losses = model(inputs.to(device), move_data_to_device(targets, device))
            else:
                with torch.no_grad():
                    losses = model(inputs.to(device), move_data_to_device(targets, device))
        
            # Compute the loss
            loss = sum([loss for loss in losses.values()])  # Sum up the losses

        # If in training mode, backpropagate the error and update the weights
        if is_training:
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                old_scaler = scaler.get_scale()
                scaler.update()
                new_scaler = scaler.get_scale()
                if new_scaler >= old_scaler:
                    lr_scheduler.step()
            else:
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                
            optimizer.zero_grad()

        # Update the total loss
        loss_item = loss.item()
        epoch_loss += loss_item
        
        # Update the progress bar
        progress_bar_dict = dict(loss=loss_item, avg_loss=epoch_loss/(batch_id+1))
        if is_training:
            progress_bar_dict.update(lr=lr_scheduler.get_last_lr()[0])
        progress_bar.set_postfix(progress_bar_dict)
        progress_bar.update()

        # If loss is NaN or infinity, stop training
        if is_training:
            stop_training_message = f"Loss is NaN or infinite at epoch {epoch_id}, batch {batch_id}. Stopping training."
            assert not math.isnan(loss_item) and math.isfinite(loss_item), stop_training_message

    # Cleanup and close the progress bar 
    progress_bar.close()
    
    # Return the average loss for this epoch
    return epoch_loss / (batch_id + 1)

def train_loop(model, 
               train_dataloader, 
               valid_dataloader, 
               optimizer,  
               lr_scheduler, 
               device, 
               epochs, 
               checkpoint_path, 
               use_scaler=False):
    """
    Main training loop.
    
    Args:
        model: A PyTorch model to train.
        train_dataloader: A PyTorch DataLoader providing the training data.
        valid_dataloader: A PyTorch DataLoader providing the validation data.
        optimizer: The optimizer to use for training the model.
        lr_scheduler: The learning rate scheduler.
        device: The device (CPU or GPU) to run the model on.
        epochs: The number of epochs to train for.
        checkpoint_path: The path where to save the best model checkpoint.
        use_scaler: Whether to scale graidents when using a CUDA device
    
    Returns:
        None
    """
    # Initialize a gradient scaler for mixed-precision training if the device is a CUDA GPU
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' and use_scaler else None
    best_loss = float('inf')  # Initialize the best validation loss

    # Loop over the epochs
    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Run a training epoch and get the training loss
        train_loss = run_epoch(model, train_dataloader, optimizer, lr_scheduler, device, scaler, epoch, is_training=True)
        # Run an evaluation epoch and get the validation loss
        with torch.no_grad():
            valid_loss = run_epoch(model, valid_dataloader, None, None, device, scaler, epoch, is_training=False)

        # If the validation loss is lower than the best validation loss seen so far, save the model checkpoint
        if valid_loss < best_loss:
            best_loss = valid_loss
            model_path = f"{checkpoint_path}/model_{epoch}.pth"
            torch.save(model.state_dict(), model_path)
            wandb.log_artifact(model_path, type='model', name='best_model')
            # Save metadata about the training process
            training_metadata = {
                'epoch': epoch,
                'train_loss': train_loss,
                'valid_loss': valid_loss, 
                'learning_rate': lr_scheduler.get_last_lr()[0],
                'model_architecture': model.name
            }
            with open(Path(checkpoint_path.parent/'training_metadata.json'), 'w') as f:
                json.dump(training_metadata, f)

    # If the device is a GPU, empty the cache
    if device.type != 'cpu':
        getattr(torch, device.type).empty_cache()
        

    import datetime

def main(dataset_dir, epochs, size, class_names=['background', 'building']):
    # Set seed for reproducibility
    seed = 1234
    set_seed(seed)
    device = get_torch_device()
    dtype = torch.float32
    
    wandb.init(project='iptu', entity='joka', config={
        "epochs": epochs,
        "model_architecture": "mrcnn",
    })

    ROOT_DIR = Path(os.path.abspath(__file__)).parent


    train_dataloader, valid_dataloader = dataset_load(size,dataset_dir, device)
    
    # Definir o modelo, otimizador e agendador de LR
    model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                       epochs=epochs, 
                                                       steps_per_epoch=len(train_dataloader))
    
    wandb.config.update({"learning_rate": optimizer.defaults['lr']})

    # Get the number of input features for the classifier
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # Get the numbner of output channels for the Mask Predictor
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features_box, num_classes=len(class_names))
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=dim_reduced, num_classes=len(class_names))

    # Set the model's device and data type
    model.to(device=device, dtype=dtype);
    model.device = device
    model.name = 'maskrcnn_resnet50_fpn_v2'
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = Path(f"logs/{timestamp}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/f"{model.name}.pth"

    print(checkpoint_path)
    
    lr = 5e-4

    # AdamW optimizer; includes weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Learning rate scheduler; adjusts the learning rate during training
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    max_lr=lr, 
                                                    total_steps=epochs*len(train_dataloader))
    wandb.config.update({"learning_rate": optimizer.defaults['lr']})
    train_loop(model=model, 
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler, 
            device=torch.device(device), 
            epochs=epochs, 
            checkpoint_path=checkpoint_path,
            use_scaler=True)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Treinar um modelo Mask R-CNN com PyTorch e W&B.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Diretório do dataset')
    parser.add_argument('--epochs', type=int, required=True, help='Número de épocas')
    parser.add_argument('--size', type=int, required=True, help='Tamanho das imagens de alimentação do modelo')
    parser.add_argument('--class', type=list, required=False, help='Class names')
    args = parser.parse_args()
    
    main(dataset_dir=args.dataset_dir, epochs=args.epochs, size=args.size, class_names=args.size)
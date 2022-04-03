import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from sklearn.metrics import f1_score
import argparse
import logging
import sys
import os
import copy
import wandb
from typing import Tuple

from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook
import smdebug.pytorch as smd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Function below copied from : https://github.com/awslabs/sagemaker-debugger/blob/master/examples/tensorflow/sagemaker_byoc/simple.py
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def create_train_data_loader(
    data_dir: str, 
    batch_size: int=64, 
    num_workers=0) -> torch.utils.data.DataLoader:
    """Create PyTorch DataLoader for training dataset.

    Args:
        data_dir (str): Folder containing training images.
        batch_size (int, optional): Batch size for training model. Defaults to 64.
        num_workers (int, optional): Number of workers to use for DataLoader. Defaults to 0.

    Returns:
        torch.utils.data.DataLoader: DataLoader for trainning dataset.
    """
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(size=256, scale=(0.85, 1.0)),
        transforms.RandomRotation(degrees=5, expand=True),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageFolder(
        data_dir, 
        transform=train_transforms
        )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers
        )

    return train_loader


def create_val_data_loader(
    data_dir: str, 
    batch_size: int=128, 
    num_workers: int=0) -> torch.utils.data.DataLoader:
    """Create PyTorch DataLoader for validation dataset.

    Args:
        data_dir (str): Folder containing validation images.
        batch_size (int, optional): Batch size for training model. Defaults to 64.
        num_workers (int, optional): Number of workers to use for DataLoader. Defaults to 0.


    Returns:
        torch.utils.data.DataLoader: Dataloader for validation iamges.
    """
    val_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = ImageFolder(
        data_dir, 
        transform=val_transforms
        )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
        )

    return val_loader


def net():
    """Create ResNet18 model.

    This will download a prebuilt PyTorch ResNet18 model with pretrained weights.

    Returns:
        torchvision.models.resnet.ResNet18: Pretrained ResNet18 model.
    """
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.fc.in_features

    model.fc = nn.Sequential(nn.Linear(num_features, 1))

    return model


class Trainer(object):
    """Trainer which takes care of each step of the model training.
    """
    def __init__(self, model, device, loss_fn, optimizer, scheduler, hook):
        """_summary_

        Args:
            model (_type_): PyTorch model to be trained.
            device (str): Whether to train on 'cpu' or 'cuda' (GPU).
            loss_fn (torch.nn): PyTorch loss function.
            optimizer (torch.optim): PyTorch optimizer.
            scheduler (torch.optim): PyTorch learning rate scheduler.
            hook (smdebug.pytorch.Hook): Hook for debugging using SageMaker debugger.
        """
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.hook = hook
    
    def train_step(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float, float]:
        """Conduct one training loop through the training data.

        Args:
            dataloader (torch.utils.data.DataLoader): PyTorch DataLoader.

        Returns:
            epoch_loss (float): Average loss for epoch.
            epoch_accuracy (float): Accuracy of epoch.
            epoch_f1 (float): Epoch F1 score.
        """
        # Set model to train mode
        self.model.train()
        running_loss = 0.0
        running_corrects = 0.0
        running_samples = 0.0
        running_outputs = []
        running_labels = []

        # Set hook mode if using debugging
        if self.hook is not None:
            self.hook.set_mode(modes.TRAIN)

        # Iterate over training batches
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass inputs through model
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs.squeeze(), labels.data.float())

            # Update weights
            loss.backward()
            self.optimizer.step()

            # Update running loss and metrics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum((outputs.squeeze() > 0.0) == labels.data.float())
            running_samples += inputs.size(0)
            running_outputs.append(outputs.squeeze().detach().cpu())
            running_labels.append(labels.detach().cpu())

        # Epoch loss and metrics
        running_outputs = np.concatenate(running_outputs)
        running_labels = np.concatenate(running_labels)
        epoch_loss = running_loss / running_samples
        epoch_accuracy = running_corrects.double() / running_samples
        epoch_f1 = f1_score((running_outputs > 0.0), running_labels)

        return epoch_loss, epoch_accuracy, epoch_f1

    def eval_step(self, dataloader: str) -> Tuple[float, float, float]:
        """Run evaluation step.

        Args:
            dataloader (torch.utils.data.DataLoader): PyTorch DataLoader.

        Returns:
            epoch_loss (float): Average loss for epoch.
            epoch_accuracy (float): Accuracy of epoch.
            epoch_f1 (float): Epoch F1 score.
        """
        # Set model to eval mode
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0.0
        running_samples = 0.0
        running_outputs = []
        running_labels = []

        # Set hook moe if using debugging
        if self.hook is not None:
            self.hook.set_mode(modes.EVAL)

        # Iterate over training batches
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Forward pass inputs through model
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs.squeeze(), labels.data.float())

            # Update running loss and metrics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum((outputs.squeeze() > 0.0) == labels.data.float())
            running_samples += inputs.size(0)
            running_outputs.append(outputs.squeeze().detach().cpu())
            running_labels.append(labels.detach().cpu())

        # Epoch loss and metrics
        running_outputs = np.concatenate(running_outputs)
        running_labels = np.concatenate(running_labels)
        epoch_loss = running_loss / running_samples
        epoch_accuracy = running_corrects.double() / running_samples
        epoch_f1 = f1_score((running_outputs > 0.0), running_labels)

        return epoch_loss, epoch_accuracy, epoch_f1

    def train(
        self, 
        train_dataloader: torch.utils.data.DataLoader, 
        val_dataloader: torch.utils.data.DataLoader, 
        epochs: int=20, 
        early_stopping: bool = True, 
        patience: int = 10, 
        wandb_tracking: bool = False):
        """Train model.

        Args:
            train_dataloader (torch.utils.data.DataLoader): PyTorch DataLoader with training data.
            val_dataloader (torch.utils.data.DataLoader): PyTorch DataLoader with validation data.
            epochs (int, optional): Number of epochs to train model. Defaults to 20.
            early_stopping (bool, optional): Whether to use early stopping. Defaults to True.
            patience (int, optional): Number of epochs to run without validation metric improvement 
                before early stopping applied. Defaults to 10.
            wandb_tracking (bool, optional): Whether tracking with Weights and Biases. Defaults to False.

        Returns:
            _type_: _description_
        """

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_val_f1 = 0.0
        _patience = patience
    
        for epoch in range(epochs):
            train_loss, train_acc, train_f1 = self.train_step(dataloader=train_dataloader)
            val_loss, val_acc, val_f1 = self.eval_step(dataloader=val_dataloader)
            self.scheduler.step(val_loss)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_wts = copy.deepcopy(self.model.state_dict())
                _patience = patience # Reset patience
            else:
                _patience -= 1

            logger.info(f'Train_epoch {epoch}: Train_loss={train_loss}, Train_f1={train_f1}; Train_accuracy={train_acc}; Patience={_patience}')      
            logger.info(f'Val_epoch {epoch}: Val_loss={val_loss}, Val_f1={val_f1}; Val_accuracy={val_acc}; Patience={_patience}')
            if wandb_tracking is True:
                wandb.log({'Train BCE loss': train_loss, 'Train f1': train_f1, 'Train acc': train_acc, 'Val BCE loss': val_loss, 'Val f1': val_f1, 'Val acc': val_acc})
            if not _patience:
                if early_stopping is True:
                    logger.info(f'Validation accuracy not improved for {patience} epochs. Stopping early. Best validation accuracy of {best_val_f1}.')
                    self.model.load_state_dict(best_model_wts) # Load the best model weights
                    break
        
        # Load best model weights
        logger.info(f'Finished training. Best validation score: {best_val_f1}. Saving best model weights.')
        self.model.load_state_dict(best_model_wts)

        return best_val_f1


    def save_model(self, model_dir: str) -> None:
        """Save PyTorch model.

        Args:
            model_dir (dtr): Directory to output saved model.
        """
        logger.info(f"Saving the model to {model_dir}.")
        path = os.path.join(model_dir, "model.pth")
        torch.save(self.model.cpu().state_dict(), path)


def model_fn(model_dir: str):
    """Function to load trained model.

    This function is required by a SageMaker PyTorch endpoint to load the model.

    Args:
        model_dir (str): Location of the trained PyTorch model.

    Returns:
        _type_: _description_
    """
    model = net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model
    

def main(args):
    # Set enviornment variables
    os.environ['SM_OUTPUT_DATA_DIR'] = args.output_dir

    # Login W&B
    if args.wandb_tracking is True:
        with open('./secrets.env', "r") as f:
            key=f.read()
        key = key.split('=')[-1][:-1]
        wandb.login(key=key)
        wandb.init(project=args.wandb_project_name)

    for key, value in vars(args).items():
        print(f"{key}:{value}")

    # Create dataloaders
    train_loader = create_train_data_loader(args.train_data_dir, args.batch_size, args.num_workers)
    val_loader = create_val_data_loader(args.val_data_dir, args.batch_size, args.num_workers)
    logger.info(f'Training dataset contains {len(train_loader.dataset)} images.')
    logger.info(f'Validation dataset contains {len(val_loader.dataset)} images.')

    # Determine whether GPU available and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Training on {device} device')

    # Inialize pretrained model
    model=net().to(device)
   
    # Create loss function and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.1, patience=5)
    
    # Create hook for debugging (if running)
    hook = None
    if args.debugging is True:
        hook = smd.Hook.create_from_json_file()
        hook.register_hook(model)

    # Start model training
    trainer = Trainer(model=model, device=device, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, hook=hook)
    trainer.train(train_dataloader=train_loader, val_dataloader=val_loader, epochs=args.epochs, wandb_tracking=args.wandb_tracking)
    trainer.save_model(model_dir=args.model_dir)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description="PyTorch hyperparameter tuning")
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=os.environ['SM_CHANNEL_TRAIN'],
        help="Path to training data (default: os.environ['SM_CHANNEL_TRAIN'] ",
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        default=os.environ['SM_CHANNEL_VAL'],
        help="Path to validation data (default: os.environ['SM_CHANNEL_VAL'] ",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ['SM_MODEL_DIR'],
        help="Directory to save model (default: os.environ['SM_MODEL_DIR'] ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ['SM_OUTPUT_DATA_DIR'],
        help="Directory to save output data (default: os.environ['SM_OUTPUT_DATA_DIR'] ",
    )
    parser.add_argument(
        "--wandb_tracking",
        type=str2bool,
        default=False,
        help="Whether to track with Weights & Biases (default: False ",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default='aws-test',
        help="Name of W&B project to track results (default: 'aws-test' ",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for training the model (default: 0.001)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to train each model (default: 50)",
    )
    parser.add_argument(
        "--early_stopping",
        type=str2bool,
        default=True,
        help="Whether to apply early stopping during training (default: True)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of epochs without improvement in validation metric before early stopping applied (default: 10)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers to use for PyTorch DataLoader (default: 0)",
    )
    parser.add_argument(
        "--debugging",
        type=str2bool,
        default=False,
        help="Run debugging (default: False)",
    )
    parser.add_argument(
        "--profiler",
        type=str2bool,
        default=False,
        help="Run profiler (default: False)",
    )
    args=parser.parse_args()
    
    main(args)


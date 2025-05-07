import torch
import time
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, precision_score, recall_score, accuracy_score
import os
from collections.abc import Iterable
from torch.amp import GradScaler
import datetime
import numpy as np
from collections import OrderedDict

# Automatic Mixed Precision
# https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html

# Automatic Mixed Precision examples
# https://pytorch.org/docs/stable/notes/amp_examples.html


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()


def train_one_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    device: torch.device, device_type: str, epoch: int, data_loader: Iterable,
                    scaler: GradScaler, loss_fn: torch.nn.Module, args=None):

    model.train(True)

    start_time = time.time()
    start_time_freq = time.time()

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 5 # prints every 5 iterations
    print_counts = 0

    accum_iter = args.accum_iter # increases effective batch size

    current_loss = 0

    for iter_step, data in enumerate(data_loader):

        if iter_step % accum_iter == 0:
            adjusted_lr = adjust_learning_rate(optimizer, iter_step / len(data_loader) + epoch, args)

        inputs, labels = data

        # print(f"Inputs shape is {inputs.shape}")

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True) # , dtype=torch.float32)

        # optimizer.zero_grad() # why zero_grad was here?

        # amp is used for mixed precision training
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=args.use_amp): 
            outputs = model(inputs)
            assert outputs.dtype is torch.float16

            mask = ~torch.isnan(labels).any(dim=1)
            loss = loss_fn(outputs[mask], labels[mask])
            assert loss.dtype is torch.float32, f"The dtype of loss is {loss.dtype}"

        # loss divided by the effective batch size
        loss /= accum_iter
        current_loss += loss

        if iter_step % accum_iter == 0:
            # Scales loss. Calls ``backward()`` on scaled loss to create scaled gradients.
            scaler.scale(current_loss).backward()

            # ``scaler.step()`` first unscales the gradients of the optimizer's assigned parameters.
            # If these gradients do not contain ``inf``s or ``NaN``s, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()
            optimizer.zero_grad() # set_to_none=True here can modestly improve performance
            last_loss = current_loss
            current_loss = 0

        if iter_step % print_freq == 0:
            total_time = str(datetime.timedelta(seconds=int(time.time() - start_time_freq)))
            start_time_freq = time.time()
            print_counts += 1
            print('{} [{}]  eta: {}  lr:{}'.format(header, str(print_counts).rjust(3), total_time, adjusted_lr))

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print('{} Total time epoch: {}'.format(header, total_time))
    print('Final Epoch Stats: loss={}, lr={}'.format(last_loss, adjusted_lr))  

    return last_loss.item()


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: Iterable, device:torch.device,
             device_type: str, loss_fn: torch.nn.Module, args=None):

    header = 'Test:'

    model.train(False)

    total_loss = 0
    all_preds = []
    all_labels = []

    start_time = time.time()

    for data in data_loader:

        inputs, labels = data

        inputs = inputs.to(device, non_blocking=True) #, dtype=torch.float32)
        labels = labels.to(device, non_blocking=True) #, dtype=torch.float32)

        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=args.use_amp): 
            outputs = model(inputs)
            assert outputs.dtype is torch.float16

            # ignore the missing row values and calculate the loss for the available labels
            mask = ~torch.isnan(labels).any(dim=1)
            loss = loss_fn(outputs[mask], labels[mask])
            total_loss += loss.item()
            assert loss.dtype is torch.float32, f"The dtype of loss is {loss.dtype}"

        all_preds.append(outputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss /  len(data_loader)
    
    all_preds = np.concatenate(all_preds, axis=0, dtype=np.float64)
    all_labels = np.concatenate(all_labels, axis=0, dtype=np.float64)

    all_preds = np.clip(all_preds, -1e6, 1e6)

    try:
        mse = mean_squared_error(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)

    except ValueError:
        print("Input contains NaN.")

        mask = ~np.isnan(all_labels).any(axis=1) & ~np.isnan(all_preds).any(axis=1)
        # Apply the mask
        all_labels = all_labels[mask]
        all_preds = all_preds[mask]

        print(all_labels.shape)
        print(all_preds.shape)

        mse = mean_squared_error(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds, multioutput='raw_values')
        print("r2 for the different outputs:")
        print(r2)
        r2 = r2_score(all_labels, all_preds)

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print('{} Total time epoch: {}'.format(header, total_time))
    print('Stats: average loss={}, mse={}, mae={}, r2={}\n'.format(avg_loss, mse, mae, r2))  

    metrics = {
        'avg_loss': avg_loss,
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

    return metrics


@torch.no_grad()
def evaluate_fine_tune(model: torch.nn.Module, data_loader: Iterable, device:torch.device,
             device_type: str, loss_fn: torch.nn.Module, args=None):

    header = 'Test:'

    model.train(False)

    total_loss = 0
    all_preds = []
    all_labels = []

    start_time = time.time()

    for data in data_loader:

        inputs, labels = data

        inputs = inputs.to(device, non_blocking=True) #, dtype=torch.float32)
        labels = labels.to(device, non_blocking=True) #, dtype=torch.float32)

        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=args.use_amp): 
            outputs = model(inputs)
            assert outputs.dtype is torch.float16

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            assert loss.dtype is torch.float32, f"The dtype of loss is {loss.dtype}"

        all_preds.append(outputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss /  len(data_loader)
    
    all_preds = np.concatenate(all_preds, axis=0, dtype=np.float64)
    all_labels = np.concatenate(all_labels, axis=0, dtype=np.float64)

    all_preds = np.clip(all_preds, -1e6, 1e6)

    roc = roc_auc_score(all_labels, all_preds)

    all_preds = (all_preds > 0).astype(int)
    pre = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print('{} Total time epoch: {}'.format(header, total_time))
    print('Stats: average loss={}, roc score={}, accuracy={}, precision={}, recall={}\n'.format(avg_loss, roc, acc, pre, rec))  

    metrics = {
        'avg_loss': avg_loss,
        'accuracy': acc,
        'roc': roc,
        'precision': pre,
        'recall': rec
    }

    return metrics


def load_checkpoint(model, checkpoint_path, device, optimizer=None):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Model checkpoint loaded succesfully")

        return checkpoint

    else:
        print(f"Not checkpoint provided. Starting model training with default model.")

def load_resnet50(cmr_model, checkpoint_path_cmr, device):
    try:
        load_checkpoint(cmr_model, checkpoint_path_cmr, device)
    except:
        print("Loading model with new keys to match model keys...")
        checkpoint = torch.load(checkpoint_path_cmr, map_location=device)
        print(checkpoint['model_state_dict'].keys())
        encoder_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('model.0'):
                new_k = "model.conv1" + k[7:]
                encoder_state_dict[new_k] = v
            elif k.startswith('model.1'):
                new_k = "model.bn1" + k[7:]
                encoder_state_dict[new_k] = v
            elif k.startswith('model.4'):
                new_k = "model.layer1" + k[7:]
                encoder_state_dict[new_k] = v
            elif k.startswith('model.5'):
                new_k = "model.layer2" + k[7:]
                encoder_state_dict[new_k] = v
            elif k.startswith('model.6'):
                new_k = "model.layer3" + k[7:]
                encoder_state_dict[new_k] = v
            elif k.startswith('model.7'):
                new_k = "model.layer4" + k[7:]
                encoder_state_dict[new_k] = v
            elif k.startswith('encoder'):
                encoder_state_dict[k] = v
            elif k.startswith('output_layer'):
                encoder_state_dict[k] = v
            elif k in ['fc.weight', 'fc.bias']:
                encoder_state_dict[k] = v

        cmr_model.load_state_dict(encoder_state_dict, strict=False)
        print("Pre-trained CMR encoder loaded")


def load_resnet50_3D(cmr_model, checkpoint_path_cmr, device):
    try:
        load_checkpoint(cmr_model, checkpoint_path_cmr, device)
    except:
        checkpoint = torch.load(checkpoint_path_cmr, map_location=device)

        encoder_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('model'):
                new_k = "model.blocks" + k[5:]
                encoder_state_dict[new_k] = v
            else:
                encoder_state_dict[k] = v
    
        cmr_model.load_state_dict(encoder_state_dict)
        print("Pre-trained CMR encoder loaded")


def save_checkpoint(model, checkpoint_path: str, optimizer=None, epoch=None,
                    loss=None, args=None):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'epoch': epoch,
        'loss': loss,
        'model_class': model.__class__.__name__,  # Store model class name
    }

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    for filename in os.listdir(checkpoint_path):
        if filename.endswith('.pth'):
            # eliminate other checkpoint in path
            os.remove(os.path.join(checkpoint_path, filename))

    checkpoint_name = os.path.join(checkpoint_path,
        f"{model.__class__.__name__}_checkpoint-{epoch}-loss-{loss}-lr-{args.lr}-wd-{args.weight_decay}-ai-{args.accum_iter}.pth"
    )
    torch.save(checkpoint, checkpoint_name)

    print(f"{model.__class__.__name__} model's checkpoint saved at {checkpoint_path}")
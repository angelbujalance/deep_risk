import torch
import time
import math
from sklearn.metrics import mean_squared_error, r2_score
import os
from collections.abc import Iterable
from torch.amp import GradScaler
import datetime
import numpy as np

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
    print_freq = 20 # prints every 20 iterations
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

            loss = loss_fn(outputs, labels)
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
            print('{} [{}]  eta: {}  lr:{}'.format(header, str(print_counts).rjust(4), total_time, adjusted_lr))

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print('{} Total time epoch: {}'.format(header, total_time))
    print('Final Epoch Stats: loss={}, lr={}'.format(last_loss, adjusted_lr))  

    return last_loss.item()


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: Iterable, device:torch.device,
             device_type: str, loss_fn: torch.nn.Module, args=None):

    header = 'Test:'

    model.train()

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
            print("outputs", outputs[:10])
            assert outputs.dtype is torch.float16

            loss = loss_fn(outputs, labels)
            print("loss", loss)
            total_loss += loss.item()
            print("total_loss", total_loss)
            assert loss.dtype is torch.float32, f"The dtype of loss is {loss.dtype}"

        all_preds.append(outputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss /  len(data_loader)
    print("avg_loss", avg_loss)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    mse = mean_squared_error(all_labels, all_preds)
    print("mse", mse)
    print("all_preds", all_preds[:100])
    print("all_labels", all_labels[:20])
    r2 = r2_score(all_labels, all_preds)

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print('{} Total time epoch: {}'.format(header, total_time))
    print('Stats: average loss={}, mse={}, r2={}'.format(avg_loss, mse, r2))  

    metrics = {
        'avg_loss': avg_loss,
        'mse': mse,
        'r2': r2
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


def save_checkpoint(model, checkpoint_path: str, optimizer=None, epoch=None, loss=None):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'epoch': epoch,
        'loss': loss,
        'model_class': model.__class__.__name__,  # Store model class name
    }

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    checkpoint_name = os.path.join(checkpoint_path,
        f"{model.__class__.__name__}_checkpoint-{epoch}-loss-{loss}.pth"
    )
    torch.save(checkpoint, checkpoint_name)

    print(f"{model.__class__.__name__} model's checkpoint saved at {checkpoint_path}")
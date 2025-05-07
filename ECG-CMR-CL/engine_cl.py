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


def train_one_epoch(clip_cl: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    device: torch.device, device_type: str, epoch: int, data_loader: Iterable,
                    scaler: GradScaler, loss_fn: torch.nn.Module, args=None):

    clip_cl.train(True)

    start_time = time.time()
    start_time_freq = time.time()

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20 # prints every 20 iterations
    print_counts = 0

    accum_iter = args.accum_iter # increases effective batch size

    current_loss = 0.0
    epoch_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    for iter_step, data in enumerate(data_loader):

        if iter_step % accum_iter == 0:
            adjusted_lr = adjust_learning_rate(optimizer, iter_step / len(data_loader) + epoch, args)

        cmr_inputs, ecg_inputs = data

        # print(f"Inputs shape is {inputs.shape}")

        cmr_inputs = cmr_inputs.to(device, non_blocking=True)
        ecg_inputs = ecg_inputs.to(device, non_blocking=True) # , dtype=torch.float32)

        # optimizer.zero_grad() # why zero_grad was here?

        # amp is used for mixed precision training
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=args.use_amp): 
            # TODO The model should be a model designed for CL, that inputs ecg and cmr and returns the combined loss 
            # ecg_output = ecg_model(ecg_inputs)
            # cmr_output = cmr_model(cmr_inputs) # return the latent, then compute loss
            cmr_features, ecg_features = clip_cl(cmr_inputs, ecg_inputs)

            assert ecg_features.dtype is torch.float16
            assert cmr_features.dtype is torch.float16

            # TODO validate this is the way to do the loss for ECG CMR
            # loss_ecg = loss_fn(ecg_output, labels)
            # loss_cmr = loss_fn(cmr_output, labels)

            clip_loss, clip_logits, clip_labels = loss_fn(ecg_features, cmr_features)

            # clip_loss = (1-args.lambda_) * loss_ecg + args.lambda_ * loss_cmr
            assert clip_loss.dtype is torch.float32, f"The dtype of loss is {clip_loss.dtype}"

            # loss_ecg, loss_cmr = model(cmr_inputs, ecg_inputs)
            # assert loss_ecg.dtype is torch.float16
            # assert loss_cmr.dtype is torch.float16

            # clip_loss = (1-args.lambda_) * loss_ecg + args.lambda_ * loss_cmr
            # assert clip_loss.dtype is torch.float32, f"The dtype of loss is {clip_loss.dtype}"

        with torch.no_grad():
            # Get the correct predictions for both modalities
            ecg_to_cmr_pred = torch.argmax(clip_logits, dim=1)
            cmr_to_ecg_pred = torch.argmax(clip_logits.T, dim=1)

            ecg_to_cmr_correct = (ecg_to_cmr_pred == clip_labels).sum().item()
            cmr_to_ecg_correct = (cmr_to_ecg_pred == clip_labels).sum().item()

            correct_predictions += (ecg_to_cmr_correct + cmr_to_ecg_correct) / 2 

            epoch_loss += clip_loss.item() 

            batch_size = cmr_inputs.size(0)
            total_samples += batch_size

        # loss divided by the effective batch size
        clip_loss /= accum_iter
        current_loss += clip_loss

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
            current_loss = 0

        if iter_step % print_freq == 0:
            total_time = str(datetime.timedelta(seconds=int(time.time() - start_time_freq)))
            current_accuracy = 100 * correct_predictions / (total_samples * 2)  
            current_avg_loss = epoch_loss / total_samples
            start_time_freq = time.time()
            print_counts += 1
            print('{} [{}]  eta: {}  lr: {} loss: {} acc: {}'.format(header, str(print_counts).rjust(4), total_time, adjusted_lr, current_avg_loss, current_accuracy))

    
    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    epoch_accuracy = 100 * correct_predictions / (total_samples * 2)
    print("len data and total samples:", total_samples, len(data_loader.dataset))
    epoch_accuracy = 100 * correct_predictions / (len(data_loader.dataset) * 2)
    
    print('{} Total time epoch: {}'.format(header, total_time))
    print('Final Epoch Stats: loss={:.3f}, accuracy={:.2f}, lr={:.3f}'.format(epoch_loss, epoch_accuracy, adjusted_lr))  
    return epoch_loss


# TODO change the eval function
@torch.no_grad()
def evaluate(clip_cl: torch.nn.Module, data_loader: Iterable, device:torch.device,
             device_type: str, loss_fn: torch.nn.Module, args=None):

    header = 'Test:'

    clip_cl.eval()

    total_loss = 0

    start_time = time.time()

    for data in data_loader:

        cmr_inputs, ecg_inputs = data

        cmr_inputs = cmr_inputs.to(device, non_blocking=True)
        ecg_inputs = ecg_inputs.to(device, non_blocking=True)  #, dtype=torch.float32)

        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=args.use_amp):
            ecg_features, cmr_features = clip_cl(cmr_inputs, ecg_inputs)

            clip_loss, clip_logits, clip_labels = loss_fn(ecg_features, cmr_features)
            assert clip_loss.dtype is torch.float32, f"The dtype of loss is {clip_loss.dtype}"
            total_loss += clip_loss.item()


    avg_loss = total_loss /  len(data_loader)
    

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print('{} Total time epoch: {}'.format(header, total_time))
    print('Stats: average loss={}'.format(avg_loss))  

    metrics = {
        'avg_loss': avg_loss,
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
    # save FULL CL model
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
        if filename.endswith('.pth') and filename.startswith("Clip"):
            # eliminate other checkpoint in path
            os.remove(os.path.join(checkpoint_path, filename))

    checkpoint_name = os.path.join(checkpoint_path,
        f"{model.__class__.__name__}_checkpoint-{epoch}-loss-{loss}.pth"
    )
    torch.save(checkpoint, checkpoint_name)

    print(f"{model.__class__.__name__} model's checkpoint saved at {checkpoint_path}")

    # save ECG encoder model
    ecg_checkpoint = {
        'model_state_dict': model.ecg_encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'epoch': epoch,
        'loss': loss,
        'model_class': model.__class__.__name__,  # Store model class name
    }

    for filename in os.listdir(checkpoint_path):
        if filename.endswith('.pth') and filename.startswith("ECGEncoder"):
            # eliminate other checkpoint in path
            os.remove(os.path.join(checkpoint_path, filename))

    checkpoint_name = os.path.join(checkpoint_path,
        f"{model.ecg_encoder.__class__.__name__}_checkpoint-{epoch}-loss-{loss}.pth"
    )

    torch.save(ecg_checkpoint, checkpoint_name)

    print(f"{model.ecg_encoder.__class__.__name__} model's checkpoint saved at {checkpoint_path}")


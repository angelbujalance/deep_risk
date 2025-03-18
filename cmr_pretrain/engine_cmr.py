import torch
import time
import math
from sklearn.metrics import mean_squared_error, r2_score

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


def train_one_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, data_loader: Iterable,
                    epoch_index, tb_writer, scaler: GradScaler, args=None):

    model.train(True)

    start_time = time.time()
    start_time_freq = time.time()

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20 # prints every 20 iterations
    print_counts = 0

    accum_iter = args.accum_iter # increases effective batch size

    current_loss = 0

    loss_fn = torch.nn.CrossEntropyLoss()

    for iter_step, data in enumerate(data_loader):

        if iter_step % accum_iter == 0:
            adjusted_lr = adjust_learning_rate(optimizer, iter_step / len(data_loader) + epoch, args)

        inputs, labels = data

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # amp is used for mixed precision training
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=args.use_amp): 
            outputs = model(inputs)
            assert output.dtype is torch.float16

            loss = loss_fn(outputs, labels)
            assert loss.dtype is torch.float32

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

    return last_loss


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: Iterable):
    loss_fn = torch.nn.CrossEntropyLoss()

    header = 'Test:'

    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    start_time = time.time()

    for data in data_loader:

        inputs, labels = data

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device, dtype=torch.float16, enabled=args.use_amp): 
            outputs = model(inputs)
            assert output.dtype is torch.float16

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            assert loss.dtype is torch.float32

        all_preds.append(outputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / data_loader.size(0)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    mse = mean_squared_error(all_labels, all_preds)
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

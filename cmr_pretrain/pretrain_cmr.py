import torch

import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('CMR pre-training', add_help=False)

    # Main arguments
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--model', default="swin_transformer")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Optimizer arguments
    parser.add_argument('--weight_decay', default=0.005)
    parser.add_argument('--lr', default=0.001, type=)

    #
    parser.add_argument('--use_amp', default=True)

    return parser

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)

    model.to(device, non_blocking=True)
    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    grad_scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp)

    

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

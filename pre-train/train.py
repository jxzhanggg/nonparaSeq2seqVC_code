import os
import time
import argparse
import math
from numpy import finfo
import numpy as np

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import Parrot, ParrotLoss, lcm
from reader import TextMelIDLoader, TextMelIDCollate
from logger import ParrotLogger
from hparams import create_hparams


def batchnorm_to_float(module):
    """Converts batch norm modules to FP32"""
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        batchnorm_to_float(child)
    return module


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelIDLoader(hparams.training_list, hparams.mel_mean_std)
    valset = TextMelIDLoader(hparams.validation_list, hparams.mel_mean_std)
    collate_fn = TextMelIDCollate(lcm(hparams.n_frames_per_step_encoder,
                                      hparams.n_frames_per_step_decoder))

    train_sampler = DistributedSampler(trainset) \
        if hparams.distributed_run else None

    train_loader = DataLoader(trainset, num_workers=1, shuffle=True,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = ParrotLogger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Parrot(hparams).cuda()
    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print(("Warm starting model from checkpoint '{}'".format(checkpoint_path)))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    return model


def load_checkpoint(checkpoint_path, model, optimizer_main, optimizer_sc):
    assert os.path.isfile(checkpoint_path)
    print(("Loading checkpoint '{}'".format(checkpoint_path)))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer_main.load_state_dict(checkpoint_dict['optimizer_main'])
    optimizer_sc.load_state_dict(checkpoint_dict['optimizer_sc'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print(("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration)))
    return model, optimizer_main, optimizer_sc, learning_rate, iteration


def save_checkpoint(model, optimizer_main, optimizer_sc, learning_rate, iteration, filepath):
    print(("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath)))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer_main': optimizer_main.state_dict(),
                'optimizer_sc': optimizer_sc.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                drop_last=True,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss_tts, val_loss_vc = 0.0, 0.0
        reduced_val_tts_losses, reduced_val_vc_losses = np.zeros([9], dtype=np.float32), np.zeros([9], dtype=np.float32)
        reduced_val_tts_acces, reduced_val_vc_acces = np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32)

        for i, batch in enumerate(val_loader):

            x, y = model.parse_batch(batch)

            if i%2 == 0:
                y_pred = model(x, True)
            else:
                y_pred = model(x, False)
            
            losses, acces, l_main, l_sc = criterion(y_pred, y, False)
            if distributed_run:
                reduced_val_losses = []
                reduced_val_acces = []

                for l in losses:
                    reduced_val_losses.append(reduce_tensor(l.data, n_gpus).item())
                for a in acces:
                    reduced_val_acces.append(reduce_tensor(a.data, n_gpus).item())
                
                l_main = reduce_tensor(l_main.data, n_gpus).item()
                l_sc = reduce_tensor(l_sc.data, n_gpus).item()
            else:
                reduced_val_losses = [l.item() for l in losses]
                reduced_val_acces = [a.item() for a in acces]
                l_main = l_main.item()
                l_sc = l_sc.item()
            
            if i%2 == 0:
                val_loss_tts += l_main  + l_sc
                y_tts = y
                y_tts_pred = y_pred
                reduced_val_tts_losses += np.array(reduced_val_losses)
                reduced_val_tts_acces += np.array(reduced_val_acces)
            else:
                val_loss_vc += l_main + l_sc
                y_vc = y
                y_vc_pred = y_pred
                reduced_val_vc_losses += np.array(reduced_val_losses)
                reduced_val_vc_acces += np.array(reduced_val_acces)
            
        if i % 2 == 0:
            num_tts = i / 2 + 1
            num_vc = i / 2
        else:
            num_tts = (i + 1) / 2 
            num_vc = (i + 1) / 2
        
        val_loss_tts = val_loss_tts / num_tts
        val_loss_vc = val_loss_vc / num_vc
        reduced_val_tts_acces = reduced_val_tts_acces / num_tts
        reduced_val_vc_acces = reduced_val_vc_acces / num_vc
        reduced_val_tts_losses = reduced_val_tts_losses / num_tts
        reduced_val_vc_losses = reduced_val_vc_losses / num_vc

    model.train()
    if rank == 0:
        print(("Validation loss {}: TTS {:9f}  VC {:9f}".format(iteration, val_loss_tts, val_loss_vc)))
        logger.log_validation(val_loss_tts, reduced_val_tts_losses, reduced_val_tts_acces, model, y_tts, y_tts_pred, iteration, 'tts')
        logger.log_validation(val_loss_vc, reduced_val_vc_losses, reduced_val_vc_acces, model, y_vc, y_vc_pred, iteration, 'vc')



def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams):
    
    """Training and validation logging results to tensorboard and stdout
    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """

    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate

    parameters_main, parameters_sc = model.grouped_parameters()

    optimizer_main = torch.optim.Adam(parameters_main, lr=learning_rate,
                                 weight_decay=hparams.weight_decay)
    optimizer_sc = torch.optim.Adam(parameters_sc, lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = ParrotLoss(hparams).cuda()

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(checkpoint_path, model)
        else:
            model, optimizer_main, optimizer_sc, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer_main, optimizer_sc)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print(("Epoch: {}".format(epoch)))
        
        for i, batch in enumerate(train_loader):

            start = time.time()
            
            for param_group in optimizer_main.param_groups:
                param_group['lr'] = learning_rate
            
            for param_group in optimizer_sc.param_groups:
                param_group['lr'] = learning_rate
            


            model.zero_grad()
            x, y = model.parse_batch(batch)

            if i % 2 == 0:
                y_pred = model(x, True)
                losses, acces, l_main, l_sc  = criterion(y_pred, y, True)
            else:
                y_pred = model(x, False)
                losses, acces, l_main, l_sc  = criterion(y_pred, y, False)

            if hparams.distributed_run:
                reduced_losses = []
                for l in losses:
                    reduced_losses.append(reduce_tensor(l.data, n_gpus).item())
                reduced_acces = []
                for a in acces:
                    reduced_acces.append(reduce_tensor(a.data, n_gpus).item())
                redl_main = reduce_tensor(l_main.data, n_gpus).item()
                redl_sc = reduce_tensor(l_sc.data, n_gpus).item()
            else:
                reduced_losses = [l.item() for l in losses]
                reduced_acces = [a.item() for a in acces]
                redl_main = l_main.item()
                redl_sc = l_sc.item()

            for p in parameters_sc:
                p.requires_grad_(requires_grad=False)
          
            l_main.backward(retain_graph=True)
            grad_norm_main = torch.nn.utils.clip_grad_norm_(
                parameters_main, hparams.grad_clip_thresh)

            optimizer_main.step()

            for p in parameters_sc:
                p.requires_grad_(requires_grad=True)
            for p in parameters_main:
                p.requires_grad_(requires_grad=False)
            
         
            l_sc.backward()
            grad_norm_sc = torch.nn.utils.clip_grad_norm_(
                parameters_sc, hparams.grad_clip_thresh)


            optimizer_sc.step()

            for p in parameters_main:
                p.requires_grad_(requires_grad=True)

            if not math.isnan(redl_main) and rank == 0:

                duration = time.time() - start
                task = 'TTS' if i%2 == 0 else 'VC'
                print(("Train {} {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    task, iteration, redl_main+redl_sc, grad_norm_main, duration)))
                logger.log_training(
                    redl_main+redl_sc, reduced_losses, reduced_acces, grad_norm_main, learning_rate, duration, iteration)

            if (iteration % hparams.iters_per_checkpoint == 0):
                validate(model, criterion, valset, iteration,
                         hparams.batch_size, n_gpus, collate_fn, logger,
                         hparams.distributed_run, rank)
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer_main, optimizer_sc, learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load the model only (warm start)')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print(("Distributed Run:", hparams.distributed_run))
    print(("cuDNN Enabled:", hparams.cudnn_enabled))
    print(("cuDNN Benchmark:", hparams.cudnn_benchmark))

    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
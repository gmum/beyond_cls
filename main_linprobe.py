
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
from torchvision.datasets import STL10

import models_simmim
# assert timm.__version__ == "0.3.2" # version check
# from timm.models.layers import trunc_normal_

import util.misc as misc
from abmilp import ABMILPHead
from models_vit import CLS_FT_CHOICES
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler, AMP_PRECISIONS
from util.lars import LARS
from util.crop import RandomResizedCrop

import models_vit

from engine_finetune import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('MAE linear probing for image classification', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')
    parser.add_argument('--optimizer', type=str, default="lars", choices=['lars', 'sgd'])

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')

    parser.add_argument("--cls_features",
                        choices=CLS_FT_CHOICES,
                        default="cls", help="cls token / positional tokens for classification")
    parser.add_argument("--return_block", type=int, default=None)
    parser.add_argument("--checkpoint_key", default="model", type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default=Path('/datasets01/imagenet_full_size/061417/'), type=Path,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument("--dataloader_affinity_hack", "-dlah",
                        action='store_true',
                        help="See: https://github.com/pytorch/pytorch/issues/101850#issuecomment-1717363898")
    parser.add_argument("--amp", default="float16", choices=list(AMP_PRECISIONS.keys()), type=str)

    parser.add_argument("--no_cls_token", action='store_true', default=False,
                        help="Disable CLS token (e.g. for I-JEPA). You still have to select appropriate --cls_features"
                        )
    parser.add_argument("--simmim", action="store_true", default=False)

    parser.add_argument("--abmilp_act", choices=["tanh", "relu"], default="tanh",
                        help="abmilp activation function"
                        )
    parser.add_argument("--abmilp_sa", choices=["none", "map", "both"], default="both",
                        help="how to apply the self-attention in abmilp"
                        )
    parser.add_argument("--abmilp_depth", type=int, default=2, help="depth of abmilp head")

    parser.add_argument("--abmilp_cond", type=str, choices=["none", "pe"],
                        help="what to condition abmilp with?")

    parser.add_argument("--abmilp_content", type=str, choices=["all", "patch"], default="all")

    parser.add_argument("--suffix", type=str, default="")


    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # linear probe: weak augmentation
    transform_train = transforms.Compose([
            RandomResizedCrop(224, interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    if "stl10" not in str(args.data_path):
        dataset_train = datasets.ImageFolder(args.data_path / 'train', transform=transform_train)
        dataset_val = datasets.ImageFolder(args.data_path / 'val', transform=transform_val)
    else:
        dataset_train = STL10(args.data_path, split="train", transform=transform_train, download=True)
        dataset_val = STL10(args.data_path, split='test', transform=transform_val, download=True)


    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    args.eff_batch_size = eff_batch_size

    if global_rank == 0 and args.output_dir is not None and not args.eval:
        misc.maybe_setup_wandb(
            args.output_dir, args=args,
            job_type="linprobe_v1", run_name_suffix=args.suffix
        )
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    def worker_init_fn(worker_id):
        os.sched_setaffinity(0, range(os.cpu_count()))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=worker_init_fn if args.dataloader_affinity_hack else None
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        worker_init_fn=worker_init_fn if args.dataloader_affinity_hack else None
    )
    if args.simmim:
        model = models_simmim.__dict__[args.model](
            checkpoint_path=args.finetune
        )
    else:
        cls_kwargs = dict()
        if "huge" in args.model:
            cls_kwargs["class_token"] = not args.no_cls_token
        model: models_vit.VisionTransformer = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            **cls_kwargs
        )

    if args.finetune and not args.eval and not args.simmim:
        if Path(args.finetune).exists():
            print("Interpreting", args.finetune, "as path")
            checkpoint_model = torch.load(args.finetune, map_location='cpu')[args.checkpoint_key]
        else:
            print("Interpreting", args.finetune, "as timm model")
            from timm.models.vision_transformer import _create_vision_transformer

            model_to_kwargs = {
                "vit_tiny_patch16": dict(patch_size=16, embed_dim=192, depth=12, num_heads=12),
                "vit_small_patch16": dict(patch_size=16, embed_dim=384, depth=12, num_heads=12),
                "vit_base_patch16": dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
                "vit_large_patch16": dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
                "vit_huge_patch14": dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16),
            }
            model_kwargs = model_to_kwargs[args.model]
            checkpoint_model = _create_vision_transformer(args.finetune, pretrained=True, **model_kwargs).state_dict()

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        try:
            interpolate_pos_embed(model, checkpoint_model)
        except Exception as e:
            print("couldn't interpolate bc of", e)
            print("Is [cls] switched off?", args.no_cls_token)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)


        assert all([
            k.startswith("head") or k.startswith("oracle") or k.startswith("fc")
            for k in msg.missing_keys
        ]), sorted(msg.missing_keys)


    if args.cls_features.startswith("abmilp"):
        abmilp = ABMILPHead(
                dim=model.head.in_features,
                self_attention_apply_to=args.abmilp_sa,
                activation=args.abmilp_act,
                depth=args.abmilp_depth,
                cond=args.abmilp_cond,
                content=args.abmilp_content,
                num_patches=model.patch_embed.num_patches,

            )
        model.head = torch.nn.Sequential(
            abmilp,
            torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
            model.head
        )
    else:
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.optimizer == "lars":
        optimizer = LARS(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = SGD(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(optimizer)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args
        )
        test_stats = evaluate(data_loader_val, model, device, cls_features=args.cls_features, return_block=args.return_block)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp.head, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, test_stats=log_stats, include_epoch_in_filename=False)

        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar(f'test_v1_{args.cls_features}/train_acc1', train_stats['acc1'], epoch)
            log_writer.add_scalar(f'test_v1_{args.cls_features}/train_loss', train_stats['loss'], epoch)
            log_writer.add_scalar(f'test_v1_{args.cls_features}/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar(f'test_v1_{args.cls_features}/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar(f'test_v1_{args.cls_features}/test_loss', test_stats['loss'], epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

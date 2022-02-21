import argparse
import imp
from tabnanny import check
import typing
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from TAU.model.detr import build
from TAU.dataset.coco import build_coco, get_coco_api_from_dataset
from TAU.utils.TAUUtils import collate_fn

def get_args_parser():
    parser = argparse.ArgumentParser("set transformer detector", add_help=False)

    return parser

def main(args):

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentaton only"
    print(args)

    device = torch.device(args.device)

    model, criterion, postprocessors = build(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("numer of params: ", n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone
        }
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_coco(image_set="train", args=args)
    dataset_val = build_coco(image_set="val", args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers)

    base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location="cpu")
        model_without_ddp.detr.load_state_dict(checkpoint["model"])

    output_dir = Path(args.output_dir)
    if args.resume.startwith("http"):
        checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location="cpu", check_hash=True)
    else:
        checkpoint = torch.load(args.resume, map_location="cpu")

    model_without_ddp.load_state_dict(checkpoint["model"])
    if not args.eval and "optimizer" in checkpoint and "lr_scheduler" in checkpoint and "epoch" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate 



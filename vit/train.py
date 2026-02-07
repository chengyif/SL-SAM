import os
import torch
import torch.nn as nn
import numpy as np
from utils.get_args import GetArgs
from utils.get_dataset import GetDataset
from utils.step_lr import StepLR
from utils.validate import Validate
from utils.meters import AverageMeter, ProgressMeter
import pandas as pd
import time
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision import datasets, transforms

from utils.optimizers.sam import SAM
from utils.optimizers.slsam import SLSAM
from utils.optimizers.rst import RST

if __name__ == "__main__":
    args = GetArgs()
    print(args, flush=True)

    if not os.path.exists('result/%s/%s/%s' %(args.dataset, args.model, args.opt)):
        os.makedirs('result/%s/%s/%s' %(args.dataset, args.model, args.opt))

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "cifar100":
        num_classes = 100

    model_path = "..."
    model = AutoModelForImageClassification.from_pretrained(
        model_path,
        num_labels=num_classes,  
        ignore_mismatched_sizes=True 
    )
    model.to(device)
    model_num_para = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded with {model_num_para/1e6:.2f}M trainable parameters.")

    processor = AutoImageProcessor.from_pretrained(model_path)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(processor.size["height"]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(processor.size["height"]),
        transforms.CenterCrop(processor.size["height"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
    if args.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=val_transform)
    elif args.dataset == "cifar100":
        train_dataset = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=2)
    
    criterion = nn.CrossEntropyLoss().to(device)
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    
    if args.opt == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "adasam":
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "rst":
        base_optimizer = torch.optim.AdamW
        optimizer = RST(model.parameters(), base_optimizer, rho=args.rho, s=args.s, lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "slsam":
        base_optimizer = torch.optim.AdamW
        optimizer = SLSAM(model.parameters(), base_optimizer, rho=args.rho, s=args.s, lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/10)
    record = np.zeros((args.epochs, 3))

    train_time = 0.0
    for epoch in range(args.epochs):
        start_time = time.time()
        losses = AverageMeter('Loss', ':6.3f')
        progress = ProgressMeter(len(train_loader), [losses], prefix="Epoch: [{}] Train: ".format(epoch))

        model.train()
        for i, batch in enumerate(train_loader):
            cur_iter = i + epoch * len(train_loader) + 1

            inputs, targets = (b.to(device) for b in batch)
            outputs = model(pixel_values=inputs, labels=targets)
            loss = outputs.loss

            def closure():
                outputs = model(pixel_values=inputs, labels=targets)
                loss = outputs.loss
                loss.backward()
            losses.update(loss.item(), inputs.size(0))
            optimizer.zero_grad()
            loss.backward()

            if args.opt == "adamw":
                optimizer.step()
            elif args.opt in ["adasam", "slsam", "rst"]:
                optimizer.step(closure)             
        end_time = time.time()
        epoch_time = end_time - start_time
        progress.display(i, prefix='')
        scheduler.step()
        acc1, test_loss = Validate(test_loader, model, device, epoch, prefix='')
        print ('Test accuracy: {:6.2f}'.format(acc1), flush=True)
        record[epoch] = [losses.avg, acc1, epoch_time] 

    best_perf = np.array([np.min(record[:,0]), np.max(record[:,1]), np.mean(record[:,2])])
    final_record = np.vstack((record, best_perf))
    record_csv = pd.DataFrame(final_record)
    record_header = ['Losses', 'Test Accuracy', "Train Time"]
    record_csv.to_csv("./result/" + args.dataset + "/" + args.model + "/" + args.opt + "/lr_" + str(args.lr) + "_wd_" + str(args.weight_decay) + "_rho_" + str(args.rho) + "_bz_" + str(args.batch_size) + "_epochs_" + str(args.epochs) + "_sparse_" + str(args.s) + "_seed_" + str(args.seed) + "_record.csv", index=False, header=record_header)
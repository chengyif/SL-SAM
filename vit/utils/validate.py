import torch
import time
from .meters import AverageMeter, ProgressMeter
from .accuracy import Accuracy
import torch.nn as nn

def Validate(val_loader, model, device, epoch, prefix=''):
    criterion = torch.nn.CrossEntropyLoss()
    losses = AverageMeter('Loss', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader), [losses, top1, top5],
        prefix="Epoch: [{}] Test: ".format(epoch))

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inputs, targets = (b.to(device) for b in batch)
            outputs = model(pixel_values=inputs, labels=targets)
            loss = outputs.loss
            logits = outputs.logits
            acc1, acc5 = Accuracy(logits, targets, topk=(1, 5))
            #acc1, acc5 = Accuracy(predictions.logits, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

        progress.display(i, prefix=prefix)

    return top1.avg.item(), losses.avg

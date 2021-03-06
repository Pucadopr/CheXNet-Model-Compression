from utils import AverageMeter, accuracy
import time
import torch
import numpy as np
import sklearn, sklearn.metrics
import torchxrayvision as xrv


def validate(val_loader, model, criterion, device, print_freq):
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, data in enumerate(val_loader):
        data, target = data['img'].to(device), data['lab'].to(device)

        with torch.no_grad():
            input_var = torch.autograd.Variable(data)
            target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            # TODO: Log them into a file or on WandB
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg

def pretrained_model_accuracy(d_nih, model, device):
    
    outs = []
    labs = []
    with torch.no_grad():
        for i in np.random.randint(0,200,100):
            sample = d_nih[i]
            labs.append(sample["lab"])
            out = model(torch.from_numpy(sample["img"]).unsqueeze(0)).cpu()
            out = torch.sigmoid(out)
            outs.append(out.detach().numpy()[0])
    
    for i in range(14):
        if len(np.unique(np.asarray(labs)[:,i])) > 1:
            auc = sklearn.metrics.roc_auc_score(np.asarray(labs)[:,i], np.asarray(outs)[:,i])
        else:
            auc = "(Only one class observed)"
        print(xrv.datasets.default_pathologies[i], auc)

def evaluate_model(model, test_loader, device, criterion=None):

    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy
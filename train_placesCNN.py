# Modified from: https://github.com/CSAILVision/places365/blob/master/train_placesCNN.py
# Changes vs original:
#   - Python 3 compatibility (print statements, async→non_blocking, .data[0]→.item())
#   - Grayscale input (1 channel) for ARCADE angiography images
#   - Grayscale normalization (mean/std for single channel)
#   - Grayscale + RandomVerticalFlip added to transforms
#   - conv1 patched to accept 1 input channel
#   - deprecated transforms replaced (RandomSizedCrop→RandomResizedCrop, Scale→Resize)
#   - volatile=True replaced with torch.no_grad()
#   - learning rate default lowered to 0.01

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description='PyTorch ARCADE Pretraining (Places365 style)')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (expects data/train and data/val subfolders)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 6)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--num_classes', default=365, type=int,
                    help='number of classes in the dataset (auto-set if using class_map.json)')
parser.add_argument('--dataset', default='arcade',
                    help='dataset name (default: arcade)')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)

    # Auto-detect num_classes from class_map.json if present
    class_map_path = os.path.join(args.data, 'train', 'class_map.json')
    if os.path.exists(class_map_path):
        import json
        with open(class_map_path) as f:
            class_map = json.load(f)
        args.num_classes = len(class_map)
        print(f"=> auto-detected {args.num_classes} classes from class_map.json")

    # Create model
    print(f"=> creating model '{args.arch}'")
    model = models.__dict__[args.arch](num_classes=args.num_classes)

    # Patch conv1: 3 channels → 1 channel (grayscale ARCADE images)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    print("=> conv1 patched: 3 channels → 1 channel (grayscale)")

    if torch.cuda.is_available():
        device = torch.device("cuda"); model = torch.nn.DataParallel(model).cuda()
    elif torch.backends.mps.is_available():
        device = torch.device("mps"); model = model.to(device)
    else:
        device = torch.device("cpu"); model = model.to(device)
    print(model)

    # Optionally resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    cudnn.benchmark = torch.cuda.is_available()

    # Data loading
    traindir = os.path.join(args.data, 'train')
    valdir   = os.path.join(args.data, 'val')

    normalize = transforms.Normalize(mean=[0.449], std=[0.226])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),   # valid for medical images
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    criterion  = nn.CrossEntropyLoss().to(device)
    optimizer  = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch, device)
        validate(val_loader, model, criterion, device)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch':      epoch + 1,
            'arch':       args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, args.arch.lower())


def train(train_loader, model, criterion, optimizer, epoch, device):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target = target.to(device, non_blocking=True)
        input  = input.to(device, non_blocking=True)

        output = model(input)
        loss   = criterion(output, target)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(),    input.size(0))
        top1.update(prec1.item(),     input.size(0))
        top5.update(prec5.item(),     input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})')


def validate(val_loader, model, criterion, device):
    batch_time = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device, non_blocking=True)
            input  = input.to(device, non_blocking=True)

            output = model(input)
            loss   = criterion(output, target)

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(),  input.size(0))
            top1.update(prec1.item(),   input.size(0))
            top5.update(prec5.item(),   input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(f'Test: [{i}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})')

    print(f' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}')
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    os.makedirs('checkpoints', exist_ok=True)
    path = os.path.join('checkpoints', filename)
    torch.save(state, path + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(path + '_latest.pth.tar', path + '_best.pth.tar')
        print(f'  ✓ best model saved → checkpoints/{filename}_best.pth.tar')


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Decay LR by 10x every 30 epochs."""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes precision@k for the specified values of k."""
    maxk       = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

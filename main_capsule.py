#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All rights reserved.
#
'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from src import capsule_model
from utils import progress_bar
import pickle
import json

from datetime import datetime

# +
parser = argparse.ArgumentParser(description='Training Capsules using Inverted Dot-Product Attention Routing')

parser.add_argument('--resume_dir', '-r', default='', type=str, help='dir where we resume from checkpoint')
parser.add_argument('--num_routing', default=1, type=int, help='number of routing. Recommended: 0,1,2,3.')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset. CIFAR10 or CIFAR100.')
parser.add_argument('--backbone', default='resnet', type=str, help='type of backbone. simple or resnet')
parser.add_argument('--num_workers', default=2, type=int, help='number of workers. 0 or 2')
parser.add_argument('--config_path', default='./configs/full_rank_2C1F_matrix_for_iterations.json', type=str, help='path of the config')
parser.add_argument('--debug', action='store_true',
                    help='use debug mode (without saving to a directory)')
parser.add_argument('--sequential_routing', action='store_true', help='not using concurrent_routing')


parser.add_argument('--lr', default=0.1, type=float, help='learning rate. 0.1 for SGD')
parser.add_argument('--dp', default=0.0, type=float, help='dropout rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
# -

args = parser.parse_args()
assert args.num_routing > 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
assert args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100'
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

trainset = getattr(torchvision.datasets, args.dataset)(root='../data', train=True, download=True, transform=transform_train)
testset = getattr(torchvision.datasets, args.dataset)(root='../data', train=False, download=True, transform=transform_test)
num_class = int(args.dataset.split('CIFAR')[1])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.num_workers)

print('==> Building model..')
# Model parameters

image_dim_size = 32

with open(args.config_path, 'rb') as file:
    params = json.load(file)

net = capsule_model.CapsModel(image_dim_size,
                    params,
                    args.backbone,
                    args.dp,
                    args.num_routing,
                    sequential_routing=args.sequential_routing)

# +
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

lr_decay = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)


# -

def count_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.numel())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(net)
total_params = count_parameters(net)
print(total_params)

if not os.path.isdir('results') and not args.debug:
    os.mkdir('results')
if not args.debug:
    store_dir = os.path.join('results', datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
    os.mkdir(store_dir)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

loss_func = nn.CrossEntropyLoss()

if args.resume_dir and not args.debug:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(os.path.join(args.resume_dir, 'ckpt.pth'))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)

        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        v = net(inputs)
        
        loss = loss_func(v, targets)
                
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = v.max(dim=1)
            
        total += targets.size(0)
        
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return 100.*correct/total

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)

            targets = targets.to(device)
            
            v = net(inputs)
            
            loss = loss_func(v, targets)
            
            test_loss += loss.item()
            
            _, predicted = v.max(dim=1)
                
            total += targets.size(0)
            
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc and not args.debug:
        print('Saving..')
        state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
        }
        torch.save(state, os.path.join(store_dir, 'ckpt.pth'))
        best_acc = acc
    return 100.*correct/total

# +
results = {
    'total_params': total_params,
    'args': args,
    'params': params,
    'train_acc': [],
    'test_acc': [],
}

total_epochs = 350

for epoch in range(start_epoch, start_epoch+total_epochs):
    results['train_acc'].append(train(epoch))

    lr_decay.step()
    results['test_acc'].append(test(epoch))
# -

if not args.debug:    
    store_file = os.path.join(store_dir, 'dataset_' + str(args.dataset) + '_num_routing_' + str(args.num_routing) + \
                    '_backbone_' + args.backbone + '.dct')

    pickle.dump(results, open(store_file, 'wb'))

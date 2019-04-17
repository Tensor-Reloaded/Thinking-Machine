import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from tm.backprop_utils import backward
from tm.learn_utils import reset_seed
from tm.thinking_machine import TM as Net

from tm.loss_utils import compute_losses
import torchvision
import torchvision.transforms as transforms
import argparse


def train(epoch):
    net.train()

    train_cls_loss = 0
    train_conf_loss = 0
    train_a_m_loss = 0
    train_q_m_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs, all_f_cls, all_confs, actual_depth = net(inputs)
        conf_eval_losses, f_cls_losses, q_m_losses, a_m_losses = compute_losses(targets, all_confs, all_f_cls)

        total_conf_eval_loss, \
        total_final_classifier_loss, \
        total_a_m_loss, \
        total_q_m_loss = backward( net=net,
                                   conf_eval_losses=conf_eval_losses,
                                   final_classifier_losses=f_cls_losses,
                                   q_m_losses=q_m_losses,
                                   a_m_losses=a_m_losses,
        )

        optimizer.step()

        train_cls_loss += total_final_classifier_loss
        train_conf_loss += total_conf_eval_loss
        train_a_m_loss += total_a_m_loss
        train_q_m_loss += total_q_m_loss

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 20 == 0:
            print('epoch : {} [{}/{}]| cls loss: {:.3f} | conf loss: {:.3f} | a loss: {:.3f} | q loss: {:.3f} | acc: {:.3f}'
                  .format(epoch,
                          batch_idx,
                          len(train_loader),
                          train_cls_loss / (batch_idx + 1),
                          train_conf_loss / (batch_idx + 1),
                          train_a_m_loss / (batch_idx + 1),
                          train_q_m_loss / (batch_idx + 1),
                          100. * correct / total),
            )


def test(epoch, best_acc):
    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                print('epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(epoch, batch_idx,
                                                                              len(test_loader),
                                                                              test_loss / (batch_idx + 1),
                                                                              100 * correct / total))

    acc = 100 * correct / total

    if acc > best_acc:
        print('==> Saving model..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('save_model'):
            os.mkdir('save_model')
        torch.save(state, './save_model/ckpt.pth')
        best_acc = acc

    return best_acc


if __name__ == '__main__':
    reset_seed(666)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description='cifar10 classification models')
    parser.add_argument('--lr', default=0.1, help='')
    parser.add_argument('--resume', default=None, help='')
    parser.add_argument('--batch_size', default=128, help='')
    parser.add_argument('--num_worker', default=4, help='')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(device)
    print('==> Preparing data..')
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset_train = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms_train)
    dataset_test = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transforms_test)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_worker)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=100,
                                              shuffle=False, num_workers=args.num_worker)

    # there are 10 classes so the dataset name is cifar-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    print('==> Making model..')
    net = Net(device)
    net = net.to(device)
    if device == 'cuda':
        # net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume is not None:
        checkpoint = torch.load('./save_model/' + args.resume)
        net.load_state_dict(checkpoint['net'])

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.1,
    #                       momentum=0.9, weight_decay=1e-4)
    optimizer = optim.Adam(net.parameters() , lr = 0.001)
    step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)

    best_acc = 0
    if args.resume is not None:
        test(epoch=0, best_acc=0)
    else:
        for epoch in range(300):
            step_lr_scheduler.step()
            train(epoch)
            best_acc = test(epoch, best_acc)
            print('best test accuracy is ', best_acc)

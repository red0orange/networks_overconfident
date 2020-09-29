#!/home/hdh3/anaconda3/bin/python
# encoding: utf-8
"""
@author: red0orange
@file: lenet.py
@time:  上午11:04
@desc:
"""
import torch
import torch.nn as nn
from collections import OrderedDict


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self, n_classes, if_test=False):
        super(LeNet5, self).__init__()
        self.flag_train = None

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu5', nn.ReLU())
        ]))

        if if_test:
            # test的情况下不需要softmax层
            self.fc = nn.Sequential(OrderedDict([
                ('f6', nn.Linear(120, 84)),
                ('relu6', nn.ReLU()),
                ('f7', nn.Linear(84, n_classes)),
            ]))
        else:
            self.fc = nn.Sequential(OrderedDict([
                ('f6', nn.Linear(120, 84)),
                ('relu6', nn.ReLU()),
                ('f7', nn.Linear(84, n_classes)),
                ('sig7', nn.Softmax(dim=-1))
            ]))

    def forward(self, img, flag_train = True):
        self.flag_train = flag_train

        # img = img.to('cuda')
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output

    def get_loss(self, logits, labels):
        return (torch.nn.functional.cross_entropy(logits, labels)) * logits.size(0)
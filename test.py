#!/home/hdh3/anaconda3/bin/python
# encoding: utf-8
'''
@author: red0orange
@file: test.py
@time: 2020/1/13 下午4:26
@desc:
'''

import os
import torch
from torch.utils.data import DataLoader
from torchnet import meter
from datasets import Dataset
from tqdm import tqdm
import numpy as np
from lenet import LeNet5


class Opt:
    checkpoint_path = 'checkout_point/checkout_65.pth'
    model_name = 'LeNet5'
    data_path = 'data'
    device = 'cpu'
    num_class = 4
    num_workers = 4
    batch_size = 4


if __name__ == '__main__':
    opt = Opt()
    model = LeNet5(opt.num_class, if_test=True)
    if opt.checkpoint_path:
        model.load_state_dict(torch.load(opt.checkpoint_path))
    model.to(opt.device)
    img_size = (32, 32)
    test_data = Dataset(opt.data_path, train=False, test=True, img_size=img_size)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(opt.num_class)

    loss_meter.reset()
    confusion_matrix.reset()
    predict_true = 0
    predict_all = 0

    criterion = torch.nn.CrossEntropyLoss()
    for ii, (_, data, label) in tqdm(enumerate(test_dataloader)):
        input = data.to(opt.device)
        target = label.to(opt.device)
        score = model(input)
        loss = criterion(score, target)
        score = score.cpu()
        print('score: ', torch.softmax(score[:, :opt.num_class].detach(), 0))
        confusion_matrix.add(score[:, :opt.num_class].detach(), target.detach())
        target = target.cpu().numpy()
        loss_meter.add(loss.item())
        predict = torch.argmax(score, dim=1).numpy()
        predict_all += predict.shape[0]
        predict_true += np.sum(target == predict)
    print('correct: {}'.format(predict_true / predict_all))
    print('loss: {}'.format(loss_meter.value()[0]))
    print('confusion_matrix: {}'.format(confusion_matrix.value()))
    pass

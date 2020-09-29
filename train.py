#!/home/hdh3/anaconda3/bin/python
# encoding: utf-8
"""
@author: red0orange
@file: train.py
@time:  下午12:53
@desc:
"""
import os
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
from graph import Graph


class Hps:
    n_epoch = 100
    lmbd = 0.00005
    lr = 0.0005
    opt = 'adam'
    frac_perm = 0.5
    batch_size = 128
    pgd_eps = 0.3
    pgd_niter = 5
    at_frac = 0.5
    lowpass = True
    augm = True
    p = 'inf'
    loss = 'max_conf'
    n_classes = 4
    vision_path = 'exp'

    checkpoint_path = None
    model_name = 'LeNet5'
    data_path = 'data'
    checkpoint_save_path = 'checkout_point'
    num_workers = 4
    device = 'cpu'


if __name__ == '__main__':
    hps = Hps()

    img_size = (32, 32)
    train_data = Dataset(hps.data_path, train=True, test=False, img_size=img_size)
    val_data = Dataset(hps.data_path, train=False, test=False, img_size=img_size)
    train_dataloader = DataLoader(train_data, batch_size=hps.batch_size, shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=hps.batch_size, shuffle=False)

    metrics = []

    graph = Graph(hps)
    if not hps.checkpoint_path is None:
        graph.load_model(hps.checkpoint_path)
    stats_names = ['err_rate', 'max_conf', 'loss', 'reg']

    for epoch in range(0, hps.n_epoch):
        # TODO epoch init
        for ii, (_, batch_x, batch_y) in tqdm(enumerate(train_dataloader)):
            # 统一先转到cpu
            graph.model = graph.model.to('cpu')
            batch_x = batch_x.to('cpu')
            batch_y = batch_y.to('cpu')

            graph.adjust_learning_rate(epoch)
            graph.train(batch_x, batch_y, flag_train=True, rub_flag_tf=True, adv_flag_tf=True, max_conf_flag_tf=(hps.loss=='max_conf'),
                        at_frac_tf=hps.at_frac, pgd_niter_tf=hps.pgd_niter)
            pass
        if os.path.exists(os.path.join(hps.checkpoint_save_path, 'checkout_{}.pth'.format(epoch - 2))):
            os.remove(os.path.join(hps.checkpoint_save_path, 'checkout_{}.pth'.format(epoch - 2)))
        graph.save_model(os.path.join(hps.checkpoint_save_path, 'checkout_{}.pth'.format(epoch)))

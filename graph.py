#!/home/hdh3/anaconda3/bin/python
# encoding: utf-8
"""
@author: red0orange
@file: graph.py
@time:  上午10:24
@desc:
"""
import torch
import numpy as np
from lenet import LeNet5
import utils
import regularizers


class Graph(object):
    def __init__(self, hps):
        self.hps = hps
        self.model = LeNet5(hps.n_classes).to(self.hps.device)
        opt_dict = {'sgd': torch.optim.SGD(self.model.parameters(), hps.lr),
                    'adam': torch.optim.Adam(self.model.parameters(), lr=hps.lr)}
        self.optimizer = opt_dict[hps.opt]

        self.device = self.hps.device

        self.count = 0
        pass

    def train(self, batch_x, batch_y, flag_train, rub_flag_tf, adv_flag_tf, max_conf_flag_tf,
              at_frac_tf, pgd_niter_tf):
        # init one time var
        tower_grads = []

        self.hps.n_ex = batch_x.size(0)

        self.optimizer.zero_grad()

        n_clean = self.hps.batch_size
        n_adv = int(at_frac_tf * self.hps.batch_size)

        batch_y = torch.nn.functional.one_hot(batch_y, self.hps.n_classes).float()

        pgd_stepsize_tf = 1.0 * self.hps.pgd_eps / float(pgd_niter_tf) if pgd_niter_tf != 0 else 0.0
        if self.hps.p == 'inf':
            self.hps.p = np.inf
        x_adv, y_adv = utils.gen_adv(self.model, torch.cat([batch_x, batch_x, batch_x], dim=0)[:n_adv], torch.cat([batch_y, batch_y, batch_y], dim=0)[:n_adv], self.hps.p,
                                     self.hps.pgd_eps, pgd_niter_tf, pgd_stepsize_tf, rub_flag_tf, adv_flag_tf, self.hps.frac_perm, self.hps.lowpass, max_conf_flag_tf)
        if flag_train or x_adv.size(0) == 0:
            x_c, y_c = (torch.cat([x_adv, batch_x], dim=0), torch.cat([y_adv, batch_y], dim=0))
        else:
            x_c, y_c = (x_adv, y_adv)

        # for i in range(x_adv.size(0)):
            # print(x_adv[0].size())
            # cv2.imwrite('test/test-{}.jpg'.format(i), (x_adv[i]*255).numpy().transpose((1, 2, 0)))
            # cv2.imshow('test-1', x_adv[0].numpy().transpose((1, 2, 0)))
            # cv2.waitKey(0)

        # 统一转到真正的device
        self.model = self.model.to(self.device)
        x_c = x_c.to(self.device)
        y_c = y_c.to(self.device)

        logits_c = self.model(x_c, flag_train)
        probs_c = torch.nn.functional.softmax(logits_c, dim=-1)
        logits_adv = logits_c[:n_adv]
        logits = logits_c[n_adv:]
        maxclass = torch.argmax(logits_adv, dim=-1)
        if max_conf_flag_tf:
            loss_adv = -self.model.get_loss(logits_adv, maxclass)
        else:
            loss_adv = self.model.get_loss(logits_adv, torch.argmax(y_c[:n_adv],dim=-1))
        loss_clean = self.model.get_loss(logits, torch.argmax(y_c[n_adv:],dim=-1))
        reg_plain = (regularizers.weight_decay(self.model.named_parameters())).cpu()

        loss_ce = (loss_clean + loss_adv)/float(n_clean + n_adv)

        loss_tower = loss_ce + self.hps.lmbd * reg_plain

        loss_tower.backward()

        print()
        print('loss_out: ', loss_adv.item() / n_adv)
        print('loss_clean: ', loss_clean.item() / n_clean)
        print('loss_tower: ', loss_tower.item())
        print('reg_plain: ', reg_plain.item())

        self.count += 1

        grads_vars_in_tower = utils.get_trainable_grad_and_var(self.model.named_parameters())

        tower_grads.append(grads_vars_in_tower)

        grads_vars = utils.average_gradients(tower_grads)

        utils.apply_gradients(self.model, grads_vars)

        self.optimizer.step()
    pass

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if epoch <= int(self.hps.n_epoch * 0.5):
            lr = self.hps.lr
        elif (epoch > int(self.hps.n_epoch * 0.5)) and (epoch <= int(self.hps.n_epoch * 0.75)):
            lr = self.hps.lr / 10
        elif (epoch > int(self.hps.n_epoch * 0.75)) and (epoch <= int(self.hps.n_epoch * 0.9)):
            lr = self.hps.lr / 100
        elif epoch > int(self.hps.n_epoch * 0.9):
            lr = self.hps.lr / 1000
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        pass

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        pass

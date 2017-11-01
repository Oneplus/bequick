#!/usr/bin/env python


class Mixer(object):
    def __init__(self, config, net, criterion, double=True):
        self.type = 'torch.DoubleTensor' if double else 'torch.CudaTensor'
        self.net = net
        self.criterion = criterion
        self.cum_reward_predictors = {}
        self.nnl = {}
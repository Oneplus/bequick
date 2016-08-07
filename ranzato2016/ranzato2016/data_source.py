#!/usr/bin/env python
import pickle
import os
import numpy as np

class DataSource(object):
    """
    Data provider class that takes a binary tokenized dataset, and provides mini-batches.
    """
    def __init__(self,
                 batch_size,
                 root_path,
                 data_type,
                 dct_target,
                 dct_source,
                 bin_thresh,
                 sequence_length):
        self.batch_size = batch_size
        self.root = root_path
        self.dtype = data_type
        self.tdict = dct_target
        self.sdict = dct_source
        self.sepidx = self.tdict.separatorIndex
        self.bin_thresh = bin_thresh
        self.seqlength = sequence_length
        self.padidx_target = self.tdict.paddingIndex
        self.padidx_source = self.sdict.paddingIndex
        self.all_sources = pickle.load(open(os.path.join(self.root, "%s.sources.pkl" % self.dtype), "rb"))
        self.all_targets = pickle.load(open(os.path.join(self.root, "%s.targets.pkl" % self.dtype), "rb"))
        # gather the shared
        self.shard_ids = {}
        ctr = 0
        for i, v in self.all_targets.iteritems():
            gross_size = v.shape[0]
            if gross_size >= self.bin_thresh:
                self.shard_ids[ctr] = i
                ctr += 1
        # create a permutation vector of the shards
        if self.dtype == "train":
            self.perm_vec = np.random.permutation(len(self.shard_ids))
        else:
            self.perm_vec = np.arange(len(self.shard_ids))
        self.curr_shard_num = 0
        self.curr_shard_id = -1
        self.nshards = len(self.shard_ids)

    def reset(self):
        self.curr_shard_num = 0
        self.curr_shard_id = -1
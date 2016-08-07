#!/usr/bin/env python
import logging
import math
import numpy as np


class RewardFactory(object):
    def __init__(self, reward_type, bptt, dict_size, eos_indx, pad_indx, unk_indx, mbsz):
        """

        Parameters
        ----------
        reward_type: str, type of reward, either ROUGE or BLEU
        bptt
        dict_size
        eos_indx
        pad_indx
        unk_indx
        mbsz
        """
        self.reward_type = reward_type
        self.start = 0
        self.dict_size = dict_size
        self.eos_indx = eos_indx
        self.pad_indx = pad_indx
        if unk_indx is None:
            logging.info("dictionary does not have <unk>, we are not skipping then while computing BLEU")
            self.unk_indx = -1
        else:
            self.unk_indx = unk_indx
        self.mbsz = mbsz
        self.reward_val = np.zeros(mbsz)    # a vector with mbsz reward values.
        self.input_pads = np.zeros(mbsz)
        self.target_pads = np.zeros(mbsz)
        self.inputt = np.zeros(bptt - self.start + 1)   # a vector
        self.targett = np.zeros(bptt - self.start + 1)
        self.reset = np.zeros(mbsz)
        self.target = np.zeros(shape=(bptt, mbsz))
        self.input = np.zeros(shape=(bptt, mbsz))
        self.smoothing_val = 1
        self.adjust_bp = True

    def set_start(self, val):
        assert (val >= 0)
        self.start = val

    def get_reward(self, target, input, tt):
        """

        Parameters
        ----------
        target: numpy.array
        input
        tt

        Returns
        -------

        """
        self.reward_val = np.zeros(self.reward_val.shape)
        if self.reward_type == "rouge":
            pass
        else:
            def compute_blue(target, input, tt, args, i):
                start = args["start"]
                bptt = target.shape[0]
                mbsz = args["mbsz"]
                nthreads = args["nthreads"]
                eos_indx = args["eos_indx"]
                pad_indx = args["pad_indx"]
                unk_indx = args["unk_indx"]
                order = args["order"]
                for ss in range(mbsz):
                    target_length = bptt
                    input_length = bptt
                    for step in range(bptt):
                        if target[step, ss] == eos_indx:
                            target_length = step
                            break
                    for step in range(bptt):
                        if input[step, ss] == eos_indx:
                            input_length = step
                            break
                    if input[0, ss] == pad_indx:
                        input_length = 0
                    if target[0, ss] == pad_indx:
                        target_length = 0
                    assert (input_length >= 0 and target_length >= 0)
                    n = min([order, input_length - start, target_length - start])
                    if tt == min(input_length, bptt) and n > 0:
                        # compute BLEU score
                        self.reward_val[ss] = 1.

            for cc in range(0, target.shape[0]):
                self.target[cc,:] = target[cc,:]
                self.input[cc,:] = input[cc,:][0]
            for i in range(self.mbsz):
                args = {
                    "start": self.start,
                    "dict_size": self.dict_size,
                    "eos_indx": self.eos_indx,
                    "pad_indx": self.pad_indx,
                    "unk_indx": self.unk_indx,
                    "mbsz": self.mbsz,
                    "reward_val": self.reward_val,
                    "inputt": self.inputt,
                    "targett": self.targett,
                    "order": self.order,
                    "smoothing_val": self.smoothing_val,
                    "adjust_bp": self.adjust_bp
                }
                compute_blue(self.target, self.input, tt, args, i)
            return self.reward_val

    def training_mode(self):
        self.smoothing_val = 1
        self.adjust_bp = True
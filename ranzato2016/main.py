#!/usr/bin/env python
import random
import argparse
import os
import logging
import pickle
from ranzato2016.data_source import DataSource
from ranzato2016.reward_factory import RewardFactory
random.seed(1111)
logging.basicConfig(format="%(asctime)-15s %(levelname)s: %(message)s", level=logging.INFO)


def main():
    cmd = argparse.ArgumentParser()
    cmd.add_argument("-datadir", default="data", help="path to binarized training data.")
    cmd.add_argument("-lr", dest="lr", default=0.2, type=float, help="learning rate.")
    cmd.add_argument("-gparamclip", default=10, type=int, help="clipping threshold of parameter gradients")
    cmd.add_argument("-bsz", default=32, help="batch size")
    cmd.add_argument("-nhid", default=256, help="number of hidden units")
    cmd.add_argument("-bptt", default=25, help="number of backprop steps through time")
    cmd.add_argument("-deltasteps", default=3, type=int,
                     help="increment of number of words we predict using REINFORCE at next round")
    cmd.add_argument("-nepochs", default=5, type=int,
                     help="number of epochs of each stage of REINFORCE training")
    cmd.add_argument("-epoch_xent", default=25, type=int,
                     help="number of epochs we do with pure XENT to initialize the model")
    cmd.add_argument("-reward", default='bleu', help="reward type: bleu|rouge")
    opts = cmd.parse_args()
    conf = {
        "trainer": {
            "bptt": opts.bptt,
            "n_epochs": opts.nepochs,
            "initial_learning_rate": opts.lr,
            "save_dir": "./backups/"
        },
        "model" : {
            "n_hidden": opts.nhid,
            "batch_size": opts.bsz,
            "bptt": opts.bptt,
            "grad_param_clip": opts.gparamclip,
            "reward": opts.reward
        }
    }
    if not (os.path.isdir(opts.datadir) and
                os.path.isfile(os.path.join(opts.datadir, "dict.target.pkl")) and
                os.path.isfile(os.path.join(opts.datadir, "dict.source.pkl"))):
        logging.error('[[ Data not found: fetching a fresh copy and running tokenizer ]]')
        return

    dict_target = pickle.load(open(os.path.join(opts.datadir, 'dict.target.pkl')))
    dict_source = pickle.load(open(os.path.join(opts.datadir, 'dict.source.pkl')))

    padidx_target = dict_target.nwords
    dict_target.index_to_symbol[padidx_target] = '<PAD>'
    dict_target.symbol_to_index['<PAD>'] = padidx_target
    dict_target.paddingIndex = padidx_target
    dict_target.nwords += 1
    padidx_source = dict_source.nwords
    dict_source.index_to_symbol[padidx_source] = '<PAD>'
    dict_source.symbol_to_index['<PAD>'] = padidx_source
    dict_source.paddingIndex = padidx_source
    dict_source.nwords += 1
    train_data = DataSource(
        root_path=opts.datadir,
        data_type='train',
        batch_size=opts.bsz,
        bin_thresh=800,
        sequence_length=opts.bptt,
        dct_target=dict_target,
        dct_source=dict_source
    )
    valid_data = DataSource(
        root_path=opts.datadir,
        data_type='valid',
        batch_size=opts.bsz,
        bin_thresh=800,
        sequence_length=opts.bptt,
        dct_target=dict_target,
        dct_source=dict_source
    )
    test_data = DataSource(
        root_path=opts.datadir,
        data_type='test',
        batch_size=opts.bsz,
        bin_thresh=800,
        sequence_length=opts.bptt,
        dct_target=dict_target,
        dct_source=dict_source
    )
    conf["model"]["eosIndex"] = dict_target.separaterIndex
    conf["model"]["n_tokens"] = dict_target.nwords
    conf["model"]["paddingIndex"] = dict_target.paddingIndex
    unk_id = dict_target.symbol_to_index['<unk>']

    compute_reward = RewardFactory(conf["model"]["reward"],
                                   opts.bptt,
                                   conf["model"]["n_toknes"],
                                   conf["model"]["eosIndex"],
                                   conf["model"]["paddingIndex"],
                                   unk_id,
                                   opts.bsz)
    compute_reward.training_mode()

if __name__ == "__main__":
    main()

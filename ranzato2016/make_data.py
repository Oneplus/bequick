#!/usr/bin/env python
import os
import argparse
import pickle
import logging
from ranzato2016.tokenizer import Tokenizer

logging.basicConfig(format="%(asctime)-15s %(levelname)s: %(message)s", level=logging.INFO)


def main():
    cmd = argparse.ArgumentParser('Make dictionary and datasets')
    cmd.add_argument("-srcDir", default="prep", help="path to pre-processed data.")
    cmd.add_argument("-dstDir", default="data",
                     help="path to where dictionaries and dataset should be written.")
    cmd.add_argument("-shuff", default=True, type=bool, help="shuffle sentences in training data or not")
    cmd.add_argument("-threshold", default=3, type=int, help="remove words appearing less threshold")
    cmd.add_argument("-isvalid", default=True, type=bool, help="generate validation set")
    cmd.add_argument("-istest", default=True, type=bool, help="generate the test set")
    opt = cmd.parse_args()

    if not os.path.isdir(opt.dstDir):
        os.mkdir(opt.dstDir)

    conf = {
        "root_path": opt.srcDir,
        "dest_path": opt.dstDir,
        "threshold": opt.threshold,
        "targets": {"train": "train.de-en.en", "valid": "valid.de-en.en", "test": "test.de-en.en"},
        "sources": {"train": "train.de-en.de", "valid": "valid.de-en.de", "test": "test.de-en.de"}
    }

    logging.info("-- building target dictionary")
    train_target = os.path.join(opt.srcDir, conf["targets"]["train"])
    target_dic = Tokenizer.build_dictionary(train_target, conf["threshold"])
    pickle.dump(target_dic, open(os.path.join(opt.dstDir, "dict.target.pkl"), 'wb'))

    logging.info("-- building source dictionary")
    train_source = os.path.join(opt.srcDir, conf["sources"]["train"])
    source_dic = Tokenizer.build_dictionary(train_source, conf["threshold"])
    pickle.dump(source_dic, open(os.path.join(opt.dstDir, "dict.source.pkl"), 'wb'))

    logging.info("Tokenizing train...")
    train_targets, train_sources = Tokenizer.tokenize(conf, "train", target_dic, source_dic, opt.shuff)
    logging.info("Tokenizing valid...")
    valid_targets, valid_sources = Tokenizer.tokenize(conf, "valid", target_dic, source_dic, opt.shuff)
    logging.info("Tokenizing test...")
    test_targets, test_sources = Tokenizer.tokenize(conf, "test", target_dic, source_dic, opt.shuff)

    pickle.dump(train_targets, open(os.path.join(conf["dest_path"], "train.targets.pkl"), "wb"))
    pickle.dump(train_sources, open(os.path.join(conf["dest_path"], "train.sources.pkl"), "wb"))
    pickle.dump(valid_targets, open(os.path.join(conf["dest_path"], "valid.targets.pkl"), "wb"))
    pickle.dump(valid_sources, open(os.path.join(conf["dest_path"], "valid.sources.pkl"), "wb"))
    pickle.dump(test_targets,  open(os.path.join(conf["dest_path"], "test.targets.pkl"),  "wb"))
    pickle.dump(test_sources,  open(os.path.join(conf["dest_path"], "test.sources.pkl"),  "wb"))


if __name__ == "__main__":
    main()

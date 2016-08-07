#!/usr/bin/env python
from __future__ import print_function
import re
import os
import logging
import numpy as np


def cleanup_sentence(s):
    """
    Replace "\t" with space and remove the continuous space.

    Parameters
    ----------
    s: str, The input sentence

    Returns
    -------
    ret: str, The output sentence
    """
    return re.sub("\s+", " ", s.replace("\t", "").strip())


def get_source_indices(sent, dic):
    """
    map source sentence words to id vector

    Parameters
    ----------
    sent: str, The source sentence
    dic:

    Returns
    -------
    np.array, int, int:
    """
    clean_sent = cleanup_sentence(sent)
    words = clean_sent.split(' ')
    n_words = len(words) + 1  # counting for the </s>
    indices = np.zeros(n_words)
    cnt = 0
    nsrc_unk = 0
    unk_idx = dic.symbol_to_index["<unk>"]
    eos_idx = dic.symbol_to_index["</s>"]
    for i, word in enumerate(words):
        wid = dic.symbol_to_index.get(word, None)
        if wid is None:
            indices[cnt] = unk_idx
            nsrc_unk += 1
        else:
            indices[cnt] = wid
            if wid == unk_idx:
                nsrc_unk += 1
        cnt += 1
    indices[cnt] = eos_idx
    cnt += 1
    return indices, indices.shape[0], nsrc_unk


def get_target_indices(sent, dic, sidx):
    """

    Parameters
    ----------
    sent: list(str)
    dic:
    sidx: int

    Returns
    -------
    np.array, int, int:
    """
    clean_sent = cleanup_sentence(sent)
    words = clean_sent.split(' ')
    n_words = len(words) + 1
    indices = np.zeros(shape=(n_words, 3))
    cnt = 0
    ntgt_unk = 0
    unk_idx = dic.symbol_to_index["<unk>"]
    indices[cnt][0] = dic.symbol_to_index["</s>"]
    indices[cnt][1] = sidx
    indices[cnt][2] = cnt
    for i, word in enumerate(words):
        wid = dic.symbol_to_index.get(word, None)
        if wid is None:
            indices[cnt][0] = unk_idx
            indices[cnt][1] = sidx
            indices[cnt][2] = cnt
            ntgt_unk += 1
        else:
            indices[cnt][0] = wid
            indices[cnt][1] = sidx
            indices[cnt][2] = cnt
            if wid == unk_idx:
                ntgt_unk += 1
        cnt += 1
    return indices, indices.shape[0], ntgt_unk


class ObjectCreator(object):
    pass


class Tokenizer(object):
    @staticmethod
    def build_dictionary(filename, threshold):
        """

        Parameters
        ----------
        filename: str
        threshold: int

        Returns
        -------

        """
        kMaxDictSize = 5000000
        dic = ObjectCreator()
        dic.symbol_to_index = {"<unk>": 0, "</s>": 1}
        dic.index_to_symbol = {0: "<unk>", 1: "</s>"}
        dic.index_to_freq = np.zeros(kMaxDictSize)
        dic.separatorIndex = dic.symbol_to_index["</s>"]

        n_words = 2
        tot_n_words = 0
        cnt = 0
        logging.info("[ Reading from %s ]" % filename)
        for s in open(filename, 'r'):
            # remove all space and split into words
            words = re.sub("\s+", " ", s.replace("\t", "")).split()
            for i, word in enumerate(words):
                if word not in dic.symbol_to_index:
                    dic.symbol_to_index[word] = n_words
                    dic.index_to_symbol[n_words] = word
                    dic.index_to_freq[n_words] = 1
                    n_words += 1
                else:
                    indx = dic.symbol_to_index[word]
                    dic.index_to_freq[indx] += 1
                cnt += 1
            # count </s> after every line
            indx = dic.symbol_to_index["</s>"]
            dic.index_to_freq[indx] += 1
            cnt += 1

        dic.index_to_freq = dic.index_to_freq[:n_words, ]
        tot_n_words = dic.index_to_freq.sum()
        logging.info("[ Done making the dictionary. ]")
        logging.info("Training corpus statistics")
        logging.info("Unique words: %d" % n_words)
        logging.info("Total words: %d" % tot_n_words)

        removed = 0
        net_words = 1
        if threshold > 0:
            for i in range(2, dic.index_to_freq.shape[0]):
                word = dic.index_to_symbol[i]
                if dic.index_to_freq[i] < threshold:
                    dic.index_to_freq[1] += dic.index_to_freq[i]
                    dic.index_to_freq[i] = 0
                    dic.symbol_to_index[word] = 1
                    removed += 1
                else:
                    net_words += 1
                    dic.index_to_freq[net_words] = dic.index_to_freq[i]
                    dic.symbol_to_index[word] = net_words
                    dic.index_to_symbol[net_words] = word
            logging.info("[ Removed %d rare words. ]" % removed)
            dic.index_to_freq = dic.index_to_freq[:net_words, ]
        else:
            net_words = n_words
        logging.info("[ There are effectively %d words in the corpus. ]" % net_words)
        dic.nwords = net_words
        return dic

    @staticmethod
    def tokenize(conf, dtype, tdict, sdict, shuff):
        """

        Parameters
        ----------
        conf: dict
        dtype: str
        tdict: dic
        sdict: dic
        shuff

        Returns
        -------

        """
        tf = open(os.path.join(conf["root_path"], conf["targets"][dtype]), 'r')
        sf = open(os.path.join(conf["root_path"], conf["sources"][dtype]), 'r')

        source_sent_data, target_sent_data = [], []
        source_sent_len, target_sent_len = {}, {}
        source_sent_ctr, target_sent_ctr = 0, 0
        source_sent_nwords, target_sent_nwords = 0, 0
        max_target_len = 0

        for target_sen, source_sen in zip(tf, sf):
            tclean_sent = cleanup_sentence(target_sen)
            twords = tclean_sent.split(' ')
            sclean_sent = cleanup_sentence(source_sen)
            swords = sclean_sent.split(' ')

            source_sent_data.append(sclean_sent)
            target_sent_data.append(tclean_sent)

            nwords = len(swords) + 1
            source_sent_len[source_sent_ctr] = nwords
            source_sent_nwords += nwords

            nwords = len(twords) + 1
            target_sent_len[target_sent_ctr] = nwords
            target_sent_nwords += nwords
            max_target_len = max(max_target_len, nwords)

            source_sent_ctr += 1
            target_sent_ctr += 1

        assert (source_sent_ctr == target_sent_ctr)
        logging.info('Number of sentences: %d' % target_sent_ctr)
        logging.info('Max target sentence length: %d' % max_target_len)

        bins_data = {}
        n_bins = 0
        for i in range(source_sent_ctr):
            slen = source_sent_len[i]
            if slen not in bins_data:
                n_bins += 1
                bins_data[slen] = {}
                bins_data[slen]["size"] = 1
            else:
                bins_data[slen]["size"] += 1

        for bin_dim, bin in bins_data.iteritems():
            bin_size = bin["size"]
            target_tensor_len = max_target_len * bin_size
            bin["sources"] = np.zeros(shape=(bin_size, bin_dim))
            bin["soffset"] = 0
            bin["targets"] = np.zeros(shape=(target_tensor_len, 3))
            bin["toffset"] = 0

        if shuff:
            logging.info("-- shuffling the data")
            perm_vec = np.random.permutation(target_sent_ctr)
        else:
            logging.info("-- not shuffling the data")
            perm_vec = np.arange(target_sent_ctr)

        logging.info("-- Populate bins")
        nsrc_unk = 0
        ntgt_unk = 0
        nsrc = 0
        ntgt = 0
        for i in range(target_sent_ctr):
            idx = perm_vec[i]
            curr_source_sent = source_sent_data[idx]
            curr_target_sent = target_sent_data[idx]
            bnum = source_sent_len[idx]
            curr_bin = bins_data[bnum]

            curr_source_ids, ssize, nus = get_source_indices(curr_source_sent, sdict)
            curr_target_ids, tsize, nut = get_target_indices(curr_target_sent, tdict, curr_bin["soffset"])
            nsrc += ssize
            ntgt += tsize
            nsrc_unk += nus
            ntgt_unk += nut
            curr_bin["sources"][curr_bin["soffset"],] = curr_source_ids
            curr_bin["targets"][curr_bin["toffset"]: curr_bin["toffset"] + tsize,] = curr_target_ids
            curr_bin["soffset"] += 1
            curr_bin["toffset"] += tsize

        sources = {}
        targets = {}
        for bin_dim, bin in bins_data.iteritems():
            sources[bin_dim] = bin["sources"]
            targets[bin_dim] = bin["targets"]

        logging.info("nlines: %d, ntokens (src: %d, tgt: %d); UNK (src: %.2f%%, tgt: %.2f%%)" %
                     (target_sent_ctr, nsrc, ntgt, (float(nsrc_unk) / nsrc * 100), (float(ntgt_unk) / ntgt * 100)))
        return targets, sources

if __name__ == "__main__":
    print(cleanup_sentence(" a b   c\td  "))

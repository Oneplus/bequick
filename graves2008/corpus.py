#!/usr/bin/env python


def read_dataset(filename):
    dataset = []
    for raw_data in open(filename, 'r').read().strip().split('\n'):
        data = []
        for token in raw_data.split():
            form, pos = token.rsplit('/', 1)
            data.append({'form': form, 'pos': pos})
        dataset.append(data)
    return dataset

def get_alphabet(dataset, keyword):
    ret = {None: 0, 'UNK': 1}
    for data in dataset:
        for item in data:
            form = item[keyword]
            if form not in ret:
                ret[form] = len(ret)
    return ret

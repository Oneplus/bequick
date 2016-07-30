#!/usr/bin/env python


def read_dataset(filename):
    dataset = []
    for raw_data in open(filename, 'r').read().strip().split('\n\n'):
        data = []
        lines = raw_data.split('\n')
        for line in lines:
            tokens = line.split()
            data.append({
                'id': int(tokens[0]),
                'form': tokens[1],
                'pos': tokens[2],
                'head': int(tokens[6]),
                'deprel': tokens[7]
            })
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

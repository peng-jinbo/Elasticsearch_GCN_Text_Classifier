#!/usr/bin/python
#-*-coding:utf-8-*-

import json
import sys

for ct in range(100):

    dataset_name = 'name_{}'.format(ct)

    f = open('dataset/{}.json'.format(dataset_name), encoding='utf-8')
    dataset = json.loads(f.read())

    sentences = dataset['sentences']
    labels = dataset['labels']
    train_or_test_list = dataset['train_or_test_list']

    meta_data_list = []

    for i in range(len(sentences)):
        meta = str(i) + '\t' + train_or_test_list[i] + '\t' + labels[i]
        meta_data_list.append(meta)

    meta_data_str = '\n'.join(meta_data_list)

    f = open('data/' + dataset_name + '.txt', 'w')
    f.write(meta_data_str)
    f.close()

    corpus_str = '\n'.join(sentences)

    f = open('data/corpus/' + dataset_name + '.clean.txt', 'w')
    f.write(corpus_str)
    f.close()

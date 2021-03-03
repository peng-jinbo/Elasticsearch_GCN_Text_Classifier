from __future__ import division
from __future__ import print_function
from sklearn import metrics
import random
import time
import sys
import os
import json

import torch
import torch.nn as nn

import numpy as np

from utils.utils import *
from models.gcn import GCN
from models.mlp import MLP

from config import CONFIG
cfg = CONFIG()

import elasticsearch
import json

es_host = [
    "10.10.11.1", "10.10.11.2", "10.10.11.3", "10.10.11.4", 
    "10.10.11.5", "10.10.11.6", "10.10.11.7", "10.10.11.8"
]
es = elasticsearch.Elasticsearch(es_host, timeout = 30)
es.ping()

def get_ela_scores(words, K=100):
    scores = dict()
    scores_sum = 0
    search_words = words
    ret = es.search(index='affiliation_train', body={
          "size": K,
          "sort": [
            {
              "_score": {
                "order": "desc"
              }
            }
          ],
          "_source": {
            "excludes": []
          },
          "stored_fields": [
            "*"
          ],
          "script_fields": {},
          "docvalue_fields": [],
          "query": {
            "bool": {
              "must": [
                {
                  "query_string": {
                    "query": search_words,
                    "analyze_wildcard": True,
                    "time_zone": "Asia/Shanghai"
                  }
                }
              ],
              "filter": [],
              "should": [],
              "must_not": []
            }
          }
        })
    if len(ret['hits']['hits']) <= 0:
        return scores
    else:
        search_result_list_score = []
        search_result_list_norm = []
        search_result_list_ori = []
        for i, k in enumerate(ret['hits']['hits']):
            #print(i, k['_source']['normalized_name'])
            search_result_list_norm.append(k['_source']['normalized_name'])
            search_result_list_ori.append(k['_source']['original_name'])
            search_result_list_score.append(k['_score'])
        for i in range(len(search_result_list_norm)):
            if search_result_list_norm[i] not in scores.keys():
                scores[search_result_list_norm[i]] = search_result_list_score[i]
            else:
                scores[search_result_list_norm[i]] = max(search_result_list_score[i], scores[search_result_list_norm[i]])
        for i in scores.keys():
            scores_sum += scores[i]
        for i in scores.keys():
            scores[i] = scores[i] / scores_sum
        return scores

def evaluate(features, labels, mask, is_test):
    t_test = time.time()

    model.eval()

    es_pred = None
    mix_pred = None
    with torch.no_grad():
        logits = model(features)
        if is_test > -1:
            correct_gcn = 0
            correct_es = 0
            correct_mix = 0

            count = 0

            es_logits = torch.full_like(logits, 1)
            mix_logits = torch.tensor(logits)

            current_ind = logits.shape[0] - len(test_idx)
            
            for i in test_idx:
                ori_aff = origins[i]
                try:
                    # scores = {}
                    scores = get_ela_scores(ori_aff, K=100)
                except elasticsearch.exceptions.RequestError:
                    scores = {}
                
                # normalize scores
                scores_sum = 0
                ks = list(scores.keys())
                for k in ks:
                    if k not in all_labels:
                        del scores[k]
                    else:
                        scores_sum += scores[k]
                for k in scores.keys():
                    scores[k] = scores[k] / scores_sum

                log_max = torch.max(mix_logits[i], 0)[0]
                if len(scores.keys()) > 0:
                    for j in range(int(mix_logits.shape[1])):
                        if all_labels[j] in scores.keys():
                            mix_logits[current_ind][j] += log_max * scores[all_labels[j]]
                            es_logits[current_ind][j] *= scores[all_labels[j]]
                        else:
                            mix_logits[current_ind][j] = 0
                            es_logits[current_ind][j] = 0
                

                if torch.max(labels[current_ind], 0)[1] == torch.max(es_logits[current_ind], 0)[1]:
                    correct_es += 1

                if torch.max(labels[current_ind], 0)[1] == torch.max(mix_logits[current_ind], 0)[1]:
                    correct_mix += 1
                
                if torch.max(labels[current_ind], 0)[1] == torch.max(logits[current_ind], 0)[1]:
                    correct_gcn += 1

                count += 1
                current_ind += 1

                print(is_test, current_ind, logits.shape[0], correct_es, correct_gcn, correct_mix, count)

            # f = open('model_performance_res.txt', 'a')
            # f.write(str(len(test_idx)) + '\t' + str(correct_es) + '\t' + str(correct_gcn) + '\t' + str(correct_mix) + '\t' + \
            #     str(correct_es / count) + '\t' + str(correct_gcn / count) + '\t' + str(correct_mix / count) + '\n')
            # f.close()
            es_pred = torch.max(es_logits, 1)[1].numpy()
            mix_pred = torch.max(mix_logits, 1)[1].numpy()
                
        t_mask = torch.from_numpy(np.array(mask*1., dtype=np.float32))
        tm_mask = torch.transpose(torch.unsqueeze(t_mask, 0), 1, 0).repeat(1, labels.shape[1])
        loss = criterion(logits * tm_mask, torch.max(labels, 1)[1])
        pred = torch.max(logits, 1)[1]
        acc = ((pred == torch.max(labels, 1)[1]).float() * t_mask).sum().item() / t_mask.sum().item()
        
    return loss.numpy(), acc, pred.numpy(), labels.numpy(), (time.time() - t_test), [pred.numpy(), es_pred, mix_pred]

res_dict = dict()

for ct in range(100):
    try:

        # datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
        dataset = 'name_{}'.format(ct)

        # if dataset not in datasets:
        # 	sys.exit("wrong dataset name")
        cfg.dataset = dataset

        # Set random seed
        seed = random.randint(1, 200)
        seed = 2019
        np.random.seed(seed)
        torch.manual_seed(seed)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed(seed)


        # Settings
        # os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Load data
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size, test_idx = load_corpus(
            cfg.dataset)

        if len(test_idx) == 0:
            # f = open('model_performance_res.txt', 'a')
            # f.write('\n')
            # f.close()
            continue

        f = open('dataset/{}.json'.format(dataset), encoding='utf-8')
        datas = json.loads(f.read())
        origins = datas['origins']
        labels_name = datas['labels']

        all_labels = []

        for line in open("./data/corpus/{}_labels.txt".format(dataset)):
            all_labels.append(line.strip())

        all_labels = [x.lower() for x in all_labels]

        features = sp.identity(features.shape[0])  # featureless


        # Some preprocessing
        features = preprocess_features(features)
        if cfg.model == 'gcn':
            support = [preprocess_adj(adj)]
            num_supports = 1
            model_func = GCN
        elif cfg.model == 'gcn_cheby':
            support = chebyshev_polynomials(adj, cfg.max_degree)
            num_supports = 1 + cfg.max_degree
            model_func = GCN
        elif cfg.model == 'dense':
            support = [preprocess_adj(adj)]  # Not used
            num_supports = 1
            model_func = MLP
        else:
            raise ValueError('Invalid argument for model: ' + str(cfg.model))


        # Define placeholders
        t_features = torch.from_numpy(features)
        t_y_train = torch.from_numpy(y_train)
        t_y_val = torch.from_numpy(y_val)
        t_y_test = torch.from_numpy(y_test)
        t_train_mask = torch.from_numpy(train_mask.astype(np.float32))
        tm_train_mask = torch.transpose(torch.unsqueeze(t_train_mask, 0), 1, 0).repeat(1, y_train.shape[1])

        t_support = []
        for i in range(len(support)):
            t_support.append(torch.Tensor(support[i]))
                
        model = model_func(input_dim=features.shape[0], support=t_support, num_classes=y_train.shape[1])


        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)


        # Define model evaluation function
        val_losses = []

        # Train model
        for epoch in range(cfg.epochs):

            t = time.time()
            
            # Forward pass
            logits = model(t_features)
            loss = criterion(logits * tm_train_mask, torch.max(t_y_train, 1)[1])    
            acc = ((torch.max(logits, 1)[1] == torch.max(t_y_train, 1)[1]).float() * t_train_mask).sum().item() / t_train_mask.sum().item()
                
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation
            val_loss, val_acc, pred, labels, duration, _ = evaluate(t_features, t_y_val, val_mask, -1)
            val_losses.append(val_loss)

            # print_log("Epoch: {:.0f}, train_loss= {:.5f}, train_acc= {:.5f}, val_loss= {:.5f}, val_acc= {:.5f}, time= {:.5f}"\
            #             .format(epoch + 1, loss, acc, val_loss, val_acc, time.time() - t))

            if epoch > cfg.early_stopping and val_losses[-1] > np.mean(val_losses[-(cfg.early_stopping+1):-1]):
                print_log("Early stopping...")
                break


        print_log("Optimization Finished!")


        # Testing
        test_loss, test_acc, pred, labels, test_duration, pred_list = evaluate(t_features, t_y_test, test_mask, ct)
        print_log("Test set results: \n\t loss= {:.5f}, accuracy= {:.5f}, time= {:.5f}".format(test_loss, test_acc, test_duration))

        gcn_test_pred = []
        es_test_pred = []
        mix_test_pred = []
        test_norm_affs = []

        gcn_pred = pred_list[0]
        es_pred = pred_list[1]
        mix_pred = pred_list[2]

        current_ind = logits.shape[0] - len(test_idx)
        for i in test_idx:
            test_norm_affs.append(labels_name[i].lower())
            gcn_test_pred.append(all_labels[gcn_pred[current_ind]])
            es_test_pred.append(all_labels[es_pred[current_ind]])
            mix_test_pred.append(all_labels[mix_pred[current_ind]])
            current_ind += 1
        
        for i in range(len(test_norm_affs)):
            norm = test_norm_affs[i]
            gcn_pred_norm = gcn_test_pred[i]
            es_pred_norm = es_test_pred[i]
            mix_pred_norm = mix_test_pred[i]
            for j in [norm, gcn_pred_norm, es_pred_norm, mix_pred_norm]:
                if j not in res_dict.keys():
                    res_dict[j] = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] # gcn, es, mix - [TP, FP, TN, FN]
            
            model_index = 0
            for j in [gcn_pred_norm, es_pred_norm, mix_pred_norm]:
                if norm == j:
                    res_dict[norm][model_index][0] += 1
                else:
                    res_dict[norm][model_index][3] += 1
                    res_dict[j][model_index][1] += 1
                model_index += 1
    except:
        pass
        # f = open('model_performance_res.txt', 'a')
        # f.write('\n')
        # f.close()

final_res_dict = dict()

for norm in res_dict.keys():
    dl = res_dict[norm]
    final_res_dict[norm] = []
    for model_index in range(3):
        if dl[model_index][0] == 0:
            preci = 0
            recall = 0
            f1 = 0
        else:
            preci = dl[model_index][0] / (dl[model_index][0] + dl[model_index][1])
            recall = dl[model_index][0] / (dl[model_index][0] + dl[model_index][3])
            f1 = 2 * preci * recall / (preci + recall)
        final_res_dict[norm].append(preci)
        final_res_dict[norm].append(recall)
        final_res_dict[norm].append(f1)
    final_res_dict[norm].append(dl[0][0] + dl[0][3])
    with open('aff_norm_res_table.txt', 'a') as f:
        if len(norm) >= 30:
            w_norm = norm[0: 30]
        else:
            w_norm = norm + ''.join([' ' for i in range(30 - len(norm))])
        f.write(w_norm)
        for i in final_res_dict[norm]:
            f.write('\t')
            f.write(str(round(i, 3)))
        f.write('\n')


json_str = json.dumps(final_res_dict)
with open('aff_norm_res_dict.json', 'w') as json_file:
    json_file.write(json_str)

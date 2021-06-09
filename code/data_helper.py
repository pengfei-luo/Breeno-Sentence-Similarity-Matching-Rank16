import json
import torch
import logging
import pandas as pd
import numpy as np
import networkx as nx
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit

def preprocess(args):
    wvd_path = './word2vec/wvd_{}_{}_{}.json'.format(args.wv_name, args.wv_size, args.wv_window)
    wvw_path = './word2vec/wvw_{}_{}_{}.npy'.format(args.wv_name, args.wv_size, args.wv_window)
    train_path = 'data/gaiic_track3_round1_train_20210228.tsv'
    test_path = 'data/gaiic_track3_round1_testA_20210228.tsv'

    wvw = np.load(wvw_path)
    def padding_seq(seq):
        if len(seq) < args.padding_length:
            seq = seq + [0] * (args.padding_length - len(seq))
        else:
            seq = seq[:args.padding_length]
        return seq

    with open(wvd_path, 'r', encoding='utf-8') as f:
        wvd = json.loads(f.readline())

    train_set_query = []
    train_set_label = []
    with open(train_path, 'r', encoding='utf-8') as in_f:
        for line in in_f:
            line = line.strip().split('\t')
            query = []
            for sentence in line[:2]:
                sentence = [wvd[word] for word in sentence.split(' ')]
                sentence = padding_seq(sentence)
                query.append(sentence)
            train_set_query.append(query)
            train_set_label.append(int(line[2]))

    test_query = []
    with open(test_path, 'r', encoding='utf-8') as in_f:
        for line in in_f:
            line = line.strip().split('\t')
            query = []
            for sentence in line:
                sentence = [wvd[word] for word in sentence.split(' ')]
                sentence = padding_seq(sentence)
                query.append(sentence)
            test_query.append(query)

    return np.array(train_set_query, dtype=np.int32), np.array(train_set_label, np.int32), np.array(test_query, np.int32), wvw

def load_data(corpus_dir):
    train_path = '{}/gaiic_track3_round1_train_20210228.tsv'.format(corpus_dir)
    test_path = '{}/gaiic_track3_round1_testA_20210228.tsv'.format(corpus_dir)
    testb_path = '{}/gaiic_track3_round1_testB_20210317.tsv'.format(corpus_dir)
    train_round2_path = '{}/gaiic_track3_round2_train_20210407.tsv'.format(corpus_dir)
    def process(l, pad_len):
        return l
        # l = list(map(int, l.split(' ')))[:pad_len]
        # return l
        # return  ''.join(map(lambda x : str(x) + ' ', l)).strip()

    train_query_1 = []
    train_query_2 = []
    train_label = []
    with open(train_path, 'r', encoding='utf-8') as in_f:
        for line in in_f:
            line = line.strip().split('\t')
            train_query_1.append(line[0].strip())
            train_query_2.append(line[1].strip())
            train_label.append(int(line[2]))

    test_query_1 = []
    test_query_2 = []
    with open(test_path, 'r', encoding='utf-8') as in_f:
        for line in in_f:
            line = line.strip().split('\t')
            test_query_1.append(line[0].strip())
            test_query_2.append(line[1].strip(),)


    testb_query_1 = []
    testb_query_2 = []
    with open(testb_path, 'r', encoding='utf-8') as in_f:
        for line in in_f:
            line = line.strip().split('\t')
            testb_query_1.append(line[0].strip())
            testb_query_2.append(line[1].strip())

    train_round2_query_1 = []
    train_round2_query_2 = []
    train_round2_label = []
    with open(train_round2_path, 'r', encoding='utf-8') as in_f:
        for line in in_f:
            line = line.strip().split('\t')
            train_round2_query_1.append(line[0].strip())
            train_round2_query_2.append(line[1].strip())
            train_round2_label.append(int(line[2]))


    return [train_query_1, train_query_2], train_label, [test_query_1, test_query_2], [testb_query_1, testb_query_2], \
           [train_round2_query_1, train_round2_query_2], train_round2_label


def load_enhanced_data(pad_len):
    data_path = './data/enhanced_training_set.tsv'
    def process(l, pad_len):
        l = list(map(int, l.split(' ')))[:pad_len]
        # return l
        return  ''.join(map(lambda x : str(x) + ' ', l)).strip()


    train_query_1 = []
    train_query_2 = []
    train_label = []
    with open(data_path, 'r', encoding='utf-8') as in_f:
        for line in in_f:
            line = line.strip().split('\t')
            train_query_1.append(process(line[0].strip(), pad_len))
            train_query_2.append(process(line[1].strip(), pad_len))
            train_label.append(int(line[2]))

    return [train_query_1, train_query_2], train_label


def load_pseudo(file_path):
    test_q1 = []
    test_q2 = []
    test_l = []
    with open(file_path, 'r', encoding='utf-8') as in_f :
        for line in in_f :
            line = line.strip().split('\t')
            test_q1.append(line[0].strip())
            test_q2.append(line[1].strip())
            test_l.append(int(line[2]))

    return [test_q1, test_q2], test_l

def train_dev_split(train_set_query, train_set_label):
    data = pd.DataFrame(train_set_query).T
    label = np.array(train_set_label)
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=0)
    for train_idx, dev_idx in ss.split(data, label):
        train_query_1 = data.iloc[train_idx][0].values.tolist()
        train_query_2 = data.iloc[train_idx][1].values.tolist()
        train_label = label[train_idx]
        dev_query_1 = data.iloc[dev_idx][0].values.tolist()
        dev_query_2 = data.iloc[dev_idx][1].values.tolist()
        dev_label = label[dev_idx]

    # train_query = [(train_query_1[i], train_query_2[i]) for i in range(len(train_query_1))]
    # dev_query = [(dev_query_1[i], dev_query_2[i]) for i in range(len(dev_query_1))]
    # return  train_query, train_label, dev_query, dev_label
    return [train_query_1, train_query_2], train_label, [dev_query_1, dev_query_2], dev_label

def make_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

    chlr = logging.StreamHandler()
    chlr.setLevel(logging.DEBUG)
    chlr.setFormatter(formatter)

    fhlr = logging.FileHandler(log_path, mode='w')
    fhlr.setLevel(logging.DEBUG)
    fhlr.setFormatter(formatter)

    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger

class TCPretrainData(Dataset):
    def __init__(self, encodings):
        super(TCPretrainData, self).__init__()
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


class TCData(Dataset):
    def __init__(self, encodings, labels):
        super(TCData, self).__init__()
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



class DatasetIter(Dataset):
    def __init__(self, q1, q2, label):
        super(DatasetIter, self).__init__()
        self.q1 = q1
        self.q2 = q2
        self.label = label

    def __getitem__(self, item):
        return self.q1[item, :], self.q2[item, :], self.label[item]

    def __len__(self):
        return self.q1.shape[0]

class SubmitIter(Dataset):
    def __init__(self, q1, q2):
        super(SubmitIter, self).__init__()
        self.q1 = q1
        self.q2 = q2

    def __getitem__(self, item):
        return self.q1[item, :], self.q1[item, :]

    def __len__(self):
        return self.q1.shape[0]



def training_data_enhance():
    train_path = 'data/gaiic_track3_round1_train_20210228.tsv'

    train_query_1 = []
    train_query_2 = []
    train_label = []
    with open(train_path, 'r', encoding='utf-8') as in_f :
        for line in in_f :
            line = line.strip().split('\t')
            train_query_1.append(line[0].strip())
            train_query_2.append(line[1].strip())
            train_label.append(int(line[2]))

    out_f = open('./data/enhanced_training_set.tsv', 'w', encoding='utf-8')

    index = 0
    query2id = dict()
    id2query = dict()
    g_zero = nx.DiGraph()
    g_one = nx.DiGraph()
    for q in train_query_1:
        if q not in query2id.keys():
            query2id[q] = index
            id2query[index] = q
            g_zero.add_node(index)
            g_one.add_node(index)
            index += 1
    for q in train_query_2:
        if q not in query2id.keys() :
            query2id[q] = index
            id2query[index] = q
            g_zero.add_node(index)
            g_one.add_node(index)
            index += 1

    pair = [set(), set()]
    with open(train_path, 'r', encoding='utf-8') as in_f :
        for line in in_f :
            line = line.strip().split('\t')
            node1 = query2id[line[0].strip()]
            node2 = query2id[line[1].strip()]
            if node1 < node2:
                pair[int(line[2])].add((node1, node2))
            else:
                pair[int(line[2])].add((node2, node1))

    print(len(pair[1]))

    # enhance false label query pair
    # for u, v in pair[0]:
    #     g_zero.add_edge(u, v)
    #     g_zero.add_edge(v, u)
    #
    # ret = nx.algorithms.transitive_closure(g_zero)
    # all_closure = set()
    # for node_id, visitable in ret.adj.items():
    #     closure_nodes = tuple(sorted(visitable.keys()))
    #     if len(closure_nodes) == 2:
    #         continue
    #     all_closure.add(closure_nodes)
    #
    # for closure in all_closure:
    #     for first_idx in range(0, len(closure) - 1):
    #         for second_idx in range(1, len(closure)):
    #             if (first_idx, second_idx) not in pair[0]:
    #                 out_f.write(id2query[first_idx] + '\t' + id2query[second_idx] + '\t' + '0')

    # enhance true label query pair
    for u, v in pair[1] :
        g_one.add_edge(u, v)
        g_one.add_edge(v, u)

    ret = nx.algorithms.transitive_closure(g_one)
    all_closure = set()
    for node_id, visitable in ret.adj.items() :
        closure_nodes = tuple(sorted(visitable.keys()))
        if len(closure_nodes) != 3 :
            continue
        all_closure.add(closure_nodes)
    print(len(all_closure))

    cnt = 0
    for closure in all_closure :
        for first_idx in range(0, len(closure) - 1) :
            for second_idx in range(first_idx+1, len(closure)) :
                if (first_idx, second_idx) not in pair[0] :
                    out_f.write(id2query[closure[first_idx]] + '\t' + id2query[closure[second_idx]] + '\t' + '1\n')
                    cnt += 1

    # enhance false label query pair
    for u, v in pair[0]:
        g_zero.add_edge(u, v)
        g_zero.add_edge(v, u)

    ret = nx.algorithms.transitive_closure(g_zero)
    all_closure = set()
    for node_id, visitable in ret.adj.items():
        closure_nodes = tuple(sorted(visitable.keys()))
        if len(closure_nodes) != 3:
            continue
        all_closure.add(closure_nodes)

    for closure in all_closure:
        for first_idx in range(0, len(closure) - 1):
            for second_idx in range(first_idx+1, len(closure)):
                if (first_idx, second_idx) not in pair[0]:
                    out_f.write(id2query[closure[first_idx]] + '\t' + id2query[closure[second_idx]] + '\t' + '0\n')
                    cnt -= 1
                    if cnt == 0:
                        return





if __name__ == '__main__':
    import time
    import utils
    vocab_size = utils.build_vocab('../user_data/tsv_data', '../user_data/vocab_data')
    t1 = time.time()
    train_query, train_label, test_query, testb_query, train_r2_query = load_data(0)
    enhanced_query = [train_query[0], train_query[1]]
    enhanced_query[0].extend(test_query[0])
    enhanced_query[1].extend(test_query[1])
    enhanced_query[0].extend(train_query[1])
    enhanced_query[1].extend(train_query[0])
    enhanced_query[0].extend(test_query[1])
    enhanced_query[1].extend(test_query[0])
    enhanced_query[0].extend(testb_query[0])
    enhanced_query[1].extend(testb_query[1])
    enhanced_query[1].extend(testb_query[0])
    enhanced_query[0].extend(testb_query[1])
    enhanced_query[0].extend(train_r2_query[0])
    enhanced_query[1].extend(train_r2_query[1])
    enhanced_query[0].extend(train_r2_query[1])
    enhanced_query[1].extend(train_r2_query[0])
    t2 = time.time()
    print(t2-t1)
    from transformers.models.bert import BertTokenizerFast
    tokenizer = BertTokenizerFast("../user_data/vocab_data/vocab.txt")
    train_encodings = tokenizer(enhanced_query[0], enhanced_query[1],
                                truncation=True,
                                padding='max_length',
                                max_length=18)
    t2 = time.time()
    print(t2-t1)
    # train_set_query, train_set_label, test_query = preprocess()
    # train_query, train_label, dev_query, dev_label = train_dev_split(train_set_query, train_set_label)
    # print(train_query.shape)
    # print(train_label.shape)
    # print(dev_query.shape)
    # print(dev_label.shape)
    # training_data_enhance()

    # train_q1 = train_query[:, 0, :]
    # train_q2 = train_query[:, 1, :]
    # dev_q1 = dev_query[:, 0, :]
    # dev_q2 = dev_query[:, 1, :]

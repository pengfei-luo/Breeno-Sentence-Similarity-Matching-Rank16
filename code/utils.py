import torch
import argparse
import transformers
import modeling_nezha
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss

from collections import defaultdict
from tokenizers.pre_tokenizers import Whitespace
import tokenizers
import tokenizers.models
import tokenizers.trainers
import data_helper as dh


class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                              self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss

def get_layer_params(model, initial_lr, decay_ratio):
    embedding_params = []
    all_layer_params = []
    classified_params = []
    for layer_id in range(12):
        layer_params = []
        for name, param in model.bert.encoder.layer[layer_id].named_parameters():
            layer_params.append(param)
        all_layer_params.append(layer_params)
    for name, param in model.bert.embeddings.named_parameters():
        embedding_params.append(param)
    for name, param in model.bert.pooler.named_parameters():
        classified_params.append(param)
    for name, param in model.classifier.named_parameters():
        classified_params.append(param)
    for name, param in model.pooler.named_parameters():
        classified_params.append(param)

    all_layer_params = [embedding_params] +  all_layer_params + [classified_params]
    all_layer_lr = []
    for _ in range(len(all_layer_params)):
        all_layer_lr.append(initial_lr)
        initial_lr = initial_lr * decay_ratio
    all_layer_lr = all_layer_lr[::-1]

    lr_list = []
    for _ in range(len(all_layer_params)):
        lr_list.append({
            'params': all_layer_params[_],
            'lr': all_layer_lr[_],
        })
    return lr_list

def load_model_for_inference(model_path, vocab_path):
    model = modeling_nezha.NeZhaForSequenceClassification.from_pretrained(model_path)
    tokenizer = transformers.models.bert.BertTokenizer(vocab_path)
    return model, tokenizer

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def build_vocab(corpus_dir, vocab_save_dir):
    file_path = ['{}/gaiic_track3_round1_train_20210228.tsv'.format(corpus_dir),
                 '{}/gaiic_track3_round1_testA_20210228.tsv'.format(corpus_dir),
                 '{}/gaiic_track3_round1_testB_20210317.tsv'.format(corpus_dir),
                 '{}/gaiic_track3_round2_train_20210407.tsv'.format(corpus_dir)]
    token_id = 0
    token_dict = defaultdict(int)
    for path in file_path:
        with open(path, 'r', encoding='utf-8') as in_f :
            for line in in_f :
                line = line.strip().split('\t')
                for token in line[0].strip().split():
                    if token not in token_dict.keys():
                        token_dict[token] = token_id
                        token_id += 1
                for token in line[1].strip().split():
                    if token not in token_dict.keys():
                        token_dict[token] = token_id
                        token_id += 1
    # print(len(token_dict))
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(token_dict, '[UNK]'))
    special_tokens = ["[PAD]"] + ['[UNUSED{}]'.format(i) for i in range(1, 100)] +  ["[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]
    tokenizer.pre_tokenizer = Whitespace()
    trainer = tokenizers.trainers.WordLevelTrainer(vocab_size=len(token_dict), min_frequency=1, special_tokens=special_tokens)
    tokenizer.train(file_path, trainer)
    tokenizer.model.save(vocab_save_dir)

    import json
    with open("{}/vocab.json".format(vocab_save_dir), 'r', encoding='utf-8') as f:
        word_dict = json.loads(f.readline())
    word_keys = []
    digit_keys = []
    for k in word_dict.keys():
        if k[0] == '[' or k[0] == '<':
            word_keys.append(k)
        else:
            digit_keys.append(int(k))
    digit_keys = sorted(digit_keys)
    word_keys.extend(digit_keys)
    with open("{}/vocab.txt".format(vocab_save_dir), 'w', encoding='utf-8') as f:
        for k in word_keys:
            f.write("{}\n".format(k))
    return len(token_dict)

    # model = transformers.BertTokenizer.from_pretrained('../user_data/tokenizer/vocab.txt')
    # transformers.BertTokenizerFast.from_pretrained()
    # text1 = '679 29 457 449 451 16'
    # text2 = '29 1300 11'
    # ret = model(text1, text2, padding='max_length', max_length=40)
    # print(ret)

def get_statistic():
    query_length = []
    file_path = ['../user_data/tsv_data/gaiic_track3_round1_train_20210228.tsv',
                 '../user_data/tsv_data/gaiic_track3_round1_testA_20210228.tsv',
                 '../user_data/tsv_data/gaiic_track3_round1_testB_20210317.tsv',
                 '../user_data/tsv_data/gaiic_track3_round2_train_20210407.tsv']

    for path in file_path:
        with open(path, 'r', encoding='utf-8') as in_f :
            for line in in_f :
                line = line.strip().split('\t')
                query_length.append(len(line[0].strip().split()) + len(line[1].strip().split()))
                # query_length.append()
    print(np.average(query_length))
    print(np.min(query_length))
    print(np.max(query_length))
    # sns.distplot(query_length, rug=True)
    # plt.show()

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='bert.embeddings.word_embeddings.weight'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='bert.embeddings.word_embeddings.weight'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.1, emb_name='bert.embeddings.word_embeddings.weight', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='bert.embeddings.word_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


from collections import defaultdict
from itertools import chain
from torch.optim import Optimizer
import torch
import warnings


class Lookahead(Optimizer) :
    def __init__(self, optimizer, k=5, alpha=0.5) :
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups :
            group["counter"] = 0

    def update(self, group) :
        for fast in group["params"] :
            param_state = self.state[fast]
            if "slow_param" not in param_state :
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self) :
        for group in self.param_groups :
            self.update(group)

    def step(self, closure=None) :
        loss = self.optimizer.step(closure)
        for group in self.param_groups :
            if group["counter"] == 0 :
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k :
                group["counter"] = 0
        return loss

    def state_dict(self) :
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k) : v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state" : fast_state,
            "slow_state" : slow_state,
            "param_groups" : param_groups,
        }

    def load_state_dict(self, state_dict) :
        slow_state_dict = {
            "state" : state_dict["slow_state"],
            "param_groups" : state_dict["param_groups"],
        }
        fast_state_dict = {
            "state" : state_dict["fast_state"],
            "param_groups" : state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group) :
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)


def convert_to_onnx(model, pad_length, onnx_save_dir, task_name, epoch_idx):
    dummy_input_ids = torch.LongTensor(torch.randint(10, [1, pad_length * 2]))
    dummy_attention_mask = torch.LongTensor(torch.randint(2, [1, pad_length * 2]))
    dummy_token_type_ids = torch.LongTensor(torch.randint(2, [1, pad_length * 2]))
    torch.onnx.export(
        model=model,
        args=(dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
        f='{}/{}_{}.onnx'.format(onnx_save_dir, task_name, epoch_idx),
        export_params=True,
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['logits'],
        opset_version=11
    )

import onnxruntime
import numpy as np
from transformers import BertTokenizer

def load_model_for_inference(model_path, vocab_dir):
    nezha_session = onnxruntime.InferenceSession(model_path)
    tokenizer = BertTokenizer.from_pretrained(vocab_dir)
    return nezha_session, tokenizer


def get_file_list(walkdir, suffix=None):
    import os
    file_path = []
    for fdir, subdir, file_name_list in os.walk(walkdir) :
        for filename in file_name_list :
            if suffix is None:
                abs_path = os.path.join(fdir, filename)
                file_path.append((abs_path, fdir, filename))
            elif filename.endswith(suffix):
                abs_path = os.path.join(fdir, filename)
                file_path.append((abs_path, fdir, filename))
    return file_path

def output_log():
    file_path = get_file_list(walkdir="../user_data/logs")
    for path, fdir, filename in file_path:
        with open(path, 'r', encoding='utf-8') as f:
            print("START----------" + filename + "----------START")
            for line in f:
                print(line)
            print("E N D----------" + filename + "----------E N D")

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def make_pseudo_label(corpus_dir, model_from, tokenizer_save_dir, confident=0.01, pad_length=20):
    from modeling_nezha import NeZhaForSequenceClassification
    from transformers.models.bert import BertTokenizer
    from torch.utils.data import DataLoader
    import os, tqdm
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test_path = '{}/gaiic_track3_round1_testA_20210228.tsv'.format(corpus_dir)
    testb_path = '{}/gaiic_track3_round1_testB_20210317.tsv'.format(corpus_dir)
    test_query_1 = []
    test_query_2 = []
    with open(test_path, 'r', encoding='utf-8') as in_f :
        for line in in_f :
            line = line.strip().split('\t')
            test_query_1.append(line[0].strip())
            test_query_2.append(line[1].strip(), )
    with open(testb_path, 'r', encoding='utf-8') as in_f :
        for line in in_f :
            line = line.strip().split('\t')
            test_query_1.append(line[0].strip())
            test_query_2.append(line[1].strip())
    model = NeZhaForSequenceClassification.from_pretrained(model_from).to(device)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_save_dir)
    pred_encodings = tokenizer(test_query_1, test_query_2,
                               padding='max_length',
                               truncation=True,
                               max_length=pad_length * 2)
    model.eval()
    pred_data = dh.TCPretrainData(pred_encodings)
    pred_data_loader = DataLoader(pred_data, batch_size=128, shuffle=False)

    pred_val = []
    for batch in tqdm.tqdm(pred_data_loader):
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        type_ids = batch['token_type_ids'].to(device)
        logits = model(input_ids, attn_mask, type_ids)[0]
        batch_pred = torch.softmax(logits, dim=-1)[:, 1].tolist()
        pred_val.extend(batch_pred)

    # pred_val = np.round(pred_val).tolist()
    cnt = 0

    if not os.path.exists('../user_data/pseudo_data'):
        os.mkdir('../user_data/pseudo_data')
    # if not os.path.exists('../user_data/pseudo_data/test.tsv'):
    assert len(pred_val) == len(test_query_1)
    with open('../user_data/pseudo_data/test.tsv', 'w', encoding='utf-8') as f:
        for idx in range(len(pred_val)):
            if pred_val[idx] <= confident or pred_val[idx] >= 1-confident:
                f.write('{}\t{}\t{}\n'.format(test_query_1[idx],
                                              test_query_2[idx],
                                              round(pred_val[idx])))
                cnt += 1
    print(cnt)


def ratio_check():
    r1_train_query, r1_train_label, test_query, testb_query, r2_train_query, r2_train_label = dh.load_data('../user_data/tsv_data')
    cnt0, cnt1 = 0, 0
    for x in r1_train_label:
        if x == 0:
            cnt0 += 1
        else:
            cnt1 += 1
    for x in r2_train_label:
        if x == 0:
            cnt0 += 1
        else:
            cnt1 += 1

    print(cnt0, cnt1)

if __name__ == '__main__':
    # ratio_check()
    # make_pseudo_label('../user_data/tsv_data', '../user_data/model_data/finetune/round2_ft_3gram_submit/4',
    #                   '../user_data/tokenizer_data')
    pass
    # output_log()
    # onnx_model_path = '../user_data/model_data/onnx/model.onnx'
    # vocab_dir = '../user_data/model_data/pretrain/round2_pt'
    # pad_length = 18
    # model, tokenizer = load_model_for_inference(onnx_model_path, vocab_dir)

    # pass
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model, tokenizer = load_model_for_inference('../user_data/model_data', '../user_data/vocab_data/roberta-vocab.txt')


    # model.eval()
    # text1 ='679 29 457 449 451 16'
    # text2 ='29 1300 11'
    # ret = tokenizer(text1, text2, padding='max_length', max_length=40)
    # ret = model(input_ids=torch.tensor([ret['input_ids']]).to(device),
    #             attention_mask=torch.tensor([ret['attention_mask']]).to(device))
    # ret = torch.nn.functional.softmax(ret[0], dim=-1)[0][1].cpu().item()
    # print(ret)



    # model = onnx.load('model.onnx')
    # onnx.checker.check_model(model)
    # onnx.helper.printable_graph(model.graph)

    # print(ret)
    # import onnxruntime
    # import numpy as np
    # ort_session = onnxruntime.InferenceSession("../user_data/onnx_data/model.onnx")
    # input = {
    #     'input_ids': np.array([ret['input_ids']]).astype(np.int64),
    #     'input_attn_mask': np.array([ret['attention_mask']]).astype(np.int64),
    #     'input_type_ids': np.array([ret['token_type_ids']]).astype(np.int64)
    # }
    # print(input)
    # # print(ort_session.get_inputs()[0].name)
    # # ort_inputs = {ort_session.get_inputs()[0].name : to_numpy(x)}
    # ort_outs = ort_session.run(None, input)
    #
    # print(ort_outs)

    pass
import os
import copy
import utils
import logging
import torch
import torch.optim
import transformers
import random
from prettytable import PrettyTable
from tqdm import tqdm
from torch.utils.data import DataLoader
from modeling_nezha import NeZhaForSequenceClassification, VanillaNeZhaForSequenceClassification, DoubleHeadNeZhaForSequenceClassification
from configuration_nezha import NeZhaConfig
from transformers import Trainer, TrainingArguments, AdamW, PretrainedConfig
from modeling_bert import BertForSequenceClassification
import data_helper as dh
import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers.models.bert import BertTokenizer
from tokenizers.pre_tokenizers import Whitespace
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from transformers.convert_graph_to_onnx import convert
from collections import deque

# def fine_tune(args, logger):

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train():
    torch.set_num_threads(6)
    logger.info("Tokenizing data")
    train_encodings = tokenizer(train_query[0], train_query[1],
                                padding='max_length',
                                truncation=True,
                                max_length=args.pad_length * 2)
    train_data = dh.TCData(train_encodings, train_label)
    train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    if args.test_ratio != 0.:
        dev_encodings = tokenizer(dev_query[0], dev_query[1],
                                  padding='max_length',
                                  truncation=True,
                                  max_length=args.pad_length * 2)
        dev_data = dh.TCData(dev_encodings, dev_label)
        dev_data_loader = DataLoader(dev_data, batch_size=128, shuffle=True, pin_memory=False)


    logger.info("Creating model")
    config = NeZhaConfig.from_pretrained(args.model_from)
    config.output_hidden_states = True
    config.output_attentions = True
    config.num_labels = 2
    config.multi_dropout = args.multi_dropout

    # !!! multi-gpu
    # torch.distributed.init_process_group(backend='nccl')
    model = model_class.from_pretrained(args.model_from, config=config).to(device)
    # model = torch.nn.DataParallel(model).cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    # lr_list = utils.get_layer_params(model, args.init_lr, 0.97)
    model.train()
    optim = AdamW(model.parameters(), lr=args.init_lr)
    if args.lookahead:
        optim = utils.Lookahead(optim)

    best_auc = 0.0
    best_loss = 1e10
    current_step = 0
    total_step = args.n_epoch * len(train_data_loader)
    K = args.K_PGD
    if args.adversarial == 'FGM' :
        adv = utils.FGM(model)
    elif args.adversarial == 'PGD' :
        adv = utils.PGD(model)

    scheduler = transformers.get_linear_schedule_with_warmup(optim, 0, total_step)
    loss_deque = deque([], args.eval_step)

    logger.info("Start training")
    for epoch in range(args.n_epoch) :
        logging.info("Epoch {} / {}".format(epoch + 1, args.n_epoch))
        for batch in tqdm(train_data_loader) :
            # look_ahead.zero_grad()
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            # input_ids = batch['input_ids']
            # attn_mask = batch['attention_mask']
            # type_ids = batch['token_type_ids']
            # labels = batch['labels']
            outputs = model(input_ids, attn_mask, type_ids, labels=labels)
            # outputs = model(input_ids, attn_mask, labels=labels)
            loss = outputs[0]
            loss_deque.append(loss.item())
            loss.backward()
            if args.adversarial == 'FGM' :
                adv.attack(epsilon=args.eps)
                outputs = model(input_ids, attn_mask, type_ids, labels=labels)
                # outputs = model(input_ids, attn_mask, labels=labels)
                loss = outputs[0]
                loss.backward()
                adv.restore()
            elif args.adversarial == 'PGD' :
                adv.backup_grad()
                for t in range(K) :
                    adv.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                    if t != K - 1 :
                        model.zero_grad()
                    else :
                        adv.restore_grad()
                    outputs = model(input_ids, attn_mask, labels=labels)
                    loss = outputs[0]
                    loss.backward()
                adv.restore()
            optim.step()
            scheduler.step()
            # look_ahead.step()

            current_step += 1
            if args.test_ratio != 0 :
                if current_step % args.eval_step == 0 or current_step == total_step :
                    model.eval()
                    eval_result = evaluate(dev_data_loader, model, device)
                    logger.info(eval_result)
                    if args.metric == 'auc' and eval_result['auc'] > best_auc :
                        model.save_pretrained('../user_data/model_data/finetune/{}'.format(args.task_name))
                        best_auc = eval_result['auc']
                    elif args.metric == 'auc_loss' and eval_result['auc'] > best_auc and eval_result[
                        'loss'] < best_loss :
                        model.save_pretrained('../user_data/model_data/finetune/{}'.format(args.task_name))
                        best_auc = eval_result['auc']
                        best_loss = eval_result['loss']
                    model.train()
            if current_step % args.eval_step == 0:
                logger.info("loss {}".format(np.mean(loss_deque).item()))

        if args.test_ratio == 0 and epoch + 1 >= args.n_epoch - args.n_ckpt_save + 1 :
            model.save_pretrained('../user_data/model_data/finetune/{}/{}'.format(args.task_name, epoch + 1))






def evaluate(data_loader, model, device):
    def compute_metrics(preds, labels, loss):
        precision, recall, f1, _ = precision_recall_fscore_support(labels, np.round(preds).tolist(), average='binary')
        acc = accuracy_score(labels, np.round(preds).tolist())
        auc = roc_auc_score(labels, preds)
        return {
            'loss' : np.average(loss),
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc':  auc,
        }

    accu_loss = []
    accu_pred = []
    accu_label = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attn_mask, type_ids, labels=labels)
            loss = outputs[0]
            pred = F.softmax(outputs[1], dim=-1)
            accu_loss.append(loss.item())
            accu_pred.extend(pred[:, 1].tolist())
            accu_label.extend(labels.tolist())

    return compute_metrics(accu_pred, accu_label, accu_loss)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--k_fold', type=int, default=1)
    parser.add_argument('--n_epoch', type=int, default=5)
    parser.add_argument('--task_name', type=str, default='roberta_finetune_with_FGM_10FOLD')
    parser.add_argument('--model_from', type=str, default='./model/roberta-wwm_pretrain-c')
    parser.add_argument('--tokenizer_save_dir', type=str)
    parser.add_argument('--corpus_dir', type=str)
    parser.add_argument('--enhanced_data', type=int)
    parser.add_argument('--init_lr', type=float, default=1e-5)
    parser.add_argument('--eval_step', type=int, default=200)
    parser.add_argument('--lookahead', type=int, default=0)
    parser.add_argument('--adversarial', type=str, default='FGM')
    parser.add_argument('--eps', type=float, default=0.5)
    parser.add_argument('--K_PGD', type=int, default=3)
    parser.add_argument('--metric', type=str, default='auc')
    parser.add_argument('--pad_length', type=int, default=18)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--test_ratio', type=float)
    parser.add_argument('--n_ckpt_save', type=int, default=3)
    parser.add_argument('--multi_dropout', type=int)
    parser.add_argument('--convert_to_onnx', type=int)
    parser.add_argument('--onnx_save_dir', type=str)
    parser.add_argument('--params_verbose', type=int, default=0)
    args = parser.parse_args()

    dir_check = ['../user_data/logs',
                '../user_data/model_data/pretrain',
                '../user_data/model_data/finetune',
                '../user_data/model_data/onnx',
                '../user_data/vocab_data',
                args.onnx_save_dir,
                ]

    for dir in dir_check :
        if not os.path.exists(dir) :
            os.mkdir(dir)

    logger = dh.make_logger('../user_data/logs/{}.log'.format(args.task_name))

    if args.params_verbose == 1:
        pt = PrettyTable()
        pt.field_names = ['arg', 'value']
        for k, v in vars(args).items():
            pt.add_row([k, v])
        logger.info('\n' + str(pt))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info("available device: {}".format(torch.cuda.device_count()))
    logger.info("Initializing tokenizer.")
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_save_dir)

    # if args.model_from[-1] == '0' or args.model_from[-1] == '1' or args.model_from[-1] == '2' or args.model_from[-1] == '3':
    model_class = VanillaNeZhaForSequenceClassification
    logger.info('model is nezha')
    # elif 'bert' in args.model_from or args.model_from[-1] == '2' or args.model_from[-1] == '3':
    #     model_class = BertForSequenceClassification
    #     logger.info('model is bert')

    logger.info("Loading original data.")
    r1_train_query, r1_train_label, test_query, testb_query, r2_train_query, r2_train_label = dh.load_data(
        args.corpus_dir)
    enhanced_query = [r1_train_query[0], r1_train_query[1]]
    enhanced_label = r1_train_label
    enhanced_query[0].extend(r2_train_query[0])
    enhanced_query[1].extend(r2_train_query[1])
    enhanced_label.extend(r2_train_label)

    # if os.path.exists('../user_data/pseudo_data/test.tsv'):
    #     logger.info("Pseudo data detected! Loading!")
    #     test_q, test_l = dh.load_pseudo('../user_data/pseudo_data/test.tsv')
    #     enhanced_query[0].extend(test_q[0])
    #     enhanced_query[1].extend(test_q[1])
    #     enhanced_label.extend(test_l)

    if args.enhanced_data == 1 :
        enhanced_q, enhanced_l = dh.load_enhanced_data(args.pad_length)
        enhanced_query[0] += enhanced_q[0]
        enhanced_query[1] += enhanced_q[1]
        enhanced_label += enhanced_l

    set_seed(args.seed)
    if args.test_ratio != 0.:
        ss = StratifiedShuffleSplit(args.k_fold, test_size=args.test_ratio, train_size=1 - args.test_ratio,
                                    random_state=args.seed)

    if args.test_ratio != 0:
        data = pd.DataFrame(enhanced_query).T
        label = np.array(enhanced_label)
        for fold, (train_idx, dev_idx) in enumerate(ss.split(data, label)) :
            fold += 1
            logger.info("Now {} / {} fold will be training. Please wait ...".format(fold, args.k_fold))
            logger.info("Splitting data")
            train_query = [data.iloc[train_idx][0].values.tolist(), data.iloc[train_idx][1].values.tolist()]
            dev_query = [data.iloc[dev_idx][0].values.tolist(), data.iloc[dev_idx][1].values.tolist()]
            train_label = label[train_idx].tolist()
            dev_label = label[dev_idx].tolist()
            train()
    else:
        train_query = enhanced_query
        train_label = enhanced_label
        train()

    if args.convert_to_onnx == 1:
        logger.info("Converting the model into ONNX")
        file_list = utils.get_file_list('../user_data/model_data/finetune/{}'.format(args.task_name), 'bin')
        for file_path, fdir, file_name in file_list:
            model = model_class.from_pretrained(fdir)
            model.eval()
            utils.convert_to_onnx(model, args.pad_length, args.onnx_save_dir, args.task_name, os.path.split(fdir)[-1])

        logger.info("All Done! TASK [{}] convert {} models".format(args.task_name, len(file_list)))
    else:
        logger.info("All Done!")



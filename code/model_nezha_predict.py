import os
import copy
import logging
import torch
from prettytable import PrettyTable
from tqdm import tqdm
from torch.utils.data import DataLoader
from modeling_nezha import NeZhaForSequenceClassification
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, AdamW
import data_helper as dh
import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers.models.bert import BertTokenizer
from tokenizers.pre_tokenizers import Whitespace
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

def fine_tune(args, logger):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info("available device: {}".format(torch.cuda.device_count()))
    logger.info("Initializing tokenizer.")
    tokenizer = BertTokenizer(args.vocab_path, return_attention_mask=True)
    tokenizer.pre_tokenizer = Whitespace()

    logger.info("Loading original data.")
    train_query, train_label, test_query, testb_query = dh.load_data(pad_len=args.pad_length)

    logger.info("Making dataset")
    train_encodings = tokenizer(testb_query[0], testb_query[1],
                                padding='max_length',
                                truncation=True,
                                max_length=args.pad_length * 2)
    test_data = dh.TCPretrainData(train_encodings)
    final_result = np.zeros((len(test_data), 2))

    for fold in range(1, args.k_fold + 1):
        logger.info("Now {} / {} fold will predict. Please wait ...".format(fold, args.k_fold))
        logger.info("Creating model")
        eval_args = TrainingArguments(
            output_dir='../user_data/temp_data/{}'.format(args.task_name),
            per_device_eval_batch_size=128,  # batch size for evaluation
        )
        model = NeZhaForSequenceClassification.from_pretrained(args.pred_model_from + "/{}/".format(fold), num_labels=2).to(device)
        model.eval()

        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=eval_args,
        )

        pred = trainer.predict(test_dataset=test_data)
        ret = F.softmax(torch.Tensor(pred.predictions), -1)
        final_result += ret.numpy()

    final_result /= args.k_fold
    logger.info("../user_data/temp_data/predict/{}.csv".format(args.task_name))
    np.savetxt('../user_data/temp_data/predict/{}.csv'.format(args.task_name), final_result[:, 1])
    logger.info("Done!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--n_epoch', type=int, default=5)
    parser.add_argument('--k_fold', type=int, default=5)
    parser.add_argument('--task_name', type=str, default='roberta_predict_with_FGM_5FOLD')
    # parser.add_argument('--model_from', type=str, default='./model/roberta-wwm_pretrain-c')
    parser.add_argument('--pred_model_from', type=str, default='./model/roberta_finetune_with_FGM_5FOLD')
    # parser.add_argument('--enhanced_data', type=bool, default=False)
    # parser.add_argument('--init_lr', type=float, default=3e-5)
    parser.add_argument('--pad_length', type=int, default=20)
    parser.add_argument('--vocab_path', type=str)
    args = parser.parse_args()

    if not os.path.exists("../user_data/temp_data/") :
        os.mkdir("../user_data/temp_data/")
    if not os.path.exists('../user_data/temp_data/logs') :
        os.mkdir('../user_data/temp_data/logs')
    if not os.path.exists('../user_data/temp_data/predict') :
        os.mkdir('../user_data/temp_data/predict')

    logger = dh.make_logger('../user_data/temp_data/logs/{}.log'.format(args.task_name))

    pt = PrettyTable()
    pt.field_names = ['arg', 'value']
    for k, v in vars(args).items() :
        pt.add_row([k, v])
    logger.info(pt)

    fine_tune(args=args, logger=logger)


import os
import torch
import transformers
from prettytable import PrettyTable
from configuration_nezha import NeZhaConfig
from modeling_nezha import NeZhaForMaskedLM
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, TextDataset, DataCollatorForWholeWordMask
from transformers import BertConfig, BertForMaskedLM
from transformers.models.bert import BertTokenizer, BertForMaskedLM, BertModel
from tokenizers.pre_tokenizers import Whitespace
from ngram import DataCollatorForLanguageModeling as Ngram
import data_helper as dh
import utils
import os
from ngram_albert import DataCollatorForWholeWordMask as Ngram_albert
import numpy as np
import random
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pretrain(args, logger):
    torch.set_num_threads(6)
    if not os.path.exists('../user_data/model_data/pretrain/{}'.format(args.task_name)):
        os.mkdir('../user_data/model_data/pretrain/{}'.format(args.task_name))

    logger.info('available device: {}'.format(torch.cuda.device_count()))
    logger.info("initializing tokenizer")
    if not os.path.exists("../user_data/tokenizer_data/vocab.txt"):
        logger.info("Generating new tokenizer")
        vocab_size = utils.build_vocab(args.corpus_dir, args.vocab_save_dir)
        tokenizer = BertTokenizer("{}/vocab.txt".format(args.vocab_save_dir))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.save_pretrained('../user_data/model_data/pretrain/{}'.format(args.task_name))
        tokenizer.save_pretrained(args.tokenizer_save_dir)
    else:
        logger.info("Found existed tokenizer")
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_save_dir)
        vocab_size = tokenizer.vocab_size


    logger.info("loading data")
    train_query, train_label, test_query, testb_query, train_r2_query, r2_train_label = dh.load_data(args.corpus_dir)
    enhanced_query = [train_query[0], train_query[1]]
    enhanced_query[0].extend(test_query[0])
    enhanced_query[1].extend(test_query[1])
    enhanced_query[0].extend(testb_query[0])
    enhanced_query[1].extend(testb_query[1])
    enhanced_query[0].extend(train_r2_query[0])
    enhanced_query[1].extend(train_r2_query[1])

    if args.is_dual == 1:
        enhanced_query[0].extend(train_query[1])
        enhanced_query[1].extend(train_query[0])
        enhanced_query[0].extend(test_query[1])
        enhanced_query[1].extend(test_query[0])
        enhanced_query[0].extend(testb_query[1])
        enhanced_query[1].extend(testb_query[0])
        enhanced_query[0].extend(train_r2_query[1])
        enhanced_query[1].extend(train_r2_query[0])


    logger.info("tokenizing data")
    train_encodings = tokenizer(enhanced_query[0], enhanced_query[1],
                                truncation=True,
                                padding='max_length',
                                max_length=args.pad_length * 2,
                                return_special_tokens_mask=True)

    train_data = dh.TCPretrainData(train_encodings)


    logger.info("creating model")
    config = NeZhaConfig.from_pretrained(args.pretrain_model)
    model = NeZhaForMaskedLM.from_pretrained(args.pretrain_model, config=config)
    model.resize_token_embeddings(vocab_size)

    mlm_2gram = True if args.mlm_2gram_p != 0.0 else False
    mlm_3gram = True if args.mlm_3gram_p != 0.0 else False
    # data_collator = Ngram(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_1gram_p,
    #                                                 mlm_2gram=mlm_2gram, mlm_2gram_p=args.mlm_2gram_p,
    #                                                 mlm_3gram=mlm_3gram, mlm_3gram_p=args.mlm_3gram_p)
    data_collator = Ngram_albert(tokenizer, True, args.mlm_1gram_p)

    report_to = ['wandb'] if args.is_wandb==1 else None
    training_args = TrainingArguments(
        output_dir='../user_data/model_data/pretrain/{}'.format(args.task_name),
        logging_dir='./logs',
        logging_steps=args.logging_step,
        overwrite_output_dir=True,
        num_train_epochs=args.n_epoch,
        per_device_train_batch_size=args.batch_size,
        save_steps=args.save_step,
        save_total_limit=1,
        report_to=report_to,
        learning_rate=args.init_lr,
        warmup_steps=args.warm_up_step,
        fp16=False if args.apex == 0 else True,
        dataloader_num_workers=6,
        group_by_length=True,
        weight_decay=0.01,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        callbacks=[],
    )

    trainer.train()
    trainer.save_model('../user_data/model_data/pretrain/{}'.format(args.task_name))
    logger.info('Done!')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoch', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--task_name', type=str, default='nezha_pretrain')
    parser.add_argument('--pretrain_model', type=str)
    parser.add_argument('--init_lr', type=float, default=5e-5)
    parser.add_argument('--pad_length', type=int, default=20)
    parser.add_argument('--logging_step', type=int, default=200)
    parser.add_argument('--vocab_save_dir', type=str)
    parser.add_argument('--tokenizer_save_dir', type=str)
    parser.add_argument('--corpus_dir', type=str)
    parser.add_argument('--mlm_1gram_p', type=float, default=0.15)
    parser.add_argument('--mlm_2gram_p', type=float)
    parser.add_argument('--mlm_3gram_p', type=float)
    parser.add_argument('--is_dual', type=int, default=0)
    parser.add_argument('--warm_up_step', type=int)
    parser.add_argument('--save_step', type=int)
    parser.add_argument('--is_wandb', type=int)
    parser.add_argument('--apex', type=int)
    parser.add_argument('--params_verbose', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    dir_check = ['../user_data/logs',
                '../user_data/model_data/pretrain',
                '../user_data/model_data/finetune',
                '../user_data/model_data/onnx',
                '../user_data/vocab_data',
                args.vocab_save_dir,
                args.tokenizer_save_dir]

    for dir in dir_check:
        if not os.path.exists(dir):
            os.mkdir(dir)

    logger = dh.make_logger('../user_data/logs/{}.log'.format(args.task_name))

    if args.params_verbose == 1:
        pt = PrettyTable()
        pt.field_names = ['arg', 'value']
        for k, v in vars(args).items():
            pt.add_row([k, v])
        logger.info('\n' + str(pt))

    if args.is_wandb == 1:
        import os
        import wandb
        os.environ['WANDB_PROJECT'] = '{}'.format(args.task_name)
        wandb.login()
    else:
        pass

    pretrain(args=args, logger=logger)

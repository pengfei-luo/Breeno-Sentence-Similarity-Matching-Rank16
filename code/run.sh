mode='all'
is_submit=1
# ![√] Checked
# ![√] Tested
# ---
pretrain_task_name='pt_ngram'
pretrain_n_epoch=(150 150 150 150)
# ---
train_task_name='ft_ngram'
train_k_fold=1
train_n_epoch=(4 4 4 4)
# --------------------
pretrain_batch_size=128
#pretrain_model=("../user_data/model_data/download/nezha_cn_base" "../user_data/model_data/download/nezha_base_wwm" "../user_data/model_data/download/macbert_cn_base" "../user_data/model_data/download/roberta_wwm_ext")
pretrain_model=("../user_data/model_data/download/nezha_cn_base" "../user_data/model_data/download/nezha_cn_base" "../user_data/model_data/download/nezha_base_wwm" "../user_data/model_data/download/nezha_base_wwm")
pretrain_init_lr=5e-5
pretrain_pad_length=18
pretrain_logging_step=100000
# ![√] decide whether dual based on experiments
pretrain_is_dual=(0 0 0 0)
pretrain_seed=(42 43 42 43)
vocab_save_dir='../user_data/vocab_data'
tokenizer_save_dir='../user_data/tokenizer_data'
mlm_1gram_p=(0.15 0.175 0.15 0.175)
mlm_2gram_p=0.5
mlm_3gram_p=0.4
pretrain_warm_up_step=15000
pretrain_save_step=50000
pretrain_apex=1
if [ "${is_submit}" -eq 1 ]; then
  params_verbose=1
  output_log=1
  corpus_dir='/tcdata'
  is_wandb=0
else
  params_verbose=1
  output_log=0
  corpus_dir='../user_data/tsv_data'
  is_wandb=1
fi
# --------------------
train_batch_size=128
train_model_from="../user_data/model_data/pretrain/${pretrain_task_name}"
train_enhanced_data=0
train_init_lr=5e-5
train_eval_step=500
train_adversarial='FGM'
train_eps=0.6
train_K_PGD=3
train_metric='auc'
train_pad_length=${pretrain_pad_length}
train_seed=(0 10001 0 10001)
train_test_ratio=0
n_ckpt_save=(1 1 1 1)
train_multi_dropout=10
train_convert_to_onnx=1
train_onnx_save_dir="../user_data/model_data/onnx/${train_task_name}"
#----------------------
if [ "${mode}" = 'all' ] || [ "${mode}" = 'pt' ]; then
  for idx in {0..3}
    do
      CUDA_VISIBLE_DEVICES=${idx} python model_nezha_pretrain.py\
       --n_epoch ${pretrain_n_epoch[idx]}\
       --batch_size ${pretrain_batch_size}\
       --task_name "${pretrain_task_name}${idx}"\
       --pretrain_model ${pretrain_model[idx]}\
       --init_lr ${pretrain_init_lr}\
       --pad_length ${pretrain_pad_length}\
       --logging_step ${pretrain_logging_step}\
       --vocab_save_dir ${vocab_save_dir}\
       --tokenizer_save_dir ${tokenizer_save_dir}\
       --corpus_dir ${corpus_dir}\
       --mlm_1gram_p ${mlm_1gram_p[idx]}\
       --mlm_2gram_p ${mlm_2gram_p}\
       --mlm_3gram_p ${mlm_3gram_p}\
       --is_dual ${pretrain_is_dual[idx]}\
       --warm_up_step ${pretrain_warm_up_step}\
       --save_step ${pretrain_save_step}\
       --is_wandb ${is_wandb}\
       --apex ${pretrain_apex}\
       --seed ${pretrain_seed[idx]}\
       --params_verbose ${params_verbose}&
    done
  wait
fi
#-----------------------
if [ "${mode}" = 'all' ] || [ "${mode}" = 'ft' ]; then
  for idx in {0..3}
    do
      CUDA_VISIBLE_DEVICES=${idx} python model_nezha_finetune.py\
       --batch_size ${train_batch_size}\
       --k_fold ${train_k_fold}\
       --n_epoch ${train_n_epoch[idx]}\
       --task_name "${train_task_name}${idx}"\
       --model_from "${train_model_from}${idx}"\
       --corpus_dir ${corpus_dir}\
       --tokenizer_save_dir ${tokenizer_save_dir}\
       --enhanced_data ${train_enhanced_data}\
       --init_lr ${train_init_lr}\
       --eval_step ${train_eval_step}\
       --n_ckpt_save ${n_ckpt_save[idx]}\
       --adversarial ${train_adversarial}\
       --eps ${train_eps}\
       --K_PGD ${train_K_PGD}\
       --metric ${train_metric}\
       --pad_length ${train_pad_length}\
       --seed ${train_seed[idx]}\
       --test_ratio ${train_test_ratio}\
       --multi_dropout ${train_multi_dropout}\
       --convert_to_onnx ${train_convert_to_onnx}\
       --onnx_save_dir ${train_onnx_save_dir}\
       --params_verbose ${params_verbose}&
    done
  wait
fi
#-------------------------
if [ "${mode}" = 'all' ] || [ "${mode}" = 'inf' ]; then
  CUDA_VISIBLE_DEVICES=0 python inf.py\
   --pad_length ${train_pad_length}\
   --onnx_model_dir ${train_onnx_save_dir}\
   --tokenizer_save_dir ${tokenizer_save_dir}\
   --output_log ${output_log}
fi

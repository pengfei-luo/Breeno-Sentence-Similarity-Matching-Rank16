import json
from ai_hub import inferServer
import argparse
import onnxruntime
from onnxruntime.backend import backend
import numpy as np
from transformers import BertTokenizer
import multiprocessing
import torch.multiprocessing as mp
import utils
import time


def create_InferProcess(queue, connect):
    # fetch the model path from queue, and the create a InferProcess with model path and pipe connection object.
    path = queue.get()
    InferProcess(path, connect).run()


class InferProcess(object):
    def __init__(self, path, connect):
        super().__init__()
        # load the corresponding onnx model
        self.sess = onnxruntime.InferenceSession(path)
        self.connect = connect

    def run(self):
        # start the InferProcess, we use pipe connection here.
        while True:
            data = self.connect.recv()
            if isinstance(data, dict):
                logits = self.sess.run(None, data)
                val = softmax(logits[0][0])[1].item()
                self.connect.send(val)
            elif isinstance(data, str):
                self.connect.close()
                break


class AIHubInfer(inferServer) :
    def __init__(self, pool) :
        super().__init__(pool)

    # 数据前处理
    def pre_process(self, req_data) :
        input_batch = {}
        input_batch["input"] = req_data.form.getlist("input")
        input_batch["index"] = req_data.form.getlist("index")

        return input_batch

    # 数据后处理，如无，可空缺
    def post_process(self, predict_data) :
        response = json.dumps(predict_data)
        return response

    # 如需自定义，可覆盖重写
    def predict(self, preprocessed_data) :
        input_list = preprocessed_data["input"]
        index_list = preprocessed_data["index"]

        response_batch = {}
        response_batch["results"] = []
        for i in range(len(index_list)) :
            index_str = index_list[i]

            response = {}
            try :
                input_sample = input_list[i].strip()
                elems = input_sample.strip().split("\t")
                query_A = elems[0].strip()
                query_B = elems[1].strip()
                predict = infer(pool, query_A, query_B)
                response["predict"] = predict
                response["index"] = index_str
                response["ok"] = True
            except Exception as e :
                response["predict"] = 0
                response["index"] = index_str
                response["ok"] = False
            response_batch["results"].append(response)

        return response_batch

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def infer(pool, query_A, query_B) :

    batch = tokenizer(query_A, query_B, truncation=True, padding='max_length', max_length=args.pad_length * 2)
    input_data = {
        'input_ids' : np.array([batch['input_ids']]).astype(np.int64),
        'attention_mask' : np.array([batch['attention_mask']]).astype(np.int64),
        'token_type_ids' : np.array([batch['token_type_ids']]).astype(np.int64),
    }

    # feed a single example to every pipe which connects a single model
    pred_val = []
    for connect in connect_list:
        connect.send(input_data)
    # and then fetch the result
    for connect in connect_list:
        val = connect.recv()
        pred_val.append(val)
    # mean pooling
    return np.mean(pred_val).item()


def load_model_for_inference(model_dir, tokenizer_save_dir):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_save_dir)
    models_path_list = utils.get_file_list(model_dir, 'onnx')
    logger.info("Read to load {} onnx model".format(len(models_path_list)))

    model_list = []
    return model_list, models_path_list, tokenizer


if __name__ == "__main__" :
    import time
    import logging
    import data_helper as dh
    parser = argparse.ArgumentParser()
    parser.add_argument('--pad_length', type=int, default=18)
    parser.add_argument('--onnx_model_dir', type=str, default='../user_data/model_data/onnx')
    parser.add_argument('--tokenizer_save_dir', type=str)
    parser.add_argument('--output_log', type=int, default=0)
    args = parser.parse_args()
    logger = logging.getLogger('werkzeug')
    logger.disabled = True
    logger = dh.make_logger('../user_data/logs/inf.log')

    multiprocessing.set_start_method('spawn')

    queue = multiprocessing.Queue()
    model_list, models_path_list, tokenizer = load_model_for_inference(args.onnx_model_dir, args.tokenizer_save_dir)

    # put model path into queue
    for path, fdir, file_name in models_path_list:
        queue.put(path)
        logger.info('putting model {} into queue'.format(path))

    pool = []
    connect_list = []
    # create multi infer process
    for path, fdir, file_name in models_path_list:
        c1, c2 = multiprocessing.Pipe()
        connect_list.append(c1)
        p = multiprocessing.Process(target=create_InferProcess, args=(queue, c2))
        p.start()
        pool.append(p)

    # sleep for 15 secs, waiting for loading the models
    time.sleep(15)
    aihub_infer = AIHubInfer(pool)
    aihub_infer.run(debuge=False)
    for connect in connect_list:
        connect.send('exit')
    for p in pool:
        p.join()
        p.close()
    if args.output_log:
        utils.output_log()

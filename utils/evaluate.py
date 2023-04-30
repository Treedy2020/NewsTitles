import os
import time
import paddle
import numpy as np
from tqdm import tqdm
from utils.get_data import test_titles
from utils.constant import (
                                label_dict,
                                TOTAL_SIZE,
                                MAX_SEQ_LEN,
                            )

def evaluate(model, criterion, metric, data_loader):
    """
    Given a dataset, it evals model and computes the metric.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()
    losses = []
    for batch in tqdm(data_loader):
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval loss: %.5f, accu: %.5f" % (np.mean(losses), accu))
    model.train()
    metric.reset()
    return accu

def infer(model_save_path, tokenizer_save_path, test_data_loader, test_dataset_part2):
    inf_model = paddle.jit.load(model_save_path)
    inf_model.eval()
    res = []

    inf_start_t = time.time()
    for input_ids, token_type_ids in tqdm(test_data_loader):
        logits = inf_model(input_ids, token_type_ids)
        curr_ind = paddle.argmax(logits, axis=1)
        res += curr_ind.numpy().tolist()

    for input_ids, token_type_ids in tqdm(test_dataset_part2):
        input_ids, token_type_ids = paddle.to_tensor(input_ids.reshape(1, MAX_SEQ_LEN) , dtype='int64'), paddle.to_tensor(token_type_ids.reshape(1, MAX_SEQ_LEN) , dtype='int64')
        logits = inf_model(input_ids, token_type_ids)
        curr_ind = paddle.argmax(logits, axis=1)
        res += curr_ind.numpy().tolist()
    print(f'Finished {TOTAL_SIZE} items in {time.time() - inf_start_t} seconds!')
    
    assert len(res) == TOTAL_SIZE, 'The length of final results is NOT CORRECT.'
    with open(os.path.join(tokenizer_save_path, 'result.txt'), 'w') as f:
        print('Examples:')
        for i in range(TOTAL_SIZE):
            text = label_dict[res[i]] + '\t' + test_titles[i] + '\n'
            # text = label_dict[res[i]] + '\n'
            if not i%(TOTAL_SIZE//20):
                print('\t', label_dict[res[i]] + '\t' + test_titles[i])
            f.write(text)
        
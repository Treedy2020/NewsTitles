import os
import time
import paddle
import paddle.nn.functional as F 
from paddle.io import BatchSampler, DataLoader
from paddlenlp.transformers import  AutoModelForSequenceClassification, AutoTokenizer
from utils.evaluate import evaluate, infer
from utils.dataset import TextDataset
from utils.get_data import (
                                train_titles,
                                val_titles,
                                test_data_part1,
                                test_data_part2
                            )
from utils.constant import (
                                EPOCHS,
                                LEARNING_RATE, 
                                BATCH_SIZE,
                                SAVE_FREQUENCY,
                                LOG_FREQUENCY,
                                BASE_LINE,
                                NUM_WORKERS,
                                SAVE_PATH
                            )



def train(model_name:str):
    """ The main training process for a model with model_name, the model files will save in the SAVE_PATH automaticlly.

    Args:
        model_name (str): the name for the model you wana do a fine-tuning which support 
        sequence classification at doc https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html#id1.
        
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_classes=14)
    model = paddle.jit.to_static(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset, val_dataset = TextDataset(data=train_titles, tokenizer=tokenizer), TextDataset(data=val_titles, tokenizer=tokenizer)
    test_dataset_part1 = TextDataset(data=test_data_part1, tokenizer=tokenizer, isTest=True)
    test_dataset_part2 = TextDataset(data=test_data_part2, tokenizer=tokenizer, isTest=True)

    # get the samplers
    train_batch_sampler = BatchSampler(train_dataset,
                                        shuffle=False,
                                        batch_size=BATCH_SIZE,)

    val_batch_sampler = BatchSampler(val_dataset,
                                    shuffle=False,
                                    batch_size=BATCH_SIZE,)
    
    test_batch_sampler = paddle.io.BatchSampler(test_dataset_part1,
                                                shuffle=False, 
                                                batch_size=BATCH_SIZE)

    # get the data loaders
    train_data_loader = DataLoader(dataset=train_dataset,
                                    batch_sampler=train_batch_sampler,
                                    return_list=True,
                                    num_workers=NUM_WORKERS)
    
    val_data_loader = DataLoader(dataset=val_dataset,
                                batch_sampler=val_batch_sampler,
                                return_list=True,
                                num_workers=NUM_WORKERS)

    test_data_loader = paddle.io.DataLoader(dataset=test_dataset_part1,
                                            batch_sampler=test_batch_sampler,
                                            return_list=True,
                                            num_workers=NUM_WORKERS)

    # get the items before the training loop
    optimizer = paddle.optimizer.Adam(learning_rate=LEARNING_RATE,
                                    parameters=model.parameters(),)
    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    # the training loop
    model.train() 
    best_acc = BASE_LINE
    model_last_name = model_name.split('/')[-1]
    model_save_path = os.path.join(SAVE_PATH, model_last_name, model_last_name)
    tokenizer_save_path = os.path.join(SAVE_PATH, model_last_name)
    for epoch in range(EPOCHS):
        print(f"epoch: {epoch + 1}, {time.ctime()}")
        metric.reset()
        epoch_start_t = time.time()
        for ind, item in enumerate(train_data_loader):
            start_t = time.time()
            input_ids, token_type_ids, labels = item
            logits = model(input_ids, token_type_ids)
            loss = criterion(logits, labels)
            probs = F.softmax(logits, axis=1)

            correct = metric.compute(probs, labels)
            batch_acc = metric.update(correct)
            acc = metric.accumulate()
            
            loss.backward()
            ave_t = (time.time() - start_t)

            if ind and (not ind%SAVE_FREQUENCY):
                accu = evaluate(model, criterion, metric, val_data_loader)
                if accu > best_acc:
                    best_acc = accu
                    print('\t Best Acc: {:.9f}'.format(best_acc))
                    paddle.jit.save(model, model_save_path)
                    tokenizer.save_pretrained(tokenizer_save_path)
            
            if ind and (not ind%LOG_FREQUENCY):
                print(f'\t step:{ind}/{len(train_data_loader)},', 'average time: {:.4f},'.format(ave_t), 'loss: {:.6f}'.format(loss.numpy()[0]), 'Batch Acc:{:.9f}, Acc:{:.9f}'.format(batch_acc, acc))

            optimizer.step()
            optimizer.clear_grad()
        print(f'Epoch training finished in {time.time()-epoch_start_t} seconds for {len(train_data_loader)} items!')
    if best_acc != BASE_LINE: 
        print('Last Evaluate:')
        last_acc = evaluate(model, criterion, metric, val_data_loader)
        print(f'Result of infer saved in: {tokenizer_save_path} as result.txt')
        infer(model_save_path, tokenizer_save_path, test_data_loader, test_dataset_part2)
    else:
        print('This model has not reached the BASE_LINE during training, the inffer process has ignored.')
    
    paddle.device.cuda.empty_cache()

    # Record the model results
    with open('./record.txt', 'a') as f:
        f.write(f'{model_name.split("/")[-1]}\t{best_acc}\t{last_acc}\n')


if __name__ == '__main__':
    model_list = [
                    'hfl/rbt6',
                    'ernie-3.0-mini-zh',
                    'tinybert-4l-312d-zh',
                    'hfl/rbt3',
                ]
    for model_name in model_list:
        train(model_name)
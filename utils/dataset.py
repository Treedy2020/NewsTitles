import numpy as np
from paddle.io import Dataset
from utils.constant import (MAX_SEQ_LEN)

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length=MAX_SEQ_LEN, isTest=False):
        super(TextDataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.isTest = isTest

    def __getitem__(self, index):
        if  not self.isTest:
            text, label = self.data[index][0], self.data[index][1]
            encoded = self.tokenizer.encode(text, max_seq_len=self.max_seq_length, pad_to_max_seq_len=True)
            input_ids, token_type_ids  = encoded['input_ids'], encoded['token_type_ids']
            return tuple([np.array(x, dtype='int64') for x in [input_ids, token_type_ids, [label]]])
        else:
            title = self.data[index]
            encoded = self.tokenizer.encode(title, max_seq_len=self.max_seq_length, pad_to_max_seq_len=True)
            input_ids, token_type_ids  = encoded['input_ids'], encoded['token_type_ids']
            return tuple([np.array(x, dtype='int64') for x in [input_ids, token_type_ids]])

    def __len__(self):
        return len(self.data)


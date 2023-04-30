from sklearn.model_selection import train_test_split
from paddle.io import BatchSampler, DataLoader
from utils.dataset import TextDataset
from utils.constant import (
                                TEST_SIZE,
                                RANDOM_STATE, 
                                TEST_LEN1,
                                NUM_WORKERS
                            )


# Get Train titles
title_labels, titles = [], []
with open('./data/Train.txt', 'r') as f:
    for line in f.readlines():
        label, classify, title = line.strip('\n').split('\t')
        title_labels.append(int(label))
        titles.append(title)

# Get Test titles
test_titles = []
with open('./data/Test.txt', 'r') as f:
    count = 0
    for line in f.readlines():
        test_titles.append(line.strip('\n'))

title_with_labels = [(t, l) for t,l in zip(titles, title_labels)]
train_titles, val_titles = train_test_split(title_with_labels,
                                            test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Split the test data into two parts
test_data_part1, test_data_part2 = test_titles[:TEST_LEN1], test_titles[TEST_LEN1:]

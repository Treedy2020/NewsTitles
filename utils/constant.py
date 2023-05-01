# Some constant for the training process
EPOCHS = 1
LEARNING_RATE = 5e-5
BATCH_SIZE = 1024
SAVE_FREQUENCY = 100
LOG_FREQUENCY = 20
TOTAL_SIZE = 83599
TEST_LEN1 = (TOTAL_SIZE//BATCH_SIZE)*BATCH_SIZE
TEST_SIZE = 0.10
RANDOM_STATE = 1024
BASE_LINE = 0.0
NUM_WORKERS = 4
MAX_SEQ_LEN = 48
SAVE_PATH = './models/'

class_lis = ['财经', '彩票', '房产', '股票', '家居', '教育', '科技', '社会', '时尚', '时政', '体育', '星座', '游戏', '娱乐']
label_dict = {ind: content for ind, content in enumerate(class_lis)}
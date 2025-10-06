import os
from re import T

# OpenBG500 paths (reuse same as TransE)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'OpenBG500')
ENTITY_PATH = os.path.join(DATA_DIR, 'OpenBG500_entity2text.tsv')
RELATION_PATH = os.path.join(DATA_DIR, 'OpenBG500_relation2text.tsv')
TRAIN_PATH = os.path.join(DATA_DIR, 'OpenBG500_train.tsv')
DEV_PATH = os.path.join(DATA_DIR, 'OpenBG500_dev.tsv')
TEST_PATH = os.path.join(DATA_DIR, 'OpenBG500_test.tsv')

# Training/eval
TRAIN_BATCH_SIZE = 1024 #1024
DEV_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1

# RotatE hyperparams
MODEL = 'RotatE'
DOUBLE_ENTITY = True
DOUBLE_RELATION = False  #False
NEGATIVE_SAMPLE_SIZE = 500  #500
HIDDEN_DIM = 1000   #1000
GAMMA = 9.0 #9.0
ADVERSARIAL = True
ADV_TEMPERATURE = 1.0   #1.0
LEARNING_RATE = 0.00005  #0.00005
REGULARIZATION = 0.0
CPU_NUM = 10

# Train loop control
MAX_STEPS = 100000   #100000
# VALIDATION = True
VALIDATION = False
VALID_STEPS = 1000
# VALID_STEPS = 10000
SAVE_CHECKPOINT_STEPS = 2000
LOG_STEPS = 100
TEST_LOG_STEPS = 1000

# Result dir
RESULT_DIR = 'result'

# Finetune on train+dev
FINETUNE_AFTER_TRAIN = False
# FINETUNE_AFTER_TRAIN = True
FINETUNE_EPOCHS = 5
FINETUNE_LR = 0.00005



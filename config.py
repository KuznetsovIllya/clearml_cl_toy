import os

class TrainConfig:
    BATCH_SIZE = 4
    NUM_OF_CLASSES = 10
    INIT_LR = 0.00001
    NUM_EPOCHS = 2
    MODEL_PATH = "D:/Projects/ClearML_test/data/output"

    DATA_PATH = "D:/Projects/ClearML_test/data/data1"
    TRAIN_PATH = os.path.join(DATA_PATH, "train")
    EVAL_PATH = os.path.join(DATA_PATH, "eval")
    TEST_PATH = os.path.join(DATA_PATH, "test")

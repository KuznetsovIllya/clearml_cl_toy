import os

class TrainConfig:
    BATCH_SIZE = 4
    NUM_OF_CLASSES = 2
    INIT_LR = 0.00001
    NUM_EPOCHS = 2
    MODEL_PATH = "/projects/clearml_cl_toy/output"

    DATA_PATH = "/projects/clearml_cl_toy/destination"
    TRAIN_PATH = os.path.join(DATA_PATH, "train")
    EVAL_PATH = os.path.join(DATA_PATH, "eval")
    TEST_PATH = os.path.join(DATA_PATH, "test")

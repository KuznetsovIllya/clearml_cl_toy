# set the matplotlib backend so figures can be saved in the background
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
# from imutils import paths
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from config import TrainConfig as config
from build_dataset import list_images
import json

from clearml import Task
task = Task.init(project_name='rabbit_fox', task_name='train_2nd_nn')
configuration_dict = {'batch_size': 1}
configuration_dict = task.connect(configuration_dict)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(
            set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

# determine the total number of image paths in training, validation,
# and testing directories
trainImages = len(list(list_images(config.TRAIN_PATH)))
evalImages = len(list(list_images(config.EVAL_PATH)))
testImages = len(list(list_images(config.TEST_PATH)))

# initialize the training training data augmentation object
trainAug = ImageDataGenerator(
	# rotation_range=25,
	# zoom_range=0.1,
	# width_shift_range=0.1,
	# height_shift_range=0.1,
	# shear_range=0.2,
	# horizontal_flip=True,
    vertical_flip=True,
	fill_mode="nearest")

valAug = ImageDataGenerator()

mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

# initialize the training generator
trainGen = trainAug.flow_from_directory(
	config.TRAIN_PATH,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=True,
	batch_size=configuration_dict.get('batch_size', 100))

# initialize the validation generator
valGen = valAug.flow_from_directory(
	config.EVAL_PATH,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=config.BATCH_SIZE)

# initialize the testing generator
testGen = valAug.flow_from_directory(
	config.TEST_PATH,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=config.BATCH_SIZE)

print('We have the following classes for train: ')
print(trainGen.classes)
print('We have the following classes for validation: ')
print(valGen.classes)
print('We have the following classes for test: ')
print(testGen.classes)


# load the ResNet-152 V2 network, ensuring the head FC layer sets are left
# off
print("[INFO] preparing model...")
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                     input_tensor=Input(shape=(224, 224, 3)))
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(config.NUM_OF_CLASSES, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the training process
for layer in baseModel.layers:
	layer.trainable = False

# compile the model
opt = Adam(lr=config.INIT_LR, decay=config.INIT_LR / config.NUM_EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["acc", "mse"])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(config.MODEL_PATH, "cp{epoch:04d}.ckpt"),
    save_weights_only=False,
    monitor='acc',
    save_freq = 'epoch',
    period = 10,
    mode='auto',
    save_best_only=True,
    verbose = 1
)
early_stoping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto',
    baseline=None, restore_best_weights=False
)

logdir = './logging/'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, update_freq=100)

# train the model
print("[INFO] training model...")
H = model.fit_generator(
	trainGen,
	steps_per_epoch=trainImages // config.BATCH_SIZE,
	validation_data=valGen,
	validation_steps=evalImages // config.BATCH_SIZE,
	epochs=config.NUM_EPOCHS,
    callbacks=[model_checkpoint_callback, tensorboard_callback])

# make predictions on the data
print("[INFO] testing network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,
                                   steps=(testImages // config.BATCH_SIZE) + 1)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
# show a nicely formatted classification report
report_dict = classification_report(testGen.classes, predIdxs,
                            target_names=testGen.class_indices.keys(), output_dict=True)
print(classification_report(testGen.classes, predIdxs,
                            target_names=testGen.class_indices.keys()))

with open('best_score.json') as bs:
    best_score_dict = json.load(bs)
bs.close()

if report_dict["accuracy"]>best_score_dict["accuracy"]:
    print("[INFO] better model found")
    best_score_dict["accuracy"] = report_dict["accuracy"]
    with open("best_score.json", "w") as bs:
        json.dump(best_score_dict,bs)
    bs.close()
    # # serialize the model to disk
    print("[INFO] saving model...")
    tf.saved_model.save(model, "./output")
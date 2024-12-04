#!~/miniconda3/envs/tf2/bin/python
import os
import pathlib
import tensorflow as tf
from model import BlazePose
from config import total_epoch, train_mode, continue_train_from_filename, batch_size, dataset, continue_train, best_pre_train_filename
from data import coordinates, visibility, heatmap_set, data, number_images
import logger

checkpoint_path_heatmap = "checkpoints_heatmap"
checkpoint_path_regression = "checkpoints_regression"
loss_func_mse = tf.keras.losses.MeanSquaredError()
loss_func_bce = tf.keras.losses.BinaryCrossentropy()

model = BlazePose().call()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer, loss=[loss_func_bce, loss_func_mse, loss_func_bce])

if train_mode:
    checkpoint_path = checkpoint_path_regression
else:
    checkpoint_path = checkpoint_path_heatmap
pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

# Optimize RAM for GPU if exist
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Define the callbacks
model_folder_path = os.path.join(checkpoint_path, "models")
pathlib.Path(model_folder_path).mkdir(parents=True, exist_ok=True)
mc = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(
    model_folder_path, "model_ep{epoch:02d}.weights.h5"), save_freq='epoch', save_weights_only=True, verbose=1)

# continue train
if continue_train > 0:
    print("Load heatmap weights", os.path.join(checkpoint_path, "models/{}".format(continue_train_from_filename)))
    model.load_weights(os.path.join(checkpoint_path, "models/{}".format(continue_train_from_filename)))
else:
    if train_mode:
        print("Load heatmap weights", os.path.join(checkpoint_path_heatmap, "models/{}".format(best_pre_train_filename)))
        model.load_weights(os.path.join(checkpoint_path_heatmap, "models/{}".format(best_pre_train_filename)))

if train_mode:
    print("Freeze these layers:")
    for layer in model.layers:
        if not layer.name.startswith("regression"):
            print(layer.name)
            layer.trainable = False
# Freeze heatmap branch when training regression
else:
    print("Freeze these layers:")
    for layer in model.layers:
        if layer.name.startswith("regression"):
            print(layer.name)
            layer.trainable = False

try:
    if dataset == "lsp":
        x_train = data[:(number_images - 400)]
        y_train = [heatmap_set[:(number_images - 400)], coordinates[:(number_images - 400)], visibility[:(number_images - 400)]]

        x_val = data[-400:-200]
        y_val = [heatmap_set[-400:-200], coordinates[-400:-200], visibility[-400:-200]]
    else:
        x_train = data[:(number_images - 2000)]
        y_train = [heatmap_set[:(number_images - 2000)], coordinates[:(number_images - 2000)], visibility[:(number_images - 2000)]]

        x_val = data[-2000:-1000]
        y_val = [heatmap_set[-2000:-1000], coordinates[-2000:-1000], visibility[-2000:-1000]]

    model.fit(x=x_train, y=y_train,
            batch_size=batch_size,
            epochs=total_epoch,
            validation_data=(x_val, y_val),
            callbacks=[mc, logger.keras_custom_callback],
            verbose=1)

    model.summary(print_fn=logger.print_and_log, trinable=True)
    print("Finish training.")
except Exception as ex:
    print(ex)
import os
import pathlib
import tensorflow as tf
from model import BlazePose
from config import total_epoch, train_mode, best_pre_train, continue_train, batch_size, dataset
from data import coordinates, visibility, heatmap_set, data, number_images
import logger

checkpoint_path_heatmap = "checkpoints_heatmap"
checkpoint_path_regression = "checkpoints_regression"
loss_func_mse = tf.keras.losses.MeanSquaredError()
loss_func_bce = tf.keras.losses.BinaryCrossentropy()

model = BlazePose().call()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer, loss=[loss_func_bce, loss_func_mse, loss_func_bce])

if train_mode:
    checkpoint_path = checkpoint_path_regression
else:
    checkpoint_path = checkpoint_path_heatmap
pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

# Define the callbacks
model_folder_path = os.path.join(checkpoint_path, "models")
pathlib.Path(model_folder_path).mkdir(parents=True, exist_ok=True)

# Callback to save the best checkpoint
class SaveBestCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, folder_path):
        super(SaveBestCheckpoint, self).__init__()
        self.folder_path = folder_path
        self.best_loss = float("inf")
        self.best_checkpoint = None

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss is not None and val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_checkpoint = os.path.join(self.folder_path, f"model_ep{epoch + 1:02d}.weights.h5")
            logger.tf_logger.info(f"New best checkpoint: {self.best_checkpoint} with validation loss: {self.best_loss}")

# Instantiate the callback
best_checkpoint_callback = SaveBestCheckpoint(model_folder_path)

mc = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(
    model_folder_path, "model_ep{epoch:02d}.weights.h5"),
    save_freq='epoch',
    save_weights_only=True,
    verbose=1)

# Load the best checkpoint if continuing training
if continue_train > 0:
    checkpoint_file = os.path.join(checkpoint_path, f"models/model_ep{continue_train:02d}.weights.h5")
    print(f"Load heatmap weights: {checkpoint_file}")
    model.load_weights(checkpoint_file)
else:
    if train_mode and best_pre_train:
        checkpoint_file = os.path.join(checkpoint_path_heatmap, f"models/model_ep{best_pre_train:02d}.weights.h5")
        print(f"Load heatmap weights: {checkpoint_file}")
        model.load_weights(checkpoint_file)

# Freeze layers based on training mode
if train_mode:  # train_mode 1 is for training the heatmap branch
    print("Freeze these layers:")
    for layer in model.layers:
        if not layer.name.startswith("regression"):
            print(layer.name)
            layer.trainable = False
else:  # train_mode = 0
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

    print(f"Number of training samples: {len(x_train)}")
    print(f"Number of validation samples: {len(x_val)}")
    
    model.fit(x=x_train, y=y_train,
              batch_size=batch_size,
              epochs=total_epoch,
              validation_data=(x_val, y_val),
              callbacks=[mc, best_checkpoint_callback],
              verbose=1)

    model.summary()
    print("Finish training.")
except Exception as ex:
    logger.tf_logger.error(f"Exception occurred: {ex}")
    print(ex)

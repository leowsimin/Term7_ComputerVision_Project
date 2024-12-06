#!~/miniconda3/envs/tf2/bin/python
import os
import pathlib
import tensorflow as tf
from model import BlazePose
from config import total_epoch, train_mode, continue_train_from_filename, batch_size, dataset, continue_train, best_pre_train_filename, img_idxs
from data import x_train, y_train, x_val, y_val, x_test, y_test
from utils import metrics
from utils.draw import draw_heatmaps, draw_images
import utils.logger as logger

checkpoint_path_heatmap = "checkpoints_heatmap"
checkpoint_path_regression = "checkpoints_regression"
loss_func_mse = tf.keras.losses.MeanSquaredError()
loss_func_bce = tf.keras.losses.BinaryCrossentropy()

model = BlazePose().call()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer, loss=[loss_func_bce, loss_func_mse, loss_func_bce], 
            metrics=[None, metrics.PCKMetric(), None])

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

# Continue training
if continue_train > 0:
    model_folder_path = os.path.join(checkpoint_path, "models")
    print("Load heatmap weights", os.path.join(model_folder_path, "{}".format(continue_train_from_filename)))
    model.load_weights(os.path.join(model_folder_path, "{}".format(continue_train_from_filename)))
else:
    if train_mode:
        print("Load heatmap weights", os.path.join(checkpoint_path_heatmap, "models/{}".format(best_pre_train_filename)))
        model.load_weights(os.path.join(checkpoint_path_heatmap, "models/{}".format(best_pre_train_filename)))

# Define the callbacks
if continue_train_from_filename:
    model_folder_path = os.path.join(checkpoint_path, f"models_1")
else:
    model_folder_path = os.path.join(checkpoint_path, "models")
pathlib.Path(model_folder_path).mkdir(parents=True, exist_ok=True)

# Save only the best model
mc = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(model_folder_path, "best_model.weights.h5"),
    monitor='val_loss',  # Monitor validation loss
    save_best_only=True,  # Save only the best model
    save_weights_only=True,  # Save weights only
    verbose=1
)

# Custom callback to log the best loss
class LogBestLossCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss is not None and val_loss < self.best_loss:
            self.best_loss = val_loss
            print(f"New best validation loss: {self.best_loss:.4f}")

log_best_loss_cb = LogBestLossCallback()

# Freeze layers based on the training mode
if train_mode:
    print("Freeze these layers:")
    for layer in model.layers:
        if not layer.name.startswith("regression"):
            print(layer.name)
            layer.trainable = False
else:
    print("Freeze these layers:")
    for layer in model.layers:
        if layer.name.startswith("regression"):
            print(layer.name)
            layer.trainable = False

try:
    # Train the model with callbacks
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=total_epoch,
        validation_data=(x_val, y_val),
        callbacks=[mc, log_best_loss_cb, logger.keras_custom_callback],
        shuffle=True,
        verbose=1
    )
    
    # Evaluate the model on the test set
    res = model.evaluate(
        x=x_test,
        y=y_test,
        batch_size=batch_size,
        callbacks=[logger.keras_custom_callback]
    )
    print("Test PCK score:", res[-1])

    # Save the newest weights if in regression mode
    if train_mode > 0:
        newest_weight_file = os.path.join(model_folder_path, f"model_ep{total_epoch:02d}.weights.h5")
        print(f"Newest weights saved to: {newest_weight_file}")

    # Generate images for evaluation
    image_files = draw_images(model, img_idxs=img_idxs)

    print("Finish training.")
except Exception as ex:
    print(ex)

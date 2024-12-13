#!~/miniconda3/envs/tf2/bin/python
import os
import pathlib
import tensorflow as tf
from config import total_epoch, train_mode, best_pre_train_filename, continue_train, batch_size, dataset, select_model
from deeper_base_model import Deeper_Base_BlazePose
from data import coordinates, visibility, heatmap_set, data, number_images
from deeper_base_model import Deeper_Base_BlazePose
from CBAM_model import CBAM_BlazePose
from EXTRA_model import EXTRA_BlazePose
from base_model import BlazePose
from VIT_model import VIT_BlazePose
import utils.metrics as metrics

checkpoint_path_heatmap = "checkpoints_heatmap"
checkpoint_path_regression = "checkpoints_regression"
loss_func_mse = tf.keras.losses.MeanSquaredError()
loss_func_bce = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

def load_model():
    model = None
    if select_model == 0:
        print("Using Deeper_Base_BlazePose")
        model = Deeper_Base_BlazePose().call()
    elif select_model == 1:
        print("Using CBAM_BlazePose")
        model = CBAM_BlazePose().call()
    elif select_model == 2:
        print("Using EXTRA_BlazePose")
        model =  EXTRA_BlazePose().call()
    elif select_model == 3:
        print("Using VIT_BlazePose")
        model = VIT_BlazePose().call()
    else:
        model = BlazePose().call()
    assert model is not None, "Invalid model selected. Change select_model value to something that enters the if-else blocks"
    model.compile(optimizer, loss=[loss_func_bce, loss_func_mse, loss_func_bce], metrics=[None, metrics.PCKMetric(), None])
    return model

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
except Exception as ex:
    print(ex)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only allocate necessary GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU instead.")

model = load_model()
#optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#model.compile(optimizer, loss=[loss_func_bce, loss_func_mse, loss_func_bce])

pathlib.Path(checkpoint_path_heatmap).mkdir(parents=True, exist_ok=True)
pathlib.Path(checkpoint_path_regression).mkdir(parents=True, exist_ok=True)

heatmap_model_path = os.path.join(checkpoint_path_heatmap, "models")
regression_model_path = os.path.join(checkpoint_path_regression, "models")

# Define the callbacks
# model_folder_path = os.path.join(checkpoint_path, "models")
# pathlib.Path(model_folder_path).mkdir(parents=True, exist_ok=True)
mc_heatmap = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(
    heatmap_model_path, "model_ep{epoch:02d}.weights.h5"), save_freq='epoch', save_weights_only=True, verbose=1)

print("Freezing regression layers:")
for layer in model.layers:
    if layer.name.startswith("regression"):
        print(layer.name)
        layer.trainable = False

model.fit(x=x_train, y=y_train,
            batch_size=batch_size,
            epochs=total_epoch,
            validation_data=(x_val, y_val),
            callbacks=mc_heatmap,
            verbose=1)

print("Finished training heatmap branches")

print("Un-freezing regression layers:")
for layer in model.layers:
    if layer.name.startswith("regression"):
        print(layer.name)
        layer.trainable = True

mc_regression = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(
    regression_model_path, "model_ep{epoch:02d}.weights.h5"), save_freq='epoch', save_weights_only=True, verbose=1)

print("Load heatmap weights", os.path.join(checkpoint_path_heatmap, "models/model_ep{}.weights.h5".format(best_pre_train_filename)))
model.load_weights(os.path.join(checkpoint_path_heatmap, "models/model_ep{}.weights.h5".format(best_pre_train_filename)))

print("Freeze non-regression layers:")
for layer in model.layers:
    if not layer.name.startswith("regression"):
        print(layer.name)
        layer.trainable = False

model.fit(x=x_train, y=y_train,
        batch_size=batch_size,
        epochs=total_epoch,
        validation_data=(x_val, y_val),
        callbacks=mc_regression,
        verbose=1)

model.summary()
print("Finish training.")

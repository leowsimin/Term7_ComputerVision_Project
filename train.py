#!~/miniconda3/envs/tf2/bin/python
import os
import pathlib
import tensorflow as tf
from model import BlazePose
from config import (
    total_epoch,
    train_mode,
    continue_train_from_filename,
    batch_size,
    dataset,
    continue_train,
    best_pre_train_filename,
    img_idxs_lsp,
    img_idxs_lspet,
    pretrain,
)
from data import x_train, y_train, x_val, y_val, x_test, y_test
from utils import metrics
from utils.draw import draw_heatmaps, draw_images
import utils.logger as logger

# import utils.experiment_tracker as experiment_tracker

# import mlflow

checkpoint_path_heatmap = "checkpoints_heatmap_lspet"
checkpoint_path_regression = "checkpoints_regression"
loss_func_mse = tf.keras.losses.MeanSquaredError()
loss_func_bce = tf.keras.losses.BinaryCrossentropy()


model = BlazePose().call()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 0.0005 * (0.5 ** (epoch // 10))
)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

if pretrain:
    # NOTE: FOR PRETRAINING ON LSPET
    print("Pretraining:")
    model.compile(
        optimizer=optimizer,
        loss=[loss_func_bce, loss_func_mse, loss_func_bce],
        loss_weights=[0.8999, 0.0001, 0.1],  # majority loss is heatmap loss
        metrics=[None, metrics.PCKMetric(), None],
    )
else:
    model.compile(
        optimizer,
        loss=[loss_func_bce, loss_func_mse, loss_func_bce],
        metrics=[None, metrics.PCKMetric(), None],
    )


if train_mode:
    checkpoint_path = checkpoint_path_regression
else:
    checkpoint_path = checkpoint_path_heatmap
pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

# Optimize RAM for GPU if exist
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# continue train
if continue_train > 0:
    model_folder_path = os.path.join(checkpoint_path, "models")
    print(
        "Load heatmap weights",
        os.path.join(model_folder_path, "{}".format(continue_train_from_filename)),
    )
    model.load_weights(
        os.path.join(model_folder_path, "{}".format(continue_train_from_filename))
    )
else:
    if train_mode:
        print(
            "Load heatmap weights",
            os.path.join(
                checkpoint_path_heatmap, "models/{}".format(best_pre_train_filename)
            ),
        )
        model.load_weights(
            os.path.join(
                checkpoint_path_heatmap, "models/{}".format(best_pre_train_filename)
            )
        )

# Define the callbacks
if continue_train_from_filename:
    model_folder_path = os.path.join(checkpoint_path, f"models_1")
    pathlib.Path(model_folder_path).mkdir(parents=True, exist_ok=True)

else:
    model_folder_path = os.path.join(checkpoint_path, "models")
    pathlib.Path(model_folder_path).mkdir(parents=True, exist_ok=True)

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
    if pretrain:
        # Pretrain the Model - redefining mc to save the best pretrained model only.
        mc = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_folder_path, "best_model.weights.h5"),
            monitor="val_loss",  # Monitor validation loss
            save_best_only=True,  # Save only the best model
            save_weights_only=True,
            verbose=1,
        )
    else:  # finetuning
        mc = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_folder_path, "model_ep{epoch:02d}.weights.h5"),
            save_freq="epoch",
            save_weights_only=True,
            verbose=1,
        )

    model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=total_epoch,
        validation_data=(x_val, y_val),
        callbacks=[mc, logger.keras_custom_callback, lr_schedule],
        shuffle=True,
        verbose=1,
    )

    res = model.evaluate(
        x=x_test,
        y=y_test,
        batch_size=batch_size,
        callbacks=[logger.keras_custom_callback],
    )
    print("Test PCK score:", res[-1])
    # mlflow.log_metrics({"test_coordinates_pck": res[-1]})

    # model.summary(print_fn=logger.print_and_log, show_trainable=True)
    # tf.keras.utils.plot_model(model, to_file='/tmp/model_architecture.png', show_shapes=True, show_layer_activations=True, show_trainable=True)
    # mlflow.log_artifact('/tmp/model_architecture.png')

    if pretrain:
        image_files = draw_images(model, img_idxs=img_idxs_lspet)
        image_files = draw_heatmaps(model, img_idxs=img_idxs_lspet)
        image_files = draw_images(model, img_idxs=img_idxs_lsp)
        image_files = draw_heatmaps(model, img_idxs=img_idxs_lsp)
    else:
        image_files = draw_images(model, img_idxs=img_idxs_lsp)
        # for image_file in image_files:
        #     mlflow.log_artifact(image_file)

        image_files = draw_heatmaps(model, img_idxs=img_idxs_lsp)
        # for image_file in image_files:
        # mlflow.log_artifact(image_file)

    if (
        train_mode > 0
    ):  # if regression mode -> model prbl quite good -> save model weights
        model_folder_path = os.path.join(checkpoint_path, "models")
        newest_weight_file = os.path.join(
            model_folder_path, f"model_ep{total_epoch:02d}.weights.h5"
        )
        # mlflow.log_artifact(newest_weight_file)

    print("Finish training.")
except Exception as ex:
    print(ex)

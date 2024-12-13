import cv2, os
import tensorflow as tf
#from tensorflow.python.ops import math_ops, array_ops, gen_math_ops
import numpy as np
from config import epoch_to_test, eval_mode, dataset, use_existing_model_weights, pck_metric, batch_size, img_idxs, select_model
from data import x_test, y_test
#from utils.draw import draw_images, draw_heatmaps
import utils.logger as logger
#import mlflow
from deeper_base_model import Deeper_Base_BlazePose
from CBAM_model import CBAM_BlazePose
from EXTRA_model import EXTRA_BlazePose
from base_model import BlazePose
from VIT_model import VIT_BlazePose
from ViT2_model import VIT2_BlazePose
from HEATMAP_model import HEATMAP_BlazePose
from utils.losses import loss_func_bce_negative_joint
import utils.metrics as metrics

assert tf.__version__ == "2.18.0", f"Wrong tensorflow version being used. Expected 2.18.0, Received {tf.__version__}"

loss_func_mse = tf.keras.losses.MeanSquaredError()
loss_func_bce = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


def load_model():
    model = None
    weight_filepath = ""
    if select_model == 0:
        model = Deeper_Base_BlazePose().call()
        weight_filepath = "model.weights.h5"
    elif select_model == 1:
        print("Using CBAM_BlazePose")
        model = CBAM_BlazePose().call()
        weight_filepath = "CBAM_best_model.weights.h5"
    elif select_model == 2:
        print("Using Extra_BlazePose")
        model =  EXTRA_BlazePose().call()
        weight_filepath = "EXTRA_best_model.weights.h5"
    elif select_model == 3:
        print("Using VIT_BlazePose")
        weight_filepath = "VIT_model.weights.h5"
        model = VIT_BlazePose().call()
    elif select_model == 4:
        print("Using VIT2_BlazePose")
        weight_filepath = "VIT2_model.weights.h5"
        model = VIT2_BlazePose().call()
    elif select_model == 5:
        print("Using HEATMAP_BlazePose")
        weight_filepath = "HEATMAP_model.weights.h5"
        model = HEATMAP_BlazePose().call()
    else:
        weight_filepath = "base_model.weights.h5"
        model = BlazePose().call()
    assert model is not None, "Invalid model selected. Change select_model value to something that enters the if-else blocks"
    if select_model == 5:
        model.compile(optimizer, loss=[loss_func_bce, loss_func_mse, loss_func_bce, loss_func_bce_negative_joint], metrics=[None, metrics.PCKMetric(), None, None])
    else:
        model.compile(optimizer, loss=[loss_func_bce, loss_func_mse, loss_func_bce], metrics=[None, metrics.PCKMetric(), None])
    print("Load regression weights", weight_filepath)
    model.load_weights(weight_filepath)
    return model


model = load_model()

res = model.evaluate(x=x_test, y=y_test, batch_size=batch_size, callbacks=[logger.keras_custom_callback])       
print("Test PCK score:", res[-1])
#mlflow.log_metrics({"test_coordinates_pck": res[-1]})

# model.summary(print_fn=logger.print_and_log, show_trainable=True)
tmp_dir = os.path.join(os.getcwd(), "tmp")
os.makedirs(tmp_dir, exist_ok=True)
output_file = os.path.join(tmp_dir, "model_architecture.png")
tf.keras.utils.plot_model(model, to_file=output_file, show_shapes=True, show_layer_activations=True, show_trainable=True)
#mlflow.log_artifact(output_file)

# image_files = draw_images(model, img_idxs=img_idxs)
# for image_file in image_files:
#     mlflow.log_artifact(image_file)  

# image_files = draw_heatmaps(model, img_idxs=img_idxs)
# for image_file in image_files:
#     mlflow.log_artifact(image_file)                            

from video_processor import video_processor
from base_model import Base_BlazePose
from VIT_model import VIT_BlazePose
from CBAM_model import CBAM_BlazePose
from EXTRA_model import EXTRA_BlazePose
import tensorflow as tf
import utils.metrics as metrics
from config import input_video_path, output_video_path, batch_size
from data import prepare_datasets
import utils.logger as logger
import os
import warnings
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
# Suppress Python warnings
warnings.filterwarnings('ignore')

# Suppress numpy warnings
np.seterr(all='ignore')

loss_func_mse = tf.keras.losses.MeanSquaredError()
loss_func_bce = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

x_train, y_train, x_val, y_val, x_test, y_test = prepare_datasets()

print("PCK Scores: ")

base_model = Base_BlazePose().call()
base_model.compile(optimizer, loss=[loss_func_bce, loss_func_mse, loss_func_bce], metrics=[None, metrics.PCKMetric(), None])
base_model.load_weights("base_model.weights.h5")
res = base_model.evaluate(x=x_test, y=y_test, batch_size=batch_size) 
print(f"{base_model.name}: {res[-1]}")

cbam_model = CBAM_BlazePose().call()
cbam_model.compile(optimizer, loss=[loss_func_bce, loss_func_mse, loss_func_bce], metrics=[None, metrics.PCKMetric(), None])
cbam_model.load_weights("CBAM_best_model.weights.h5")
res = cbam_model.evaluate(x=x_test, y=y_test, batch_size=batch_size) 
print(f"{cbam_model.name}: {res[-1]}")

extra_model = EXTRA_BlazePose().call()
extra_model.compile(optimizer, loss=[loss_func_bce, loss_func_mse, loss_func_bce], metrics=[None, metrics.PCKMetric(), None])
extra_model.load_weights("EXTRA_best_model.weights.h5")
res = extra_model.evaluate(x=x_test, y=y_test, batch_size=batch_size) 
print(f"{extra_model.name}: {res[-1]}")

x_train, y_train, x_val, y_val, x_test, y_test = prepare_datasets(heat_size=64)

vit_model = VIT_BlazePose().call()
vit_model.compile(optimizer, loss=[loss_func_bce, loss_func_mse, loss_func_bce], metrics=[None, metrics.PCKMetric(), None])
vit_model.load_weights("VIT_model.weights.h5")
res = vit_model.evaluate(x=x_test, y=y_test, batch_size=batch_size) 
print(f"{vit_model.name}: {res[-1]}")

models = [base_model,vit_model,cbam_model,extra_model]

print("Latency scores: ")
for model in models:
    latency = video_processor(model,input_video_path,output_video_path)
    print(f"{model.name} average processing time per frame: {np.mean(latency)}")



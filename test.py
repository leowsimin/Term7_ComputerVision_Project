import cv2, os
import tensorflow as tf
from tensorflow.python.ops import math_ops, array_ops, gen_math_ops
import numpy as np
import pathlib
from model import BlazePose
from config import epoch_to_test, eval_mode, dataset, use_existing_model_weights, pck_metric, batch_size, img_idxs
from data import x_test, y_test
from utils.draw import draw_images
import utils.logger as logger
# import utils.experiment_tracker
import utils.metrics as metrics
from data import coordinates, visibility, heatmap_set, data, number_images
# import mlflow

def Eclidian2(a, b):
# Calculate the square of Eclidian distance
    assert len(a)==len(b)
    summer = 0
    for i in range(len(a)):
        summer += (a[i] - b[i]) ** 2
    return summer

checkpoint_path_regression = "checkpoints_regression"
loss_func_mse = tf.keras.losses.MeanSquaredError()
loss_func_bce = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

if use_existing_model_weights:
    weight_filepath = "model.weights.h5"
else:
    weight_filepath = os.path.join(checkpoint_path_regression, "models/model_ep{}_val_loss_{val_loss:.2f}.weights.h5".format(epoch_to_test))

model = BlazePose().call()
model.compile(optimizer, loss=[loss_func_bce, loss_func_mse, loss_func_bce], metrics=[None, metrics.PCKMetric(), None])

print("Load regression weights", weight_filepath)
model.load_weights(weight_filepath)

res = model.evaluate(x=x_test, y=y_test, batch_size=batch_size, callbacks=[logger.keras_custom_callback])       
print("Test PCK score:", res[-1])
# mlflow.log_metrics({"test_coordinates_pck": res[-1]})

# model.summary(print_fn=logger.print_and_log, show_trainable=True)
tf.keras.utils.plot_model(model, to_file='/tmp/model_architecture.png', show_shapes=True, show_layer_activations=True, show_trainable=True)
# mlflow.log_artifact('/tmp/model_architecture.png')

image_files = draw_images(model, img_idxs=img_idxs)
# for image_file in image_files:
    # mlflow.log_artifact(image_file)                            

# if dataset == "lsp":
#     coordinates = np.zeros((200, 14, 2)).astype(np.uint8)
#     visibility = np.zeros((200, 14, 1)).astype(np.uint8)

#     for i in range(number_images - 200, number_images, batch_size):
#         if i + batch_size >= number_images:
#             # last batch
#             _, coordinates[(i - 1800):(number_images - 1800)], visibility[(i - 1800):(number_images - 1800)] = model.predict(data[i:number_images], callbacks=[logger.keras_custom_callback])
#         else:
#             # other batches
#             _, coordinates[(i - 1800):(i - 1800 + batch_size)], visibility[(i - 1800):(i - 1800 + batch_size)] = model.predict(data[i:(i + batch_size)], callbacks=[logger.keras_custom_callback])
#         print("=", end="")
#     print(">")

#     if eval_mode:
#         # CALCULATE PCK SCORE
#         y = coordinates.astype(float)
#         label = label[:, :, 0:2].astype(float)

#         score_j = np.zeros(14)
#         total = 0
#         for i in range(number_images - 200, number_images):
#             # validation part
#             pck_h = Eclidian2(label[i][12], label[i][13])
#             for j in range(14):
#                 pck_j = Eclidian2(y[i - 1800][j], label[i][j])
#                 # pck_j <= pck_h * 0.5 --> True
#                 if pck_j <= pck_h * pck_metric:
#                     # True estimation
#                     score_j[j] += 1
#                 total += 1
#         # convert to percentage
#         print(total)
#         score_avg = np.sum(score_j)/total
#         logger.logger.info(f'Average PCK score of images {number_images - 200} to {number_images-1} for each of the {14} keypoints = {score_j}')
#         print(score_j)
#         print("Average = %f%%" % score_avg)
#         logger.logger.info("Average PCK score of all keypoints = %f%%" % score_avg)
#     else:
#         pathlib.Path("result").mkdir(parents=True, exist_ok=True)
#         # GENERATE RESULT IMAGES
#         for t in range(number_images - 200, number_images):
#             skeleton = coordinates[t - 1800]
#             img = data[t].astype(np.uint8)
#             # draw the joints
#             for i in range(14):
#                 cv2.circle(img, center=tuple(skeleton[i][0:2]), radius=2, color=(0, 255, 0), thickness=2)
#             # draw the lines
#             for j in ((13, 12), (12, 8), (12, 9), (8, 7), (7, 6), (9, 10), (10, 11), (2, 3), (2, 1), (1, 0), (3, 4), (4, 5)):
#                 cv2.line(img, tuple(skeleton[j[0]][0:2]), tuple(skeleton[j[1]][0:2]), color=(0, 0, 255), thickness=1)
#             # solve the mid point of the hips
#             cv2.line(img, tuple(skeleton[12][0:2]), tuple(skeleton[2][0:2] // 2 + skeleton[3][0:2] // 2), color=(0, 0, 255), thickness=1)

#             cv2.imwrite("./result/lsp_%d.jpg"%t, img)
# else:
#     number_images = 10000
#     coordinates = np.zeros((1000, 14, 2)).astype(np.uint8)
#     visibility = np.zeros((1000, 14, 1)).astype(np.uint8)
#     batch_size = 20
#     for i in range(number_images - 1000, number_images, batch_size):
#         if i + batch_size >= number_images:
#             # last batch
#             _, coordinates[(i - 9000):(number_images - 9000)], visibility[(i - 9000):(number_images - 9000)] = model.predict(data[i:number_images])
#         else:
#             # other batches
#             _, coordinates[(i - 9000):(i - 9000 + batch_size)], visibility[(i - 9000):(i - 9000 + batch_size)] = model.predict(data[i:(i + batch_size)])
#         print("=", end="")
#     print(">")

#     if eval_mode:
#         # CALCULATE PCK SCORE
#         y = coordinates.astype(float)
#         label = label[:, :, 0:2].astype(float)
#         score_j = np.zeros(14)
#         for i in range(number_images - 1000, number_images):
#             # validation part
#             pck_h = Eclidian2(label[i][12], label[i][13])
#             for j in range(14):
#                 pck_j = Eclidian2(y[i - 9000][j], label[i][j])
#                 # pck_j <= pck_h * 0.5 --> True
#                 if pck_j <= pck_h * pck_metric:
#                     # True estimation
#                     score_j[j] += 1
#         # convert to percentage
#         score_j = score_j * 0.1
#         score_avg = sum(score_j) / 14
#         print(score_j)
#         print("Average = %f%%" % score_avg)
#     else:
#         pathlib.Path("result").mkdir(parents=True, exist_ok=True)
#         # GENERATE RESULT IMAGES
#         for t in range(number_images - 1000, number_images):
#             skeleton = coordinates[t - 9000]
#             img = data[t].astype(np.uint8)
#             # draw the joints
#             for i in range(14):
#                 cv2.circle(img, center=tuple(skeleton[i][0:2]), radius=2, color=(0, 255, 0), thickness=2)
#             # draw the lines
#             for j in ((13, 12), (12, 8), (12, 9), (8, 7), (7, 6), (9, 10), (10, 11), (2, 3), (2, 1), (1, 0), (3, 4), (4, 5)):
#                 cv2.line(img, tuple(skeleton[j[0]][0:2]), tuple(skeleton[j[1]][0:2]), color=(0, 0, 255), thickness=1)
#             # solve the mid point of the hips
#             cv2.line(img, tuple(skeleton[12][0:2]), tuple(skeleton[2][0:2] // 2 + skeleton[3][0:2] // 2), color=(0, 0, 255), thickness=1)

#             cv2.imwrite("./result/lsp_%d.jpg"%t, img)

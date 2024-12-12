import cv2, os
import tensorflow as tf
import numpy as np

# import pathlib
from model import BlazePose
from config import epoch_to_test, eval_mode, dataset, use_existing_model_weights

# from data import data, label


def Eclidian2(a, b):
    # Calculate the square of Eclidian distance
    assert len(a) == len(b)
    summer = 0
    for i in range(len(a)):
        summer += (a[i] - b[i]) ** 2
    return summer


checkpoint_path_regression = "checkpoints_regression"
loss_func_mse = tf.keras.losses.MeanSquaredError()
loss_func_bce = tf.keras.losses.BinaryCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

weight_filepath = "model.weights.h5"  # change to whatever weight file


model = BlazePose().call()
# model.compile(optimizer, loss=[loss_func_bce, loss_func_mse, loss_func_bce])

print(
    "Load regression weights",
    os.path.join(
        checkpoint_path_regression, "models/model_ep{}.weights.h5".format(epoch_to_test)
    ),
)
model.load_weights(weight_filepath)

video_file_path = "goblet-squad.mp4"
video = cv2.VideoCapture(video_file_path)

result_save_path = "./processed_frames"
os.makedirs(result_save_path, exist_ok=True)

if not video.isOpened():
    print("Error: Could not open the video file.")
    exit()

frameCounter = 0

while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_frame = cv2.resize(rgb_frame, (256, 256))
    input_frame = np.expand_dims(input_frame, axis=0)

    heatmap, coordinates, visibility = model.predict(input_frame)

    if coordinates.shape[0] > 0:  # Check if valid landmarks are detected
        # Rescale coordinates to the original frame dimensions
        height, width, _ = frame.shape
        scaled_coordinates = coordinates[0] * [width / 256, height / 256]

        # Draw landmarks on the original frame
        for joint in scaled_coordinates:
            x, y = int(joint[0]), int(joint[1])
            cv2.circle(
                frame, (x, y), 5, (0, 255, 0), -1
            )  # Draw a green dot for each joint

        # Save the processed frame to the result save path
        save_path = os.path.join(result_save_path, f"frame_{frameCounter:04d}.jpg")
        cv2.imwrite(save_path, frame)

        print(f"Processed frame {frameCounter}, saved to {save_path}")
    else:
        print(f"No landmarks detected for frame {frameCounter}")

    frameCounter += 1

video.release()
print("Processing complete.")

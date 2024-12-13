import cv2
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time  # Import for measuring processing time

from base_model import BlazePose
from config import epoch_to_test, input_video_path, output_video_path, select_model

# Set parameters
checkpoint_path_regression = "checkpoints_regression"
loss_func_mse = tf.keras.losses.MeanSquaredError()
loss_func_bce = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Load model and weights
weight_filepath = "model.weights.h5"  # Replace with the correct weight file path
model = BlazePose().call()
model.compile(optimizer, loss=[loss_func_bce, loss_func_mse, loss_func_bce])

print("Load regression weights", os.path.join(checkpoint_path_regression, "models/model_ep{}.weights.h5".format(epoch_to_test)))
model.load_weights(weight_filepath)

# Input and output video paths
video_file_path = input_video_path
video = cv2.VideoCapture(video_file_path)

result_video_path = output_video_path

if not video.isOpened():
    print("Error: Could not open the video file.")
    exit()

# Video properties
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(video.get(cv2.CAP_PROP_FPS))
frame_counter = 0

# Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out_video = cv2.VideoWriter(result_video_path, fourcc, frame_rate, (frame_width, frame_height))

# Timing variables
cumulative_time = []
frame_numbers = []

# Process video frames
start_time = time.time()  # Start timing the entire process

while video.isOpened():
    success, frame = video.read()
    if not success:
        break

    frame_start_time = time.time()  # Start timing the current frame

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_frame = cv2.resize(rgb_frame, (256, 256))
    input_frame = np.expand_dims(input_frame, axis=0)

    if select_model == 5:
        heatmap, coordinates, visibility, _ = model.predict(input_frame)

    else:
        heatmap, coordinates, visibility = model.predict(input_frame)

    if coordinates.shape[0] > 0:  # Check if valid landmarks are detected
        # Rescale coordinates to the original frame dimensions
        height, width, _ = frame.shape
        scaled_coordinates = coordinates[0] * [width / 256, height / 256]

        # Draw landmarks on the original frame
        for joint in scaled_coordinates:
            x, y = int(joint[0]), int(joint[1])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw a green dot for each joint

    else:
        print(f"No landmarks detected for frame {frame_counter}")

    # Write the processed frame to the output video
    out_video.write(frame)

    # Record timing for the current frame
    frame_end_time = time.time()
    processing_time = frame_end_time - start_time
    cumulative_time.append(processing_time)
    frame_numbers.append(frame_counter)

    frame_counter += 1

video.release()
out_video.release()

print("Processing complete.")

# Plot the time taken
plt.figure(figsize=(10, 6))
plt.plot(frame_numbers, cumulative_time, marker='o', linestyle='-', color='b', label="Cumulative Time")
plt.title("Video Processing Time vs. Number of Frames")
plt.xlabel("Number of Frames")
plt.ylabel("Cumulative Processing Time (seconds)")
plt.grid(True)
plt.legend()
plt.savefig("processing_time_graph.png")  # Save the graph
plt.show()

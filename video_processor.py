import cv2
import os
import tensorflow as tf
import numpy as np
import time


def video_processor(model: tf.keras.Model, video_file_path: str, result_video_path: str, output_video=False):
    # Use model name safely
    model_name = getattr(model, 'name', 'unnamed_model')
    result_video_path = f"{model_name}_{os.path.basename(result_video_path)}"

    # Open video file
    video = cv2.VideoCapture(video_file_path)
    if not video.isOpened():
        print(f"Error: Could not open the video file '{video_file_path}'.")
        return

    # Get video properties
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))
    frame_counter = 0

    # Initialize VideoWriter if output_video is True
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
        out_video = cv2.VideoWriter(result_video_path, fourcc, frame_rate, (frame_width, frame_height))

    # Timing variables
    cumulative_time = []
    frame_numbers = []

    try:
        # Process video frames
        start_time = time.time()  # Start timing the entire process

        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            frame_start_time = time.time()  # Start timing the current frame

            # Preprocess the frame for the model
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_frame = cv2.resize(rgb_frame, (256, 256))
            input_frame = np.expand_dims(input_frame, axis=0)

            # Make predictions with the model
            predictions = model.predict(input_frame)

            if len(predictions) == 3:
                heatmap, coordinates, visibility = predictions

                if coordinates.shape[0] > 0:  # Check if valid landmarks are detected
                    # Rescale coordinates to the original frame dimensions
                    height, width, _ = frame.shape
                    scaled_coordinates = coordinates[0] * [width / 256, height / 256]

                    # Draw landmarks on the original frame
                    for joint in scaled_coordinates:
                        x, y = int(joint[0]), int(joint[1])
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw a green dot for each joint
                else:
                    print(f"No landmarks detected for frame {frame_counter}.")
            else:
                print(f"Unexpected model output format for frame {frame_counter}.")

            # Write the processed frame to the output video if enabled
            if output_video:
                out_video.write(frame)

            # Record timing for the current frame
            frame_end_time = time.time()
            processing_time = frame_end_time - frame_start_time
            cumulative_time.append(processing_time)
            frame_numbers.append(frame_counter)

            frame_counter += 1

    finally:
        # Release resources
        video.release()
        if output_video:
            out_video.release()

    total_processing_time = time.time() - start_time
    print(f"Video processing complete for model: {model_name}.")
    print(f"Total processing time: {total_processing_time:.2f} seconds.")
    return cumulative_time

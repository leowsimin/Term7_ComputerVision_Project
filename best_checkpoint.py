import os
import tensorflow as tf
from model import BlazePose
from data import data, heatmap_set, coordinates, visibility, number_images

# Path to checkpoints
checkpoint_folder = "checkpoints_heatmap/models"

# Load validation data
x_val = data[-400:-200]  # Replace with actual validation data slice
y_val = [heatmap_set[-400:-200], coordinates[-400:-200], visibility[-400:-200]]

# Initialize the model
model = BlazePose().call()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=[
        tf.keras.losses.BinaryCrossentropy(),  # Heatmap loss
        tf.keras.losses.MeanSquaredError(),    # Coordinate loss
        tf.keras.losses.BinaryCrossentropy()  # Visibility loss
    ]
)

# Function to evaluate a checkpoint
def evaluate_checkpoint(model, checkpoint_path, x_val, y_val):
    model.load_weights(checkpoint_path)
    loss = model.evaluate(x_val, y_val, verbose=0)  # Evaluate on validation set
    return loss[0]  # Return total loss

# Find the best checkpoint
best_checkpoint = None
best_loss = float("inf")

for filename in os.listdir(checkpoint_folder):
    if filename.endswith(".h5"):
        checkpoint_path = os.path.join(checkpoint_folder, filename)
        print(f"Evaluating {checkpoint_path}...")
        loss = evaluate_checkpoint(model, checkpoint_path, x_val, y_val)
        print(f"Validation loss: {loss}")
        if loss < best_loss:
            best_loss = loss
            best_checkpoint = checkpoint_path

# Output the best checkpoint
print(f"Best checkpoint: {best_checkpoint} with validation loss: {best_loss}")

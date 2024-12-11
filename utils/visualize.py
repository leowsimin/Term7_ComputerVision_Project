import tensorflow as tf
import keras
import numpy as np
import math
import mlflow

img_width, img_height = 256, 256

def visualize(model, layer_name, num_filters):
    layer = model.get_layer(name=layer_name)
    feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)

    def compute_loss(input_image, filter_index):
        activation = feature_extractor(input_image)
        # We avoid border artifacts by only involving non-border pixels in the loss.
        filter_activation = activation[:, 2:-2, 2:-2, filter_index]
        return tf.reduce_mean(filter_activation)
    
    @tf.function
    def gradient_ascent_step(img, filter_index, learning_rate):
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = compute_loss(img, filter_index)
        # Compute gradients.
        grads = tape.gradient(loss, img)
        # Normalize gradients.
        grads = tf.math.l2_normalize(grads)
        img += learning_rate * grads
        return loss, img
    
    def initialize_image():
        # We start from a gray image with some random noise
        img = tf.random.uniform((1, img_width, img_height, 3))
        # ResNet50V2 expects inputs in the range [-1, +1].
        # Here we scale our random inputs to [-0.125, +0.125]
        return (img - 0.5) * 0.25


    def visualize_filter(filter_index):
        # We run gradient ascent for 20 steps
        iterations = 120
        learning_rate = 10.0
        img = initialize_image()
        for iteration in range(iterations):
            loss, img = gradient_ascent_step(img, filter_index, learning_rate)

        print(loss)
        # Decode the resulting input image
        img = deprocess_image(img[0].numpy())
        return loss, img


    def deprocess_image(img):
        # Normalize array: center on 0., ensure variance is 0.15
        img -= img.mean()
        img /= img.std() + 1e-5
        img *= 0.15

        # Center crop
        img = img[25:-25, 25:-25, :]

        # Clip to [0, 1]
        img += 0.5
        img = np.clip(img, 0, 1)

        # Convert to RGB array
        img *= 255
        img = np.clip(img, 0, 255).astype("uint8")
        return img
    
    all_imgs = []
    for filter_index in range(num_filters):
        print("Processing filter %d" % (filter_index,))
        loss, img = visualize_filter(filter_index)
        all_imgs.append(img)

    # Build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    n = int(math.ceil(math.sqrt(num_filters)))
    cropped_width = img_width - 25 * 2
    cropped_height = img_height - 25 * 2
    width = n * cropped_width + (n - 1) * margin
    height = n * cropped_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # Fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            if i * n + j >= num_filters:
                break
            img = all_imgs[i * n + j]
            stitched_filters[
                (cropped_width + margin) * i : (cropped_width + margin) * i + cropped_width,
                (cropped_height + margin) * j : (cropped_height + margin) * j
                + cropped_height,
                :,
            ] = img
    keras.utils.save_img(f"tmp/{layer_name}_filters.png", stitched_filters)
    mlflow.log_artifact(f"tmp/{layer_name}_filters.png")
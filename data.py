import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from config import num_joints, dataset, num_images, train_split, val_split, test_split

def prepare_datasets(heat_size=128):
    def getGaussianMap(joint=(16, 16), heat_size=128, sigma=2):
        heatmap = np.zeros((heat_size, heat_size), dtype=np.float32)
        tmp_size = sigma * 3
        ul = [int(joint[0] - tmp_size), int(joint[1] - tmp_size)]
        br = [int(joint[0] + tmp_size + 1), int(joint[1] + tmp_size + 1)]
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma ** 2)))
        g_x = max(0, -ul[0]), min(br[0], heat_size) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heat_size) - ul[1]
        img_x = max(0, ul[0]), min(br[0], heat_size)
        img_y = max(0, ul[1]), min(br[1], heat_size)
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return heatmap

    annotations = loadmat(f"./dataset/{dataset}/joints.mat")

    if dataset == "lsp":
        number_images = num_images or 2000
        label = annotations["joints"].swapaxes(0, 2)
    else:
        number_images = num_images or 10000
        label = annotations["joints"].swapaxes(0, 1).swapaxes(0, 2)
    
    label = label[:number_images, :, :]

    # Prepare flipped labels
    #flipped_labels = label.copy()
    #flipped_labels[:, :, 0] = 256 - flipped_labels[:, :, 0]  # Assuming the images are resized to 256x256

    data = np.zeros([number_images, 256, 256, 3])
    #flipped_data = np.zeros([number_images, 256, 256, 3])

    heatmap_set = np.zeros((number_images, heat_size, heat_size, num_joints), dtype=np.float32)
    #flipped_heatmap_set = np.zeros((number_images, heat_size, heat_size, num_joints), dtype=np.float32)

    for i in range(number_images):
        if dataset == "lsp":
            FileName = f"./dataset/{dataset}/images/im{i + 1:04d}.jpg"
        else:
            FileName = f"./dataset/{dataset}/images/im{i + 1:05d}.jpg"

        img = tf.io.read_file(FileName)
        img = tf.image.decode_image(img)
        img_shape = img.shape

        label[i, :, 0] *= (256 / img_shape[1])
        label[i, :, 1] *= (256 / img_shape[0])
        #flipped_labels[i, :, 0] *= (256 / img_shape[1])
        #flipped_labels[i, :, 1] *= (256 / img_shape[0])

        data[i] = tf.image.resize(img, [256, 256])
        #flipped_data[i] = tf.image.flip_left_right(data[i])

        for j in range(num_joints):
            _joint = (label[i, j, 0:2] // (256 / heat_size)).astype(np.uint16)
            heatmap_set[i, :, :, j] = getGaussianMap(joint=_joint, heat_size=heat_size, sigma=4)

            #flipped_joint = (flipped_labels[i, j, 0:2] // (256 / heat_size)).astype(np.uint16)
            #flipped_heatmap_set[i, :, :, j] = getGaussianMap(joint=flipped_joint, heat_size=heat_size, sigma=4)

    coordinates = label[:, :, 0:2]
    #flipped_coordinates = flipped_labels[:, :, 0:2]

    visibility = label[:, :, 2:]
    #flipped_visibility = flipped_labels[:, :, 2:]

    #data = np.concatenate([data, flipped_data], axis=0)
    #heatmap_set = np.concatenate([heatmap_set, flipped_heatmap_set], axis=0)
    #coordinates = np.concatenate([coordinates, flipped_coordinates], axis=0)
    #visibility = np.concatenate([visibility, flipped_visibility], axis=0)

    #number_images = data.shape[0]

    train_start, train_end = 0, int(train_split * number_images)
    val_start, val_end = int(train_split * number_images), int((train_split + val_split) * number_images)
    test_start, test_end = int((train_split + val_split) * number_images), number_images

    x_train = data[train_start:train_end]
    y_train = [heatmap_set[train_start:train_end], coordinates[train_start:train_end], visibility[train_start:train_end]]

    x_val = data[val_start:val_end]
    y_val = [heatmap_set[val_start:val_end], coordinates[val_start:val_end], visibility[val_start:val_end]]

    x_test = data[test_start:test_end]
    y_test = [heatmap_set[test_start:test_end], coordinates[test_start:test_end], visibility[test_start:test_end]]

    return x_train, y_train, x_val, y_val, x_test, y_test

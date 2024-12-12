#!~/miniconda3/envs/tf2/bin/python
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from config import (
    num_joints,
    dataset,
    num_images,
    train_split,
    val_split,
    test_split,
    heat_size,
    negative_joint_factor
)

# import mlflow


# guassian generation
def getGaussianMap(joint=(16, 16), heat_size=128, sigma=2):
    # by default, the function returns a gaussian map with range [0, 1] of typr float32
    heatmap = np.zeros((heat_size, heat_size), dtype=np.float32)
    tmp_size = sigma * 3
    ul = [int(joint[0] - tmp_size), int(joint[1] - tmp_size)]
    br = [int(joint[0] + tmp_size + 1), int(joint[1] + tmp_size + 1)]
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma**2)))
    g.shape
    # usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], heat_size) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], heat_size) - ul[1]
    # image range
    img_x = max(0, ul[0]), min(br[0], heat_size)
    img_y = max(0, ul[1]), min(br[1], heat_size)
    heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]] = g[
        g_y[0] : g_y[1], g_x[0] : g_x[1]
    ]
    return heatmap

joint_names_dict = {
    0: "Right ankle",
    1: "Right knee",
    2: "Right hip",
    3: "Left hip",
    4: "Left knee",
    5: "Left ankle",
    6: "Right wrist",
    7: "Right elbow",
    8: "Right shoulder",
    9: "Left shoulder",
    10: "Left elbow",
    11: "Left wrist",
    12: "Neck",
    13: "Head top"
}

joint_order = [5, 0, 4, 1, 3, 2, 11, 6, 10, 7, 9, 8, 12, 13]


# read annotations
annotations = loadmat("./dataset/" + dataset + "/joints.mat")
number_images = num_images
if dataset == "lsp":
    # LSP
    number_images = number_images or 2000
    print("Num images:", number_images)
    label = annotations["joints"].swapaxes(0, 2)  # shape (3, 14, 2000) -> (2000, 14, 3)
else:
    # LSPET
    number_images = number_images or 10000
    label = annotations["joints"].swapaxes(
        0, 1
    )  # shape (14, 3, 10000) -> (3, 14, 10000)
    label = label.swapaxes(0, 2)  # shape (3, 14, 10000) -> (10000, 14, 3)
label = label[:number_images, :, :]

# read images
data = np.zeros([number_images, 256, 256, 3])
heatmap_set = np.zeros(
    (number_images, heat_size, heat_size, num_joints), dtype=np.float32
)
print("Reading dataset...")
for i in range(number_images):
    if dataset == "lsp":
        # lsp
        FileName = "./dataset/" + dataset + "/images/im%04d.jpg" % (i + 1)
    else:
        # lspet
        FileName = "./dataset/" + dataset + "/images/im%05d.jpg" % (i + 1)
    img = tf.io.read_file(FileName)
    img = tf.image.decode_image(img)
    img_shape = img.shape
    # Attention here img_shape[0] is height and [1] is width
    label[i, :, 0] *= 256 / img_shape[1]
    label[i, :, 1] *= 256 / img_shape[0]
    data[i] = tf.image.resize(img, [256, 256])
    # generate heatmap set
    heatmaps = {}
    for j in range(num_joints):
        _joint = (label[i, j, 0:2] // (256 / heat_size)).astype(np.uint16)
        heatmaps[j] = heatmaps.get(j, getGaussianMap(joint = _joint, heat_size = heat_size, sigma = 4))

        # apply negative values for opposing joint
        if negative_joint_factor != 0 and negative_joint_dict[j] is not None:
            j_neg = negative_joint_dict[j]
            if j_neg in heatmaps:
                negative_heatmap = heatmaps.get(j_neg)
            else:
                _negative_joint = (label[i, j_neg, 0:2] // (256 / heat_size)).astype(np.uint16)
                negative_heatmap = getGaussianMap(joint = _negative_joint, heat_size = heat_size, sigma = 4)
                heatmaps[j_neg] = negative_heatmap
            heatmap_set[i, :, :, j] = heatmaps[j] - negative_joint_factor * negative_heatmap
        else:
            heatmap_set[i, :, :, j] = heatmaps[j]

    # print status

coordinates = label[:, :, 0:2]
visibility = label[:, :, 2:]

print("Done.")

print("Splitting data.")
train_start, train_end = 0, int(train_split * number_images)
val_start, val_end = int(train_split * number_images), int(
    (train_split + val_split) * number_images
)
test_start, test_end = int((train_split + val_split) * number_images), number_images

x_train = data[train_start:train_end]
y_train = [
    heatmap_set[train_start:train_end],
    coordinates[train_start:train_end],
    visibility[train_start:train_end],
]
x_val = data[val_start:val_end]
y_val = [
    heatmap_set[val_start:val_end],
    coordinates[val_start:val_end],
    visibility[val_start:val_end],
]
x_test = data[test_start:test_end]
y_test = [
    heatmap_set[test_start:test_end],
    coordinates[test_start:test_end],
    visibility[test_start:test_end],
]

# try:
#     mlflow_dataset = mlflow.data.from_numpy(x_train, targets=y_train[1]) # log coord target only
#     mlflow.log_input(mlflow_dataset, context="training")
#     mlflow_dataset = mlflow.data.from_numpy(x_val, targets=y_val[1])
#     mlflow.log_input(mlflow_dataset, context="validation")
#     mlflow_dataset = mlflow.data.from_numpy(x_test, targets=y_test[1])
#     mlflow.log_input(mlflow_dataset, context="test")
# except:
#     pass

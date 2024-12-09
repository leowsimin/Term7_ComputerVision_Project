num_joints = 14  # lsp dataset
num_images = 10000  # above 400, below 2000

batch_size = 200
total_epoch = 100
dataset = "lspet"  # "lsp"
train_split, val_split, test_split = 0.6, 0.2, 0.2  # split
heat_size = 128

# Train mode: 0-heatmap, 1-regression
train_mode = 0
pretrain = 1  # set to 1 to pretrain on lspet

# Eval mode: 0-output image, 1-pck score
eval_mode = 1
pck_metric = (
    0.5  # standard; point is correct if distance to gt < 50% of person's head size
)
img_idxs_lsp = [1800, 1906, 1981, 1995]  # which images to draw when predicting
img_idxs_lspet = [1, 2, 3, 4]  # which images to draw when predicting

continue_train = 0
continue_train_from_filename = ""
best_pre_train_filename = ""

# for test only
epoch_to_test = 55
use_existing_model_weights = 1

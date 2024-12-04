num_joints = 14     # lsp dataset

batch_size = 8
total_epoch = 75
dataset = "lsp" # "lspet"

# Train mode: 0-heatmap, 1-regression
train_mode = 0

# Eval mode: 0-output image, 1-pck score
eval_mode = 0

continue_train = 0
continue_train_from_filename = "model_ep49_val_loss_785.84.weights.h5"
best_pre_train_filename = ""

# for test only
epoch_to_test = 55
use_existing_model_weights = 1
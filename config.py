num_joints = 14     # lsp dataset

batch_size = 32
total_epoch = 75
dataset = "lsp" # "lspet"

# Train mode: 0-heatmap, 1-regression
train_mode = 0

# Eval mode: 0-output image, 1-pck score
eval_mode = 1

continue_train = 0

best_pre_train = None # num of epoch where the training loss drops but testing accuracy achieve the optimal

# for test only
epoch_to_test = 55
use_existing_model_weights = 1
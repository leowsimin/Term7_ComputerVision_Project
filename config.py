num_joints = 14     # lsp dataset

batch_size = 200
total_epoch = 100 # added more epochs to see if the model can improve - gradient have last opportunities to update as a result of the larger batch size less stable with a larger batch 
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
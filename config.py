num_joints = 14     # lsp dataset
num_images = 2000 # above 400, below 2000

batch_size = 16
total_epoch = 50
dataset = "lsp" # "lspet"
train_split, val_split, test_split = 0.6, 0.2, 0.2 # split
heat_size = 128

# Train mode: 0-heatmap, 1-regression
train_mode = 0

# Eval mode: 0-output image, 1-pck score
eval_mode = 1
pck_metric = 0.5 # standard; point is correct if distance to gt < 50% of person's head size
img_idxs=[1800, 1906, 1981, 1995] # which images to draw when predicting

continue_train = 0
continue_train_from_filename = "model_ep20.weights.h5"
best_pre_train_filename = 50

# for test only
epoch_to_test = 55
use_existing_model_weights = True

# for video processing
input_video_path = "squat_1.mp4"
output_video_path = "output_video.mp4"

# for model selection
select_model = 0




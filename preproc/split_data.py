import os
import random


DATA_PATH = "/mnt/cluster/workspaces/lichenyan/cXR-DNN/data/ply"
OUTPUT_PATH = "/mnt/cluster/workspaces/lichenyan/cXR-DNN/data/split"
os.makedirs(OUTPUT_PATH, exist_ok=True)

train_ratio = 0.6 
val_ratio = 0.2
test_ratio = 0.2

# list all ply files in data_path
ply_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.ply')]
random.shuffle(ply_files)

# split ply files into train and test
train_files = ply_files[:int(len(ply_files) * train_ratio)]
val_files = ply_files[int(len(ply_files) * train_ratio):int(len(ply_files) * (train_ratio + val_ratio))]
test_files = ply_files[int(len(ply_files) * (train_ratio + val_ratio)):]

# save train and test files to output_path
with open(os.path.join(OUTPUT_PATH, 'train.txt'), 'w') as f:
    for file in train_files:
        f.write(file + '\n')
with open(os.path.join(OUTPUT_PATH, 'val.txt'), 'w') as f:
    for file in val_files:
        f.write(file + '\n')
with open(os.path.join(OUTPUT_PATH, 'test.txt'), 'w') as f:
    for file in test_files:
        f.write(file + '\n')
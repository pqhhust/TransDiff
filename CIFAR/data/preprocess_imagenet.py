# import os
# import pickle
# import numpy as np
# from PIL import Image

# def unpickle(file):
#     with open(file, 'rb') as fo:
#         data = pickle.load(fo, encoding='latin1')
#     return data

# def process_split(in_dir, out_dir):
#     """
#     Process a folder full of pickled batch files.
#     Images are assumed to be in a flat array that reshapes to (3, 32, 32).
#     If labels are 1-indexed (i.e. minimum label >= 1), they are converted to 0-indexed.
#     Each image is saved under out_dir/<label>/.
#     """
#     os.makedirs(out_dir, exist_ok=True)
#     overall_labels = []
#     # First loop to gather all labels to decide if subtraction is needed.
#     for filename in sorted(os.listdir(in_dir)):
#         batch = unpickle(os.path.join(in_dir, filename))
#         overall_labels.extend(batch['labels'])
#     # If labels are 1-indexed, subtract 1.
#     subtract = min(overall_labels) >= 1

#     count = {}
#     for filename in sorted(os.listdir(in_dir)):
#         batch_path = os.path.join(in_dir, filename)
#         batch = unpickle(batch_path)
#         labels = batch['labels']
#         nb_data = len(labels)
#         data = batch['data'].reshape((nb_data, 3, 32, 32)).transpose(0, 2, 3, 1)
        
#         for i in range(nb_data):
#             raw_label = labels[i]
#             label = raw_label - 1 if subtract else raw_label
#             count[label] = count.get(label, 0) + 1
            
#             label_dir = os.path.join(out_dir, str(label))
#             os.makedirs(label_dir, exist_ok=True)
#             out_path = os.path.join(label_dir, f"{count[label]}.png")
#             Image.fromarray(data[i].astype('uint8')).save(out_path)
#     print(f"Processed {in_dir}")
#     print("Counts per class:", count)

# if __name__ == '__main__':
#     # Update these paths to your actual pickled data directories.
#     base_in = 'IMAGENET1K'  # Contains 'train' and 'val' folders
#     base_out = 'IMAGENET1K_32'
#     splits = ['train', 'val']
#     for split in splits:
#         in_dir = os.path.join(base_in, split)
#         out_dir = os.path.join(base_out, split)
#         if os.path.exists(in_dir):
#             process_split(in_dir, out_dir)
#         else:
#             print(f"Input directory {in_dir} does not exist. Skipping.")

### extract synset_words.txt
# import scipy.io

# # Load the meta.mat file
# meta = scipy.io.loadmat("IMAGENET1K/ILSVRC2012_devkit_t12/data/meta.mat")

# # Extract synset words (make sure they are strings)
# synset_label_pairs = [(entry[0][1][0], entry[0][2][0]) for entry in meta["synsets"]]
# # for entry in meta["synsets"]:
# #     print(entry)
# #     print(entry[0][1][0])
# #     print(entry[0][2][0])
# #     break

# # Print the structure of the file to understand its contents
# # print(meta.keys())  # List the keys of the mat file to inspect its contents
# # print(meta['synsets'])  # Print the synsets data to inspect

# # # Save to synset_words.txt
# with open("IMAGENET1K/synset_words.txt", "w") as f:
#     for synset, label in synset_label_pairs:
#         # Write each pair in the format "synset_id label"
#         f.write(f"{synset} {label}\n")

# # print("✅ synset_words.txt created successfully!")



### create val_labels.txt

# with open('./IMAGENET1K/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt', 'r') as gt_file:
#     labels = gt_file.readlines()

# with open('./IMAGENET1K/val_labels.txt', 'w') as out_file:
#     for i, label in enumerate(labels):
#         image_name = f'ILSVRC2012_val_{i+1:08d}.JPEG'
#         out_file.write(f'{image_name} {label}')



### reorganize val dataset

import os
import shutil

# Paths
val_dir = "./IMAGENET1K/val"  # Update with your actual path
labels_file = "./IMAGENET1K/val_labels.txt"
synset_file = "./IMAGENET1K/synset_words.txt"

# Step 1: Read synset_words.txt to map class ID → WordNet synset
class_map = {}  # Maps class index to synset (e.g., {65: "n01440764"})
with open(synset_file, "r") as f:
    synsets = [line.strip().split()[0] for line in f.readlines()]
class_map = {i + 1: synset for i, synset in enumerate(synsets)}  # Class IDs are 1-based

# Step 2: Read val_labels.txt to map image → class
image_labels = {}  # Maps image name to synset
with open(labels_file, "r") as f:
    for line in f:
        image, class_id = line.strip().split()
        synset = class_map[int(class_id)]
        image_labels[image] = synset

# Step 3: Create folders and move images
for image, synset in image_labels.items():
    class_folder = os.path.join(val_dir, synset)
    os.makedirs(class_folder, exist_ok=True)  # Create folder if not exists
    src_path = os.path.join(val_dir, image)
    dst_path = os.path.join(class_folder, image)
    if os.path.exists(src_path):  # Avoid errors if file is missing
        shutil.move(src_path, dst_path)

print("✅ Validation set reorganized successfully!")
# # find ./IMAGENET1K/val -mindepth 1 -maxdepth 1 -type d | wc -l


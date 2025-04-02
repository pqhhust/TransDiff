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
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

import os
import shutil

val_dir = './val'  # Thư mục chứa ảnh val gốc
val_txt = './val.txt'        # File ánh xạ ảnh -> nhãn
output_val_dir = 'IMAGENET1K_FULL/val'  # Thư mục mới cho val với cấu trúc synset
os.makedirs(output_val_dir, exist_ok=True)

# Đọc file synset_words.txt để lấy danh sách synset
with open('./synset_words.txt', 'r') as f:
    synsets = [line.split()[0] for line in f]

# Đọc val.txt để lấy mapping từ tên ảnh sang synset
with open(val_txt, 'r') as f:
    lines = f.readlines()
    img_to_synset = {}
    for line in lines:
        img_name, label = line.strip().split()
        synset = synsets[int(label)]  # Ánh xạ từ nhãn số sang synset
        img_to_synset[img_name] = synset

# Tổ chức ảnh vào folder synset
for img_name in os.listdir(val_dir):
    if img_name in img_to_synset:
        synset = img_to_synset[img_name]
        synset_dir = os.path.join(output_val_dir, synset)
        os.makedirs(synset_dir, exist_ok=True)
        shutil.copy(os.path.join(val_dir, img_name), os.path.join(synset_dir, img_name))
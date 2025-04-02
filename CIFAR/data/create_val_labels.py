with open('./IMAGENET1K/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt', 'r') as gt_file:
    labels = gt_file.readlines()

with open('./IMAGENET1K/val_labels.txt', 'w') as out_file:
    for i, label in enumerate(labels):
        image_name = f'ILSVRC2012_val_{i+1:08d}.JPEG'
        out_file.write(f'{image_name} {label}')
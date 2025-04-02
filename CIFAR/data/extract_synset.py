import scipy.io

# Load the meta.mat file
meta = scipy.io.loadmat("IMAGENET1K/ILSVRC2012_devkit_t12/data/meta.mat")

# Extract synset words (make sure they are strings)
synset_label_pairs = [(entry[0][1][0], entry[0][2][0]) for entry in meta["synsets"]]
# for entry in meta["synsets"]:
#     print(entry)
#     print(entry[0][1][0])
#     print(entry[0][2][0])
#     break

# Print the structure of the file to understand its contents
# print(meta.keys())  # List the keys of the mat file to inspect its contents
# print(meta['synsets'])  # Print the synsets data to inspect

# # Save to synset_words.txt
with open("IMAGENET1K/synset_words.txt", "w") as f:
    for synset, label in synset_label_pairs:
        # Write each pair in the format "synset_id label"
        f.write(f"{synset} {label}\n")

# print("✅ synset_words.txt created successfully!")
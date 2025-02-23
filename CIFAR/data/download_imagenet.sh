
# wget -c https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
# tar -xvf ./ILSVRC2012_img_train.tar -C ./train
# rm ./ILSVRC2012_img_train.tar
# wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
# tar -xvf ./ILSVRC2012_img_val.tar -C ./val
# rm ./ILSVRC2012_img_val.tar
# wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar
# tar -xvf ./ILSVRC2012_img_test_v10102019.tar 
# rm ./ILSVRC2012_img_test_v10102019.tar
# find ./test -type f -name "*.JPEG" | wc -l
cd ./IMAGENET1K/train
for file in *.tar; do
    folder="${file%.tar}"  # Remove .tar extension to get folder name
    mkdir -p "$folder"      # Create a folder with the same name
    tar -xf "$file" -C "$folder"  # Extract images into that folder
    # rm "$file"  # (Optional) Remove the class .tar file after extraction
done
cd ..
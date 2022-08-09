# Steps to recreate data
0. `bash download_files.sh` will download and unzip the tar files
1. `python clean_and_split_data.py` to sample data for training, dev, and testing
2. `bash copy_to_data_folder.sh` to move them to the `data` folder
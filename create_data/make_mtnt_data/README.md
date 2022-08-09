# Steps to recreate data
0. `bash download.sh` will download the tar files
1. `bash preprocess.sh` will prepare the data for M2M-100 to be similar to its training data and re-write the original data
2. `python sample_train_test_data.py` to sample data for training, dev, and testing
3. `bash copy_to_data_folder.sh` to move them to the `data` folder
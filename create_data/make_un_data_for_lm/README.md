# Steps to recreate data
0. `bash download_files.sh` will download the tar files. NOTE: if this provides an error, try downloading them from the link manually
1. `unzip_all.sh` will combine the tar.gz files and unzip them into `UNv1.0-TEI`
2. `python gather_exclusive_paths.py` to gather only unique sentences from each language and create text files of them
3. `python sample_train_test_data.py` to sample data for training, dev, and testing
4. `bash copy_to_data_folder.sh` to move them to the `data` folder
1. place your .gz files under preprocess/data directory (download from https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
2. run meta_process.py to create meta_data.json (you should modify the root directory)
3. run seq_process.py to create {train, valid, test}_data.json (you should modify the root directory)
4. train_bc.py automatically generates files in path_train, path_eval but recommended to download it from here https://drive.google.com/drive/folders/10CeCEAzdGKbM3695VLDbLgVfGVbuNDkq?usp=drive_link (takes very long time to generate!)
5. before running train_bc.py you should modify arguments (path, metadata_path, checkpoint_dir)

additional : my conda environment is in requirements.txt
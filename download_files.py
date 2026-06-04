import kagglehub

def download_files():
    # https://www.kaggle.com/datasets/zfturbo/audioset?select=train.csv
    path = kagglehub.dataset_download("zfturbo/audioset")
    print("Path to dataset files:", path)

    # https://www.kaggle.com/datasets/chrisfilo/urbansound8k?select=UrbanSound8K.csv
    path = kagglehub.dataset_download("chrisfilo/urbansound8k")
    print("Path to dataset files:", path)
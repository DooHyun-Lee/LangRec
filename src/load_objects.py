from base import RecommendData
import os

def load_rawdataset(path):
    rd = RecommendData(path)
    import ipdb
    ipdb.set_trace()

PATH = os.path.join('/home/doolee13/LangRec/preprocess', 'train_data.json')

load_rawdataset(PATH)
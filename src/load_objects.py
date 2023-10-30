from base import RecommendData
from data import RecommendListDataset
import os

PATH = os.path.join('/home/doolee13/LangRec/preprocess', 'train_data.json')

rd = RecommendData(PATH)
train_raw = RecommendListDataset(rd, max_len=1024)
import ipdb
ipdb.set_trace()
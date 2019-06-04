import warnings
warnings.filterwarnings('ignore')

from flask import Flask
app = Flask(__name__)
app.config.from_object('config')

import pickle
from app.sentiment import *

filename = 'app/data/supervised.sav'
cls_task2 = pickle.load(open(filename, 'rb'))

with open("app/data/sentiment_task2.pickle", "rb") as f:
    sentiment_task2 = pickle.load(f)

from app import routes







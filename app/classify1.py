# -----------------------------
# Sentiment Classfier
# Author: Guanghao Chen
# Date: May 23 2019
# -----------------------------

from collections import defaultdict
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from app.sentiment import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import shap

def softmax(w):
    e = np.exp(w)
    return e / np.sum(e)



class Sentiment_Classfier():
    def __init__(self):
        self.l2_penal = 1.
        self.cls = LogisticRegression(random_state=0,
                                      solver='lbfgs',
                                      max_iter=10000,
                                      C=self.l2_penal,
                                      class_weight='balanced')

    def train_classifier(self, X, y):
        self.cls.fit(X, y)

    def evaluate(self, Xt, yt):
        yp = self.cls.predict(Xt)
        acc = metrics.accuracy_score(yt, yp)
        return acc

    def predict(self, Xt):
        return self.cls.predict(Xt)

    def predict_prob(self, Xt):
        return self.cls.predict_proba(Xt)


class Top_Class():
    def __init__(self):
        self.tarfname = "./app/data/sentiment.tar.gz"
        self.tokenizer = TfidfVectorizer()
        self.class_label = ['NEGATIVE', 'POSITIVE']
        self.sentiment_data = read_files(self.tarfname, token=self.tokenizer)

        self.sentiment_cls = Sentiment_Classfier()
        self.explain = None

        # a = self.sentiment_data.dev_data
        # for item in a:
        #     print(item)

    def train(self):
        self.sentiment_cls.train_classifier(self.sentiment_data.trainX, self.sentiment_data.trainy)
        self.explain = shap.LinearExplainer(self.sentiment_cls.cls,
                                            self.sentiment_data.trainX,
                                            feature_dependence="independent")

    def test_on_dev(self):
        return self.sentiment_cls.evaluate(self.sentiment_data.devX, self.sentiment_data.devy)

    def test_query(self, query):
        query_X = self.tokenizer.transform([query])
        pred_index = self.sentiment_cls.predict(query_X)
        pred_prob = self.sentiment_cls.predict_prob(query_X)
        return self.class_label[pred_index[0]], pred_prob

    def getWeight(self):
        feature_names = self.tokenizer.get_feature_names()
        weight_dict = defaultdict(float)
        weights = self.sentiment_cls.cls.coef_
        for index, name in enumerate(feature_names):
            weight_dict[name] = weights[0, index]
        return weight_dict

    def shap_force_plot(self, query):
        query_X = self.tokenizer.transform([query])
        feature_names = self.tokenizer.get_feature_names()
        shap_values = self.explain.shap_values(query_X)


        index = []
        feature = []
        value = []
        words = query.strip().split(" ")
        for word in words:
            if word in feature_names and word not in feature:
                id = feature_names.index(word)
                index.append(id)
                feature.append(word)
                value.append(shap_values[0, id])

        return value, feature






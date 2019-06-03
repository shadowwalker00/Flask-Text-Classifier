# -----------------------------
# Sentiment Classfier
# Author: Guanghao Chen
# Date: May 28 2019
# -----------------------------

import os
import imageio
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import numpy as np

class Word_Cloud:
    def __init__(self):
        self.bg_coloring = imageio.imread(os.path.join("./app/static/pic/","twitter.png"))

        self.wc = WordCloud(background_color="grey",
                       max_words=100,
                       mask=self.bg_coloring,
                       stopwords=STOPWORDS.add("said"),
                       max_font_size=40,
                       random_state=42)
        self.image_colors = ImageColorGenerator(self.bg_coloring)
    
    def convert_sigmoid(self, word_dict):
        for key, val in word_dict.items():
            word_dict[key] = 1. / (1. + np.exp(-val)) * 2
        return word_dict

    def generate_wc_image(self, word_dict):
        word_dict = self.convert_sigmoid(word_dict)
        return word_dict
        # self.wc.generate_from_frequencies(word_dict)

# if __name__=="__main__":
#     senti_cls = Top_Class()
#     senti_cls.train()
#     mydict = senti_cls.getWeight()    
#     wc = Word_Cloud()
#     wc.generate_wc_image(mydict)

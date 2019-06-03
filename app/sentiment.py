#!/bin/python
import csv


def read_files(tarfname, token):
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
                        
    class Data: pass
    sentiment = Data()
    
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)

    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)

    sentiment.trainX = token.fit_transform(sentiment.train_data)
    sentiment.devX = token.transform(sentiment.dev_data)
    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment



class Data2: pass

def read_files_task2():
    trainname = "app/data/train.tsv"
    devname = "app/data/dev.tsv"




    sentiment = Data2()
    print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv2(trainname)
    print(len(sentiment.train_data))

    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv2(devname)
    print(len(sentiment.dev_data))
    print("-- transforming data and labels")
    from sklearn.feature_extraction.text import CountVectorizer
    # sentiment.count_vect = CountVectorizer()
    from sklearn.feature_extraction.text import TfidfVectorizer
    sentiment.count_vect = TfidfVectorizer(ngram_range=(1, 2))
    # sentiment.count_vect = TfidfVectorizer()
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    # tar.close()
    return sentiment



def read_unlabeled(tarfname, sentiment):
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    
    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name

    tf = tar.extractfile(unlabeledname)
    for line in tf:
        line = line.decode("utf-8")
        text = line.strip()
        unlabeled.data.append(text)
        
            
    unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
    print(unlabeled.X.shape)
    tar.close()
    return unlabeled

def read_tsv(tar, fname):
    member = tar.getmember(fname)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels

def read_tsv2(fname):
    # member = tar.getmember(fname)
    # print(member.name)
    tsvfile = open(fname)
    tf = csv.reader(tsvfile, delimiter='\t')
    data = []
    labels = []
    for line in tf:
        label = line[0]
        text = line[1]
        labels.append(label)
        data.append(text)
    tsvfile.close()
    return data, labels


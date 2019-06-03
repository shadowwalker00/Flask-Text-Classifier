# -*-coding:utf-8-*-
from flask import render_template,request, jsonify
from app import app
from app import cls_task2, sentiment_task2
import json
from app.classify1 import *
from app.classify2 import *
from app.wordC import *
from eli5.lime import TextExplainer
from sklearn.pipeline import  make_pipeline


senti_cls = Top_Class()
senti_cls.train()



@app.route('/', methods=['GET', 'POST'])
@app.route('/display', methods=['GET', 'POST'])
def display():
    mydict = senti_cls.getWeight()
    inv_word_dict = {}
    for key, val in mydict.items():
        inv_word_dict[key] = 0 - val

    # show word_cloud
    wc = Word_Cloud()
    wc_list = wc.generate_wc_image(mydict)


    return render_template('display.html',
                           title='Display Page')


@app.route('/display2', methods=['GET', 'POST'])
def display2():
    return render_template('display2.html',
                           title='Display Page')




@app.route('/query')
def query():
    return render_template('query.html',
                           title='Query Page',)



@app.route('/query2')
def query2():
    return render_template('query2.html',
                           title='Query Task2 Page',)

@app.route('/getdata', methods=['GET', 'POST'])
def get_data():
    weight_dict = senti_cls.getWeight()
    weight_top5 = sorted(weight_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    weight_last5 = sorted(weight_dict.items(), key=lambda x: x[1], reverse=True)[-5:]
    feature = [item[0] for item in weight_top5]
    value = [item[1] for item in weight_top5]
    feature.extend([item[0] for item in weight_last5])
    value.extend([item[1] for item in weight_last5])
    return json.dumps({'feature':feature,'value':value},ensure_ascii=False)


@app.route('/getDevAcc', methods=['GET', 'POST'])
def getDevAcc():
    mydict = senti_cls.getWeight()
    inv_word_dict = {}
    for key, val in mydict.items():
        inv_word_dict[key] = 0 - val

    wc = Word_Cloud()
    wc_list = wc.generate_wc_image(mydict)

    wc_inv = Word_Cloud()
    wc_list_inv = wc_inv.generate_wc_image(inv_word_dict)

    wc_list_list = []
    for key,value in wc_list.items():
        wc_list_list.append([key,value])
    ori = sorted(wc_list_list,key=lambda x:x[1],reverse=True)
    wc_list_list = ori[:200]

    wc_list_inv_show = []
    for key,value in wc_list_inv.items():
        wc_list_inv_show.append([key,value])
    ori2 = sorted(wc_list_inv_show,key=lambda x:x[1],reverse=True)
    wc_list_inv_show = ori2[:200]

    acc = round(senti_cls.test_on_dev()*100, 3)
    return json.dumps({'accuracy':acc, 'wc_list':wc_list_list,'wc_inv':wc_list_inv_show},ensure_ascii=False)


@app.route('/getTask2Acc', methods=['GET', 'POST'])
def getTask2Acc():

    train_acc = evaluate(sentiment_task2.trainX, sentiment_task2.trainy, cls_task2, 'train') *100
    train_acc = np.round(train_acc, 2)

    dev_acc = evaluate(sentiment_task2.devX, sentiment_task2.devy, cls_task2, 'dev') * 100
    dev_acc = np.round(dev_acc, 2)

    return json.dumps({'trainAcc':train_acc,
                       'devAcc':dev_acc})

@app.route('/getTask2Data', methods=['GET', 'POST'])
def getTask2Data():
    coefficients = cls_task2.coef_[0]

    top_k = np.argsort(coefficients)[-5:]
    last_k = np.argsort(coefficients)[:5]

    top_k_words = []
    corr_weights = []
    for i in top_k:
        top_k_words.append(sentiment_task2.count_vect.get_feature_names()[i])
        corr_weights.append(np.round(coefficients[i], 2))


    for i in last_k:
        top_k_words.append(sentiment_task2.count_vect.get_feature_names()[i])
        corr_weights.append(np.round(coefficients[i], 2))

    for item in zip(top_k_words, corr_weights):
        print(item)

    return json.dumps({'feature':top_k_words,
                       'value':corr_weights},ensure_ascii=False)



@app.route('/predict', methods=['POST'])
def predict():
    query_text = request.form["text"]
    res, prob = senti_cls.test_query(query_text)


    # --- show shap display ---
    shapValue, featurename = senti_cls.shap_force_plot(query_text)
    shapValue = list(shapValue)
    featurename = list(featurename)
    badValue = []
    goodValue = []
    for item in shapValue:
        if item>=0:
            badValue.append(0)
            goodValue.append(round(item,2))
        else:
            goodValue.append(0)
            badValue.append(round(item,2))

    # ---lime---
    te = TextExplainer(random_state=42)

    pipe = make_pipeline(senti_cls.tokenizer, senti_cls.sentiment_cls.cls)
    prob = pipe.predict_proba([query_text])

    te.fit(query_text, pipe.predict_proba)

    html = te.show_prediction(targets=['0'], target_names=['0', '1'])

    height = len(featurename)
    print("height:", height)
    return jsonify({"result": res,
                    "negprob":str(round(prob[0,0]*100,2))+"%",
                    "posprob":str(round(prob[0,1]*100,2))+"%",
                    "goodvalue":goodValue,
                    "badvalue": badValue,
                    "featurename":featurename,
                    "height":height,
                    "htmlData": html})


@app.route('/predict2', methods=['POST'])
def predict2():
    query_text = request.form["text"]

    # --- shap ---
    query_X = sentiment_task2.count_vect.transform([query_text])
    featurenames = sentiment_task2.count_vect.get_feature_names()

    shapValue, shapname = shap_force_plot(cls_task2,
                                          query_text,
                                          query_X,
                                          sentiment_task2.trainX,
                                          featurenames)
    shapValue = list(shapValue)
    shapname = list(shapname)
    badValue = []
    goodValue = []
    for item in shapValue:
        if item >= 0:
            badValue.append(0)
            goodValue.append(round(item, 2))
        else:
            goodValue.append(0)
            badValue.append(round(item, 2))
    height = len(shapname)


    # --- lime ---
    te = TextExplainer(random_state=42)
    pipe = make_pipeline(sentiment_task2.count_vect, cls_task2)
    prob = pipe.predict_proba([query_text])
    negprob = prob[0,0] * 100
    posprob = prob[0,1] * 100
    if prob[0,0] > prob[0,1]:
        res = "NEGATIVE"
    else:
        res = "POSITIVE"
    te.fit(query_text, pipe.predict_proba)
    html = te.show_prediction(targets=['0'], target_names=['0', '1'])

    return jsonify({"result":res,
                    "negprob": str(round(negprob, 2)) + "%",
                    "posprob": str(round(posprob, 2)) + "%",
                    "htmlData": html,
                    "goodvalue": goodValue,
                    "badvalue": badValue,
                    "featurename": shapname,
                    "height": height
                    })


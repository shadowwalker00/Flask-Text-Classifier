#!/bin/python

import pandas as pd
import shap

def train_classifier(X, y):
	"""Train a classifier using the given training data.

	Trains logistic regression on the input data with default parameters.
	"""
	from sklearn.linear_model import LogisticRegression
	from sklearn.linear_model import LogisticRegressionCV
	cls = LogisticRegression(penalty='l2', random_state=0, solver='lbfgs', max_iter=10000, C=21.5443)
	# cls = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10000)

	cls.fit(X, y)
	return cls

def evaluate(X, yt, cls, name='data'):
	"""Evaluated a classifier on the given labeled data using accuracy."""
	from sklearn import metrics
	yp = cls.predict(X)
	acc = metrics.accuracy_score(yt, yp)
	return acc


def shap_force_plot(cls, queryText, query_X, trainX, feature_names):
	explain = shap.LinearExplainer(cls, trainX, feature_dependence="independent")
	shap_values = explain.shap_values(query_X)

	index = []
	feature = []
	value = []
	words = queryText.strip().split(" ")
	for word in words:
		if word in feature_names and word not in feature:
			id = feature_names.index(word)
			index.append(id)
			feature.append(word)
			value.append(shap_values[0, id])

	return value, feature



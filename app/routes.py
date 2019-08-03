# -*-coding:utf-8-*-
from flask import render_template, request, jsonify
from app import app
import os


@app.route('/', methods=['GET', 'POST'])
def display():
    dir_path = 'reports'
    name_list = os.listdir(dir_path)
    image_list = [os.path.join(dir_path, image) for image in name_list]
    index_list = list(range(len(image_list)))
    image_list = list(zip(index_list,name_list,image_list))
    return render_template('base.html', title='Report Page', image_list=image_list)

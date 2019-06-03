# -*-coding:utf-8-*-
from flask_wtf import FlaskForm, Form
from wtforms import StringField, PasswordField, BooleanField, SubmitField,RadioField,DateField,IntegerField
from wtforms.validators import ValidationError, DataRequired, Email, EqualTo,NumberRange
from flask_wtf.file import FileField, FileRequired, FileAllowed


class QueryForm(Form):
    query_text = StringField('Query', validators=[DataRequired()])
    submit = SubmitField('Analyse')






from flask import Flask, render_template,request,url_for
import joblib
import numpy as np
import pandas as pd

model = joblib.load('/Users/nirmal/opt/Machine learning Models/ML_WEB/Wine_Quality.pkl')
cols = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur content',
        'total sulfur dioxide','density','pH','sulphates','alcohol']
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods = ['POST'])
def predict():
    num_features = [x for x in request.form.values()]
    print(num_features)
    final = np.array(num_features).reshape(1,11)
    ans = model.predict(final)
    return render_template('predict.html',pred = f'Expected Wine Quality: {ans}')

if __name__ == '__main__':
    app.run()
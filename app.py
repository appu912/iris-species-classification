from flask import Flask , render_template , request
import joblib
import numpy as np

model = joblib.load(open('iris.joblib' , 'rb'))

app = Flask(__name__)

@app.route('https://iris-species-prd.herokuapp.com/' , methods = ["GET"])
def Home():
    return render_template('index.html' , data = "" , c = "")

@app.route('https://iris-species-prd.herokuapp.com/predict' , methods = ['POST'])
def predict():
    sepalLength = request.form['sepalLength']
    sepalWidth = request.form['sepalWidth']
    petalLength = request.form['petalLength']
    petalWidth = request.form['petalWidth']

    datapoint = np.array([[sepalLength , sepalWidth , petalLength , petalWidth]])
    predicted = model.predict(datapoint)
    print(predicted)
    if predicted[0] == 0:
        ans = "Iris Setosa"
    elif predicted[0] == 1:
        ans = "Iris Versicolor"
    else:
        ans = "Iris Virginica"
    return render_template('index.html' , data = ans , c = "show-model")

if __name__ == "__main__":
    app.run(debug = True)
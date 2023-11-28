from flask import Flask
import pickle

app=Flask(__name__)
@app.route('/')
def index():
    return "Default API"

@app.route("/predict")
def iris_pred():
    with open ('model.pkl', 'rb') as model:
        ml_model=pickle.load(model)

    SepalLengthCm = 6.4
    SepalWidthCm = 4.8
    PetalLengthCm = 5.2
    PetalWidthCm =3.75

    result = ml_model.predict([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])
    if result ==2:
        iris_flower = "Iris-Verginica"
    if result ==0:
        iris_flower = "Iris-Setosa"
    if result ==1:
        iris_flower = "Iris-Versicolor"
    return iris_flower

app.run()
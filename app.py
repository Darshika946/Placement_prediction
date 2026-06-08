from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("placement_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    degree_p = float(request.form["degree_p"])
    mba_p = float(request.form["mba_p"])

    prediction = model.predict(np.array([[degree_p, mba_p]]))[0]

    probability = model.predict_proba(
        np.array([[degree_p, mba_p]])
    )[0][1] * 100

    if prediction == 1:
        result = "Placed 🎉"
    else:
        result = "Not Placed"

    return render_template(
        "index.html",
        prediction=result,
        probability=round(probability, 2)
    )

    

if __name__ == "__main__":
    app.run(debug=True)
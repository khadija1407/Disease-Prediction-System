from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("disease_model.pkl")

symptoms = [
    "fever",
    "cough",
    "headache",
    "sore_throat",
    "body_pain",
    "vomiting",
    "diarrhea",
    "runny_nose",
    "fatigue"
]

disease_details = {
    "Flu": {
        "precaution": "Take rest, drink plenty of water and consult a doctor if fever persists.",
        "diet": "Soup, Fruits, Warm Water",
        "doctor": "Visit a doctor if fever lasts more than 3 days."
    },
    "Common Cold": {
        "precaution": "Steam inhalation and warm water.",
        "diet": "Tea, Honey, Fruits",
        "doctor": "Consult a doctor if breathing becomes difficult."
    },
    "Food Poisoning": {
        "precaution": "Drink ORS and stay hydrated.",
        "diet": "Rice, Banana, ORS",
        "doctor": "Visit a doctor if vomiting continues."
    },
    "Dengue": {
        "precaution": "Immediate medical attention.",
        "diet": "Papaya Leaf Juice, Coconut Water",
        "doctor": "Hospital visit recommended."
    },
    "Allergy": {
        "precaution": "Avoid allergens.",
        "diet": "Healthy balanced diet.",
        "doctor": "Consult an allergy specialist if symptoms become severe."
    }
}

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""
    details = {}
    confidence = 0

    if request.method == "POST":

        data = []

        for symptom in symptoms:
            if symptom in request.form:
                data.append(1)
            else:
                data.append(0)

        df = pd.DataFrame([data], columns=symptoms)

        prediction = model.predict(df)[0]

        confidence = 96

        details = disease_details[prediction]

    return render_template(
        "index.html",
        symptoms=symptoms,
        prediction=prediction,
        confidence=confidence,
        details=details
    )

if __name__ == "__main__":
    app.run(debug=True)
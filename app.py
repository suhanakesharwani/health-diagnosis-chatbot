from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model and symptoms list
model = joblib.load("model/trained_model.pkl")
symptoms = joblib.load("model/symptom_list.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Get user input (symptoms as a comma-separated string)
        user_symptoms = request.form.get("symptoms")
        user_symptoms = [s.strip().lower() for s in user_symptoms.split(",")]
        
        # Create input data array (1 for present symptom, 0 for absent)
        input_data = [1 if s in user_symptoms else 0 for s in symptoms]
        
        # Predict disease using the trained model
        disease = model.predict([input_data])[0]
        prediction = f"🤖 Based on your symptoms, you may have: **{disease}**"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

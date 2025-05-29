from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load model and label encoders
model = joblib.load('car_eval_model.pkl')
label_encoders = joblib.load('car_eval_label_encoders.pkl')

# Define options for each feature (update if your categories are different)
feature_options = {
    'buying': ["vhigh", "high", "med", "low"],
    'maint': ["vhigh", "high", "med", "low"],
    'lug_boot': ["small", "med", "big"],
    'safety': ["low", "med", "high"]
}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        inputs = []
        for feature in feature_options:
            val = request.form[feature]
            le = label_encoders[feature]
            val_enc = le.transform([val])[0]
            inputs.append(val_enc)
        input_array = np.array([inputs])
        pred_enc = model.predict(input_array)[0]
        class_label = label_encoders['class'].inverse_transform([pred_enc])[0]
        prediction = class_label
    return render_template("index.html", feature_options=feature_options, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
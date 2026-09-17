from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load ML model
try:
    with open("placement_model.pkl", "rb") as file:
        model = pickle.load(open("placement_model.pkl", "rb"))
except FileNotFoundError:
    model = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():

    # If someone directly opens /predict
    if request.method == "GET":
        return redirect(url_for("home"))

    try:
        # Get form values
        cgpa = float(request.form.get("cgpa", 0))
        tenth = float(request.form.get("tenth", 0))
        twelfth = float(request.form.get("twelfth", 0))
        backlogs = int(request.form.get("backlogs", 0))
        internship = int(request.form.get("internship", 0))
        projects = int(request.form.get("projects", 0))
        communication = int(request.form.get("communication", 0))
        technical = int(request.form.get("technical", 0))

        # Basic validation
        if not (0 <= cgpa <= 10):
            return "Invalid CGPA. Enter a value between 0 and 10."

        if not (0 <= tenth <= 100):
            return "Invalid 10th percentage."

        if not (0 <= twelfth <= 100):
            return "Invalid 12th percentage."

        if backlogs < 0:
            return "Backlogs cannot be negative."

        if projects < 0:
            return "Projects cannot be negative."

        if not (1 <= communication <= 10):
            return "Communication skill must be between 1 and 10."

        if not (1 <= technical <= 10):
            return "Technical skill must be between 1 and 10."

        # Check model
        if model is None:
            return "model.pkl not found. Run train_model.py first."

        # Prepare input
        data = np.array([[
            cgpa,
            tenth,
            twelfth,
            backlogs,
            internship,
            projects,
            communication,
            technical
        ]])

        # Prediction
        prediction = model.predict(data)[0]

        probability = model.predict_proba(data)[0][1] * 100

        # Result
        if prediction == 1:
            result = "Likely to be Placed"
        else:
            result = "Placement Needs Improvement"

        return render_template(
            "result.html",
            result=result,
            probability=round(probability, 2)
        )

    except ValueError:
        return "Please enter valid numbers in all fields."

    except Exception as e:
        return f"Prediction Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
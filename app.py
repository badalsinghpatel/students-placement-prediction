from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

try:
    with open("placement_model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    model = None


# -------------------------------------------------
# ANALYSIS FUNCTIONS
# -------------------------------------------------

def analyze_student(data):
    strengths = []
    weaknesses = []
    focus = []
    recommendations = []

    # CGPA
    if data["cgpa"] >= 8:
        strengths.append("Strong academic performance")
    elif data["cgpa"] < 6.5:
        weaknesses.append("CGPA is relatively low")
        focus.append("Improve CGPA")
        recommendations.append("Focus on your semester subjects and maintain a consistent study routine.")

    # 10th
    if data["tenth"] >= 80:
        strengths.append("Good 10th academic record")
    elif data["tenth"] < 60:
        weaknesses.append("10th percentage is comparatively low")

    # 12th
    if data["twelfth"] >= 80:
        strengths.append("Good 12th academic record")
    elif data["twelfth"] < 60:
        weaknesses.append("12th percentage needs improvement")

    # Backlogs
    if data["backlogs"] == 0:
        strengths.append("No academic backlogs")
    else:
        weaknesses.append(f"{data['backlogs']} active/past backlog(s)")
        focus.append("Clear backlogs")
        recommendations.append("Prioritize clearing backlogs because some companies have academic eligibility criteria.")

    # Attendance
    if data["attendance"] >= 85:
        strengths.append("Good attendance")
    elif data["attendance"] < 75:
        weaknesses.append("Attendance is below the recommended level")
        focus.append("Improve attendance")

    # Coding
    if data["coding"] >= 8:
        strengths.append("Strong coding skills")
    elif data["coding"] < 6:
        weaknesses.append("Coding skills need improvement")
        focus.append("Coding")
        recommendations.append("Practice programming regularly and solve coding problems.")

    # Communication
    if data["communication"] >= 8:
        strengths.append("Strong communication skills")
    elif data["communication"] < 6:
        weaknesses.append("Communication skills need improvement")
        focus.append("Communication")
        recommendations.append("Practice English speaking, presentations and mock HR interviews.")

    # Aptitude
    if data["aptitude"] >= 75:
        strengths.append("Good aptitude performance")
    elif data["aptitude"] < 60:
        weaknesses.append("Aptitude score needs improvement")
        focus.append("Aptitude")
        recommendations.append("Practice quantitative aptitude, logical reasoning and verbal ability daily.")

    # Technical
    if data["technical"] >= 8:
        strengths.append("Strong technical skills")
    elif data["technical"] < 6:
        weaknesses.append("Technical skills need improvement")
        focus.append("Technical skills")
        recommendations.append("Strengthen DBMS, OOP, Computer Networks, Operating Systems and your core subjects.")

    # Projects
    if data["projects"] >= 3:
        strengths.append("Good project experience")
    elif data["projects"] < 2:
        weaknesses.append("Limited project experience")
        focus.append("Projects")
        recommendations.append("Build 2–3 practical projects and upload them to GitHub.")

    # Internship
    if data["internship"] == 1:
        strengths.append("Internship experience")
    else:
        weaknesses.append("No internship experience")
        focus.append("Internship")
        recommendations.append("Look for internships, especially roles related to your technical interests.")

    # Certifications
    if data["certifications"] >= 3:
        strengths.append("Good certification profile")
    elif data["certifications"] == 0:
        focus.append("Certifications")
        recommendations.append("Complete relevant certifications instead of collecting unrelated certificates.")

    # DSA
    if data["dsa"] == 3:
        strengths.append("Advanced DSA level")
    elif data["dsa"] == 1:
        weaknesses.append("DSA is at beginner level")
        focus.append("DSA")
        recommendations.append("Start with arrays, strings, linked lists, stacks, queues, trees and sorting algorithms.")

    # Resume
    if data["resume"] >= 8:
        strengths.append("Strong resume")
    elif data["resume"] < 6:
        weaknesses.append("Resume needs improvement")
        focus.append("Resume")
        recommendations.append("Create a one-page resume highlighting projects, skills, achievements and experience.")

    # Mock interview
    if data["mock"] >= 75:
        strengths.append("Good mock interview performance")
    elif data["mock"] < 60:
        weaknesses.append("Interview performance needs improvement")
        focus.append("Interview preparation")
        recommendations.append("Practice technical and HR mock interviews regularly.")

    # Leadership
    if data["leadership"] >= 8:
        strengths.append("Good leadership/extracurricular profile")
    elif data["leadership"] < 5:
        focus.append("Leadership & activities")
        recommendations.append("Participate in hackathons, clubs, events or team-based projects.")

    # Remove duplicate focus areas
    focus = list(dict.fromkeys(focus))
    recommendations = list(dict.fromkeys(recommendations))

    return strengths, weaknesses, focus, recommendations


# -------------------------------------------------
# HOME ROUTE
# -------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        # ---------------------------------------------
        # GET USER INPUTS
        # ---------------------------------------------

        cgpa = float(request.form["cgpa"])
        tenth = float(request.form["tenth"])
        twelfth = float(request.form["twelfth"])
        backlogs = int(request.form["backlogs"])
        attendance = float(request.form["attendance"])

        coding = int(request.form["coding"])
        communication = int(request.form["communication"])
        aptitude = float(request.form["aptitude"])
        technical = int(request.form["technical"])

        projects = int(request.form["projects"])
        internship = int(request.form["internship"])
        certifications = int(request.form["certifications"])

        dsa = int(request.form["dsa"])
        resume = int(request.form["resume"])
        mock = float(request.form["mock"])
        leadership = int(request.form["leadership"])


        # ---------------------------------------------
        # MODEL INPUT
        # ---------------------------------------------

        features = np.array([[
            cgpa,
            tenth,
            twelfth,
            backlogs,
            attendance,
            coding,
            communication,
            aptitude,
            technical,
            projects,
            internship,
            certifications,
            dsa,
            resume,
            mock,
            leadership
        ]])


        # ---------------------------------------------
        # PREDICTION
        # ---------------------------------------------

        if model is None:
            return "Error: placement_model.pkl not found. Run train_model.py first."

        prediction = model.predict(features)[0]

        probability = model.predict_proba(features)[0][1]

        placement_probability = round(probability * 100, 2)


        # ---------------------------------------------
        # READINESS SCORE
        # ---------------------------------------------

        readiness_score = (
            cgpa * 5
            + tenth * 0.08
            + twelfth * 0.08
            + attendance * 0.05
            + coding * 2
            + communication * 1.5
            + aptitude * 0.08
            + technical * 2
            + projects * 2
            + internship * 5
            + certifications * 0.5
            + dsa * 2
            + resume * 1
            + mock * 0.05
            + leadership * 0.5
            - backlogs * 3
        )

        # Convert roughly to 0–100
        readiness_score = max(0, min(100, readiness_score))


        # ---------------------------------------------
        # RISK LEVEL
        # ---------------------------------------------

        if placement_probability >= 75:
            risk = "Low"
            risk_symbol = "🟢"
        elif placement_probability >= 50:
            risk = "Medium"
            risk_symbol = "🟡"
        else:
            risk = "High"
            risk_symbol = "🔴"


        # ---------------------------------------------
        # STUDENT ANALYSIS
        # ---------------------------------------------

        student_data = {
            "cgpa": cgpa,
            "tenth": tenth,
            "twelfth": twelfth,
            "backlogs": backlogs,
            "attendance": attendance,
            "coding": coding,
            "communication": communication,
            "aptitude": aptitude,
            "technical": technical,
            "projects": projects,
            "internship": internship,
            "certifications": certifications,
            "dsa": dsa,
            "resume": resume,
            "mock": mock,
            "leadership": leadership
        }

        strengths, weaknesses, focus, recommendations = analyze_student(student_data)


        # ---------------------------------------------
        # GENERAL RESULT
        # ---------------------------------------------

        if prediction == 1:
            result = "Good Placement Potential"
            result_icon = "🟢"
        else:
            result = "Needs Improvement"
            result_icon = "🟠"


        # ---------------------------------------------
        # MAIN REASON
        # ---------------------------------------------

        if strengths:
            main_reason = strengths[0]
        else:
            main_reason = "Your profile needs improvement in several placement-related areas."


        return render_template(
            "result.html",
            prediction=prediction,
            probability=placement_probability,
            readiness=round(readiness_score, 1),
            risk=risk,
            risk_symbol=risk_symbol,
            result=result,
            result_icon=result_icon,
            main_reason=main_reason,
            strengths=strengths,
            weaknesses=weaknesses,
            focus=focus,
            recommendations=recommendations
        )

    return render_template("index.html")


# -------------------------------------------------
# RUN APP
# -------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)

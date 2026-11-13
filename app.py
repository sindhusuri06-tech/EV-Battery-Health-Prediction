from pathlib import Path
import io
import joblib
import pandas as pd
from flask import Flask, render_template, request, send_file, flash, redirect, url_for

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "models" / "best_model_bundle.joblib"
app = Flask(__name__)
app.secret_key = "change-this-local-demo-secret"

def load_bundle():
    if not BUNDLE_PATH.exists():
        return None
    return joblib.load(BUNDLE_PATH)

def add_engineered_features(data):
    data = data.copy()
    data["Power_W"] = data["Voltage_V"] * data["Current_A"]
    data["SOH_per_Cycle"] = data["SOH_Percentage"] / (data["Cycle_Count"] + 1)
    data["Resistance_SOH_Ratio"] = data["Internal_Resistance_mOhm"] / (data["SOH_Percentage"] + 1)
    data["Thermal_Stress"] = data["Temperature_C"] * data["Current_A"] / 100
    data["Aging_Impact"] = data["Impedance_Growth"] * data["Aging_Rate"] * 100
    return data

def recommendation(label):
    return {
        "Healthy": "Continue routine monitoring and follow the vehicle manufacturer's charging guidance.",
        "Moderate": "Monitor battery performance regularly and arrange a routine battery inspection.",
        "Degraded": "Arrange a professional battery diagnostic and review charging and thermal-management practices.",
        "Critical": "Avoid relying on this prediction alone. Arrange an urgent inspection by a qualified EV technician."
    }.get(label, "Consult a qualified EV technician.")

@app.route("/", methods=["GET", "POST"])
def index():
    bundle = load_bundle()
    if request.method == "POST":
        if bundle is None:
            flash("Model not found. Run Source_Code/train_model.py with the official dataset first.")
            return redirect(url_for("index"))
        fields = ["Voltage_V","Current_A","Temperature_C","SOH_Percentage",
                  "Internal_Resistance_mOhm","Cycle_Count","Impedance_Growth","Aging_Rate",
                  "Vehicle_Type","Battery_Chemistry","Manufacturer","Driving_Load","Weather_Condition"]
        row = {}
        for f in fields:
            value = request.form.get(f, "").strip()
            row[f] = float(value) if f in fields[:8] else value
        df = add_engineered_features(pd.DataFrame([row]))
        model = bundle["model"]
        pred = model.predict(df)[0]
        probs = model.predict_proba(df)[0]
        classes = model.named_steps["classifier"].classes_
        probability = {str(c): round(float(p)*100, 2) for c,p in zip(classes, probs)}
        return render_template("index.html", result=str(pred),
                               probability=probability, advice=recommendation(str(pred)))
    return render_template("index.html", result=None)

@app.route("/batch", methods=["POST"])
def batch():
    bundle = load_bundle()
    upload = request.files.get("file")
    if bundle is None or upload is None or upload.filename == "":
        flash("Train the model and select a CSV file.")
        return redirect(url_for("index"))
    data = pd.read_csv(upload)
    model = bundle["model"]
    prepared = add_engineered_features(data)
    data["Predicted_Battery_Health"] = model.predict(prepared)
    output = io.BytesIO()
    data.to_csv(output, index=False); output.seek(0)
    return send_file(output, mimetype="text/csv", as_attachment=True,
                     download_name="battery_health_predictions.csv")

if __name__ == "__main__":
    app.run(debug=True)

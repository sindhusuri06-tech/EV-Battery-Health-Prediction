"""
Creates a structured DOCX report draft after model training.
Run: python Source_Code/report_builder.py
Replace student details and add the actual app screenshots before submission.
"""
from pathlib import Path
import json
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = OUT / "figures"
doc = Document()
sec = doc.sections[0]
sec.top_margin = sec.bottom_margin = sec.left_margin = sec.right_margin = Inches(1)
style = doc.styles["Normal"]; style.font.name = "Times New Roman"; style.font.size = Pt(12)

def title(text):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text); r.bold = True; r.font.size = Pt(18)
def heading(text, level=1): doc.add_heading(text, level=level)
def para(text): doc.add_paragraph(text)
def figure(path, caption):
    if path.exists():
        doc.add_picture(str(path), width=Inches(5.8))
        p = doc.add_paragraph(caption); p.alignment = WD_ALIGN_PARAGRAPH.CENTER

title("ELECTRIC VEHICLE BATTERY HEALTH PREDICTION USING MACHINE LEARNING")
para("Internship Task ID: AI-ML-039")
para("Student Name: [ENTER YOUR NAME]")
para("Student ID: [ENTER YOUR STUDENT ID]")
para("College / Department: [ENTER DETAILS]")
para("Academic Year: 2026")
doc.add_page_break()

sections = [
("DECLARATION", "I declare that this project report describes the work completed for the EV Battery Health Prediction task. Replace this statement with your institution's required declaration and ensure it accurately reflects your own work."),
("ACKNOWLEDGEMENT", "I thank my project mentor, faculty members, and the internship program for the guidance and learning opportunity."),
("ABSTRACT", "This project develops a machine-learning workflow to classify electric-vehicle battery health into Healthy, Moderate, Degraded, and Critical categories. The workflow includes data cleaning, domain feature engineering, categorical encoding, model training, evaluation, and a Flask-based demonstration interface. Logistic Regression and Random Forest are compared using a stratified train-test split and cross-validation. Insert the actual dataset size and measured results after running the project."),
("TABLE OF CONTENTS", "Update the table of contents in Word after editing the report."),
("LIST OF ABBREVIATIONS, FIGURES AND TABLES", "EV – Electric Vehicle; SOH – State of Health; SMOTE – Synthetic Minority Over-sampling Technique; ROC – Receiver Operating Characteristic; AUC – Area Under the Curve. Add the final figure and table lists."),
("INTRODUCTION", "Electric vehicles depend on rechargeable battery packs. Battery performance changes with operating temperature, charge-discharge activity, internal resistance, and aging. A data-driven health classification system can help summarize measured battery conditions and support maintenance planning."),
("PROBLEM STATEMENT", "Battery condition is not always easy to interpret from multiple sensor measurements. The project investigates whether supervised machine-learning models can classify battery health using recorded electrical, thermal, usage, and chemistry-related information."),
("EXISTING SYSTEM", "Battery inspection may involve diagnostic tools, manual interpretation, and periodic service checks. These approaches can be time-consuming and may not provide a simple consolidated health category from multiple input variables."),
("PROPOSED SYSTEM", "The proposed workflow loads the dataset, validates and preprocesses the measurements, engineers battery-related features, trains two classifiers, evaluates their performance, and exposes the selected trained model through a Flask interface."),
("OBJECTIVES", "1. Prepare the battery dataset. 2. Engineer five domain-specific features. 3. Compare Logistic Regression and Random Forest. 4. Evaluate using accuracy, precision, recall, weighted F1, ROC-AUC, and confusion matrices. 5. Demonstrate single-record and batch prediction through Flask."),
("SCOPE OF WORK", "The scope is limited to classification using the provided dataset and a demonstration web application. Predictions depend on dataset quality and should not be treated as a certified battery safety diagnosis."),
("LITERATURE SURVEY", "Summarize at least three relevant, verified papers or technical references here. For each source, record authors, title, publication, year, method, and relevance. Do not invent citations."),
("METHODOLOGY", "The workflow uses an 80/20 stratified train-test split. Numeric missing values are imputed with the median; categorical missing values use the most frequent category. The five engineered features are Power_W, SOH_per_Cycle, Resistance_SOH_Ratio, Thermal_Stress, and Aging_Impact. Scaling, categorical encoding, and SMOTE are placed in an imbalanced-learn pipeline so resampling occurs only within training folds. Five-fold StratifiedKFold cross-validation uses weighted F1 scoring."),
("RESULTS AND DISCUSSION", "IMPORTANT: Insert actual metrics from outputs/metrics.json. Include the generated figures and screenshots of the running Flask application. Do not use example or advertised metrics as if they were measured on your run."),
("CONCLUSION", "Write the conclusion after evaluating the actual results. State which model performed better on your test set, whether the target metrics were achieved, and the limitations observed."),
("FUTURE SCOPE", "Possible extensions include collecting real battery telemetry, validating across battery manufacturers, adding time-series degradation forecasting, and integrating secure vehicle diagnostic data."),
("REFERENCES", "Add only sources you have personally checked. Include the official dataset citation and relevant machine-learning references."),
("DATASET LINK", "Paste the exact URL from which the official Battery_RUL_Dataset.csv was obtained. This must remain the final section as required by the task.")
]
for h, body in sections:
    heading(h, 1); para(body)

metrics_path = OUT / "metrics.json"
if metrics_path.exists():
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    heading("Measured model results (replace/update after final run)", 2)
    para(json.dumps({k:v for k,v in data.items() if k != "models"}, indent=2))
    for name, values in data["models"].items():
        heading(name, 2)
        para(f"Accuracy: {values['accuracy']:.4f}; Weighted precision: {values['precision_weighted']:.4f}; "
             f"Weighted recall: {values['recall_weighted']:.4f}; Weighted F1: {values['f1_weighted']:.4f}; "
             f"5-fold CV weighted F1: {values['cv_f1_weighted_mean']:.4f} ± {values['cv_f1_weighted_std']:.4f}; "
             f"Weighted OVR ROC-AUC: {values['roc_auc_ovr_weighted']}")
for img in sorted(FIG.glob("*.png")):
    figure(img, img.stem.replace("_"," ").title())
doc.save(OUT / "AI-ML-039_Project_Report_DRAFT.docx")
print("Created", OUT / "AI-ML-039_Project_Report_DRAFT.docx")

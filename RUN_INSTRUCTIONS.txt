AI-ML-039 | EV BATTERY HEALTH PREDICTION
======================================

IMPORTANT: This is a runnable project starter package, not a verified final submission.
The official task requires the original Battery_RUL_Dataset.csv (100,000 records).
That dataset was not included with the task text, so no genuine trained model, metrics,
charts, or final results have been fabricated.

NEXT STEPS
1. Put the official Battery_RUL_Dataset.csv in the dataset/ folder.
2. Open this folder in VS Code.
3. In the terminal run:
      python -m venv .venv
      .venv\Scripts\activate
      pip install -r Source_Code/requirements.txt
      python Source_Code/train_model.py --data dataset/Battery_RUL_Dataset.csv
4. Check outputs/metrics.json and outputs/figures/.
5. Start the web app:
      python Source_Code/app.py
   Open http://127.0.0.1:5000
6. Replace the clearly marked report placeholders with the actual results and screenshots.

DATASET SCHEMA EXPECTED
Voltage_V, Current_A, Temperature_C, SOH_Percentage,
Internal_Resistance_mOhm, Cycle_Count, Impedance_Growth, Aging_Rate,
Vehicle_Type, Battery_Chemistry, Manufacturer, Driving_Load,
Weather_Condition, Battery_Health_Class

The assignment description says 16 features, but the listed columns above are 14.
Check the actual CSV header and adjust the feature list if the file contains extra fields.

ACADEMIC INTEGRITY / RESULTS
Do not submit the report with blank placeholders. Do not claim 85% accuracy unless
the run on the official dataset actually achieves it. If results differ from the target,
report the real results and discuss possible reasons.

Report format is a draft framework. Add your own name, student ID, college, date,
real charts, app screenshots, and measured metrics before submission.

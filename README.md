# Drug_side_effect_predictor
A ML project that predicts potential side effects of pharmaceutical drugs based on structured information such as drug name, category, and related attributes. Built using Python, this project applies classification algorithms to assess drug safety and assist in identifying adverse effects — a valuable tool for healthcare data analysis.

# 💊 Drug Side Effect Predictor

A machine learning project that predicts the **side effect sentiment** of drugs based on drug metadata, usage condition, and textual side effect descriptions.

---

## 📂 Project Structure

DRUG_SIDE_EFFECT_PREDICTOR/
├── notebooks/ # Jupyter notebooks for EDA, modeling, etc.
│ ├── EDA.ipynb
│ ├── feature_engineering.ipynb
│ ├── model_building.ipynb
│ └── model_training_and_testing.ipynb
│
├── src/ # Source code (Python modules)
│ ├── init.py
│ ├── data_ingestion.py
│ ├── preprocessing.py
│ └── (model.py, predictor.py coming soon)
│
├── projects_files/ # Raw and cleaned data
│ └── drugs_side_effects_drugs_com.csv
│
├── app.py/ # Streamlit or Flask app (if applicable)
│ 
│
├── report_generation.py / # Evaluation reports and charts
│ 
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
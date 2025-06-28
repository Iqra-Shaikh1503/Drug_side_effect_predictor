def predict(drug_data_dict):
    import joblib
    import pandas as pd

    # Load models
    model_se = joblib.load("models/side_effect_model.pkl")
    model_mc = joblib.load("models/medical_condition_model.pkl")

    # Load encoders
    mlb = joblib.load("models/side_effect_encoder.pkl")
    le_medical = joblib.load("models/medical_condition_encoder.pkl")

    # Convert input into DataFrame
    input_df = pd.DataFrame([drug_data_dict])

    # Predict side effects (multi-label)
    side_effect_pred = model_se.predict(input_df)
    predicted_se = mlb.inverse_transform(side_effect_pred)[0]

    # Predict medical condition (single-label)
    medical_pred = model_mc.predict(input_df)[0]
    predicted_mc = le_medical.inverse_transform([medical_pred])[0]

    return predicted_se, predicted_mc

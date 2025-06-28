import streamlit as st
import pandas as pd
import joblib

# Loading models and encoders
model_se = joblib.load(r"notebooks/models/side_effect_model.pkl")
model_mc = joblib.load(r"notebooks/models/medical_condition_model.pkl")
mlb = joblib.load(r"notebooks/models/side_effect_encoder.pkl")
le_medical = joblib.load(r"notebooks/models/medical_condition_encoder.pkl")

# Load mapping dictionaries
drug_name_map = joblib.load(r"notebooks/models/drug_name_map.pkl")
drug_class_map = joblib.load(r"notebooks/models/drug_class_map.pkl")
preg_cat_map = joblib.load(r"notebooks/models/pregnancy_category_map.pkl")

# Reverse mappings for display
drug_class_map_rev = {v: k for k, v in drug_class_map.items()}
preg_cat_map_rev = {v: k for k, v in preg_cat_map.items()}

# Streamlit UI
st.set_page_config(page_title="Drug Effect Predictor", layout="centered")
st.title("💊 Drug Side Effect & Medical Condition Predictor")
st.markdown("Enter drug details to predict possible **side effects** and **medical condition**.")

# Input: Drug Name
selected_drug = st.selectbox("🔹 Drug Name", list(drug_name_map.keys()))
drug_encoded = drug_name_map[selected_drug]

# Input: Drug Class
selected_class = st.selectbox("🔹 Drug Class", list(drug_class_map.keys()))
class_encoded = drug_class_map[selected_class]

# Input: Activity
activity = st.slider("🔹 Drug Activity (%)", 0.0, 100.0, step=1.0)

# Input: Pregnancy Category
selected_preg = st.selectbox("🔹 Pregnancy Category", list(preg_cat_map.keys()))
preg_encoded = preg_cat_map[selected_preg]

# Input: Rating
rating = st.slider("🔹 Rating (out of 10)", 0.0, 10.0, step=0.1)

# Input: Number of Reviews
reviews = st.number_input("🔹 Number of Reviews", min_value=0)

# Predicting button
if st.button("🔍 Predict Side Effects & Condition"):
    input_df = pd.DataFrame([{
        "drug_name": drug_encoded,
        "drug_classes": class_encoded,
        "activity": activity,
        "pregnancy_category": preg_encoded,
        "rating": rating,
        "no_of_reviews": reviews
    }])

    # Predicting side effects and medical condition
    se_pred = model_se.predict(input_df)
    predicted_side_effects = mlb.inverse_transform(se_pred)[0]

    med_pred = model_mc.predict(input_df)[0]
    predicted_condition = le_medical.inverse_transform([med_pred])[0]

    # Displaying results
    st.markdown("### 🧾 Results")
    st.success(f"**Predicted Medical Condition:** {predicted_condition}")
    st.info(f"**Possible Side Effects:** {', '.join(predicted_side_effects)}")
    st.caption("🔬 Note: This prediction is for demonstration purposes only.")
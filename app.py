import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder
st.set_page_config(page_title="MediScan AI", layout="wide")

# PREMIUM MEDICAL THEME

st.markdown("""
<style>

/* App background */
.stApp {
    background-color: #F4F9FF;
}

/* Sidebar slimmer */
section[data-testid="stSidebar"] {
    background-color: #E6F2FF;
    width: 280px !important;
}

section[data-testid="stSidebar"] * {
    color: #003366;
}

/* Buttons */
.stButton>button {
    background-color: #007BFF;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: #0056b3;
}

/* Metric cards */
[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #CCE0FF;
    padding: 12px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0, 76, 153, 0.15);
}

/* Input card */
.input-card {
    background-color: white;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0, 76, 153, 0.12);
}

/* Expander */
details {
    background-color: white;
    border: 1px solid #D6E9FF;
    border-radius: 10px;
    padding: 8px;
}

/* Footer */
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #004C99;
    color: white;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# HEADER SECTION


st.markdown("""
<div style='text-align:center; padding:20px;'>
    <h1 style='color:#003366;'> MediScan AI</h1>
    <p style='font-size:20px; color:#0056b3;'>
        Intelligent Clinical Decision Support System for CKD
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border: 1px solid #CCE0FF;'>", unsafe_allow_html=True)

# Clinical disclaimer
st.info("⚕ This AI system assists in early CKD risk detection. It is not a replacement for professional medical diagnosis.")


# LOAD DATA
@st.cache_data
def load_data():
    return pd.read_csv("mediscan_ckd_diagnostic.csv")

df = load_data()


# TRAIN MODEL
# Label Encoding
@st.cache_resource
def train_model(df):

    X = df.drop(["CKD_Status", "Patient_ID"], axis=1)
    y = df["CKD_Status"]

    X["Red_Blood_Cells"] = X["Red_Blood_Cells"].map({"Normal":0, "Abnormal":1})
    X["Hypertension"] = X["Hypertension"].map({"No":0, "Yes":1})
    X["Diabetes_Mellitus"] = X["Diabetes_Mellitus"].map({"No":0, "Yes":1})

    feature_columns = X.columns

    imputer = KNNImputer()
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    best_model= RandomForestClassifier(random_state=42)
    param_grid = {    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10]
}

    grid = GridSearchCV(
    best_model, param_grid=param_grid, 
    cv=3, n_jobs=-1
)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    return model, imputer, scaler, feature_columns, accuracy, recall, precision

model, imputer, scaler, feature_columns, accuracy, recall, precision = train_model(df)

# SIDEBAR

st.sidebar.header(" Model Performance")
st.sidebar.metric("Accuracy", f"{accuracy:.3f}")
st.sidebar.metric("Recall", f"{recall:.3f}")
st.sidebar.metric("Precision", f"{precision:.3f}")

st.sidebar.markdown("---")
st.sidebar.write("**Model:** Random Forest")
st.sidebar.write("**Imputation:** KNN")
st.sidebar.write("**Scaling:** StandardScaler")

st.sidebar.markdown("---")
with st.sidebar.expander(" Project Team Members"):
    st.markdown("""
    - **Devaj TN**
    - **Sahil Shahanas**
    - **Razik Rahman M S**
    - **Abhinav K**
    - **Shihan Shoukathali**
    """)

# INPUT SECTION



st.markdown("<h3 style='color:#003366;'> Patient Clinical Inputs</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

data = {}
data["Age"] = col1.number_input("Age", 1, 120, 45)
data["Blood_Pressure"] = col2.number_input("Blood Pressure", 50, 200, 80)
data["Specific_Gravity"] = col3.selectbox("Specific Gravity", [1.005,1.010,1.015,1.020,1.025])

data["Albumin"] = col1.slider("Albumin", 0, 5, 1)
data["Sugar"] = col2.slider("Sugar", 0, 5, 0)
data["Red_Blood_Cells"] = col3.selectbox("RBC", ["Normal","Abnormal"])

data["Blood_Glucose_Random"] = col1.number_input("Random Glucose", 50, 500, 120)
data["Blood_Urea"] = col2.number_input("Blood Urea", 1, 300, 40)
data["Serum_Creatinine"] = col3.number_input("Serum Creatinine", 0.1, 20.0, 1.2)

data["Sodium"] = col1.number_input("Sodium", 100, 200, 140)
data["Hemoglobin"] = col2.number_input("Hemoglobin", 3.0, 20.0, 13.5)
data["White_Blood_Cell_Count"] = col3.number_input("WBC Count", 1000, 20000, 8000)

data["Hypertension"] = col1.selectbox("Hypertension", ["No","Yes"])
data["Diabetes_Mellitus"] = col2.selectbox("Diabetes", ["No","Yes"])

st.markdown('</div>', unsafe_allow_html=True)

# Centered Button
col_btn1, col_btn2, cwol_btn3 = st.columns([1,2,1])
with col_btn2:
    analyze = st.button(" Analyze CKD Risk")

# PREDICTION

if analyze:

    input_df = pd.DataFrame([data])
    input_df["Red_Blood_Cells"] = input_df["Red_Blood_Cells"].map({"Normal":0, "Abnormal":1})
    input_df["Hypertension"] = input_df["Hypertension"].map({"No":0, "Yes":1})
    input_df["Diabetes_Mellitus"] = input_df["Diabetes_Mellitus"].map({"No":0, "Yes":1})

    input_df = input_df[feature_columns]
    input_scaled = scaler.transform(imputer.transform(input_df))

    with st.spinner("Analyzing patient data..."):
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("<hr style='border: 1px solid #CCE0FF;'>", unsafe_allow_html=True)

    st.subheader("Diagnosis Result")

    if prediction == 1:
        st.error(f" High Risk of CKD ({probability*100:.1f}%)")
    else:
        st.success(f" Low Risk ({probability*100:.1f}%)")

# FEATURE IMPORTANCE

with st.expander("View Model Feature Importance"):

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance")
    plt.tight_layout()
    st.pyplot(fig)


# FOOTER


st.markdown("""
<div class="footer">
© 2026 MediScan AI | Early Diagnosis of Chronic Kidney Disease  
Developed by Team 5 | AI & ML Project
</div>
""", unsafe_allow_html=True)

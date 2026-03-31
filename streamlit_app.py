import datetime
import pandas as pd
import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Pastikan file ini ada di direktori yang sama
DATA_CLEANED_PATH = 'employee_data_cleaned.csv'
MODEL_PATH = 'model_gb.joblib'

def data_preprocessing(data_input):
    # 1. Load data referensi untuk memastikan encoding konsisten
    try:
        df_ref = pd.read_csv(DATA_CLEANED_PATH)
    except FileNotFoundError:
        st.error(f"File {DATA_CLEANED_PATH} tidak ditemukan!")
        return None

    # Hapus kolom yang tidak diperlukan dari data referensi
    df_ref = df_ref.drop(columns=['EmployeeId', 'Attrition'], errors='ignore')
    
    # 2. Gabungkan data referensi dengan input user
    # Ini dilakukan agar LabelEncoder mengetahui semua kategori yang mungkin ada
    df_combined = pd.concat([df_ref, data_input], ignore_index=True)

    numerical = df_combined.select_dtypes(exclude='object').columns.tolist()
    categorical = df_combined.select_dtypes(include='object').columns.tolist()

    # 3. Proses Encoding (Perbaikan: Loop per kolom)
    for col in categorical:
        le = LabelEncoder()
        df_combined[col] = le.fit_transform(df_combined[col].astype(str))

    # 4. Proses Scaling
    scaler = MinMaxScaler()
    df_combined[numerical] = scaler.fit_transform(df_combined[numerical])

    # Ambil baris terakhir (data yang baru diinput user)
    return df_combined.tail(1).to_numpy()


def model_predict(data_processed):
    try:
        model = joblib.load(MODEL_PATH)
        return model.predict(data_processed)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None


def prediction(output):
    if output is None:
        return
    
    st.markdown("---")
    if output == 1:
        st.error("### 🚩 Status Attrition: YES (Karyawan Berpotensi Keluar)")
    else:
        st.success("### ✅ Status Attrition: NO (Karyawan Berpotensi Bertahan)")


def main():
    st.set_page_config(page_title="HR Attrition Predictor", layout="wide")
    st.title("📊 Permasalahan Human Resource Dashboard")
    st.write("Isi data karyawan di bawah ini untuk memprediksi potensi *attrition*.")

    # --- BAGIAN INPUT (TIDAK ADA YANG DIUBAH SESUAI REQUEST) ---
    with st.container():
        col_gender, col_age, col_marital = st.columns(3)
        with col_gender:
            gender = st.radio("Gender", ["Male", "Female"])
        with col_age:
            age = st.number_input("Age", min_value=18, max_value=60, value=30)
        with col_marital:
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    with st.container():
        col_education, col_edu_field = st.columns(2)
        with col_education:
            education = st.selectbox("Education", ["Below College", "College", "Bachelor", "Master", "Doctor"])
        with col_edu_field:
            education_field = st.selectbox("Education Field", ["Human Resources", "Life Sciences", "Marketing", "Medical", "Technical Degree", "Other"])

    with st.container():
        col_distance, col_business_travel = st.columns(2)
        with col_distance:
            distance_from_home = st.number_input("Distance From Home to Work (Km)", min_value=0, step=1)
        with col_business_travel:
            business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel Rarely", "Travel Frequently"])

    with st.container():
        col_dept, col_job_role, col_job_level = st.columns(3)
        with col_dept:
            department = st.selectbox("Department", ["Human Resources", "Research & Development", "Sales"])
        with col_job_role:
            job_role = st.selectbox("Job Role", ["Human Resources", "Sales Executive", "Sales Representative", "Healthcare Representative", "Research Scientist", "Laboratory Technician", "Manager", "Manufacturing Director", "Research Director"])
        with col_job_level:
            job_level = st.selectbox("Job Level", ["1", "2", "3", "4", "5"])

    with st.container():
        col_hourly_rate, col_daily_rate, col_monthly_rate = st.columns(3)
        with col_hourly_rate:
            hourly_rate = st.number_input("Hourly Rate", min_value=0, step=1)
        with col_daily_rate:
            daily_rate = st.number_input("Daily Rate", min_value=0, step=100)
        with col_monthly_rate:
            monthly_rate = st.number_input("Monthly Rate", min_value=0, step=1000)

    with st.container():
        col_monthly_income, col_percent_salary_hike = st.columns(2)
        with col_monthly_income:
            monthly_income = st.number_input("Monthly Income", min_value=0, step=100)
        with col_percent_salary_hike:
            percent_salary_hike = st.number_input("Percent Salary Hike (%)", min_value=0, step=1)

    with st.container():
        col_standard_hours, col_over_time = st.columns(2)
        with col_standard_hours:
            standard_hours = st.number_input("Standard Hours", value=80)
        with col_over_time:
            over_time = "Yes" if st.checkbox("Over Time") else "No"

    # Data Construction (Sesuai kolom yang diminta model)
    data = [[
        age, business_travel, daily_rate, department,
        distance_from_home, education, education_field,
        "Medium", gender, hourly_rate, "Medium",
        int(job_level), job_role, "Medium", marital_status,
        monthly_income, monthly_rate, 1, over_time,
        percent_salary_hike, "Good", "Medium",
        standard_hours, 1, 5, 1, "Good",
        2, 2, 1, 1
    ]]

    df_input = pd.DataFrame(data, columns=[
        'Age','BusinessTravel','DailyRate','Department','DistanceFromHome',
        'Education','EducationField','EnvironmentSatisfaction','Gender',
        'HourlyRate','JobInvolvement','JobLevel','JobRole','JobSatisfaction',
        'MaritalStatus','MonthlyIncome','MonthlyRate','NumCompaniesWorked',
        'OverTime','PercentSalaryHike','PerformanceRating',
        'RelationshipSatisfaction','StandardHours','StockOptionLevel',
        'TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance',
        'YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion',
        'YearsWithCurrManager'
    ])

    if st.button("✨ Predict"):
        with st.spinner('Calculating...'):
            data_ready = data_preprocessing(df_input)
            if data_ready is not None:
                output = model_predict(data_ready)
                prediction(output[0])

    # Footer
    st.markdown("---")
    year_now = datetime.date.today().year
    st.caption(f"Copyright © {year_now} Fahmi Fadillah")


if __name__ == "__main__":
    main()
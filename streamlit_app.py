import datetime
import pandas as pd
import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# --- FUNGSI PREPROCESSING ---
# Perbaikan: Menggunakan dekorator cache agar tidak membaca CSV berulang kali
@st.cache_data
def get_clean_df():
    # Pastikan file ini ada di folder yang sama dengan script
    df = pd.read_csv('employee_data_cleaned.csv')
    # Hapus kolom yang tidak digunakan di model
    if 'EmployeeId' in df.columns:
        df = df.drop(columns=['EmployeeId'])
    if 'Attrition' in df.columns:
        df = df.drop(columns=['Attrition'])
    return df

def data_preprocessing(data_input):
    df_base = get_clean_df()
    
    # Gabungkan data input dengan dataset asli untuk menjaga konsistensi kategori
    df_combined = pd.concat([df_base, data_input], ignore_index=True)

    numerical = df_combined.select_dtypes(exclude='object').columns.tolist()
    categorical = df_combined.select_dtypes(include='object').columns.tolist()

    # Perbaikan: Label Encoding harus dilakukan per kolom dengan benar
    le = LabelEncoder()
    for col in categorical:
        df_combined[col] = le.fit_transform(df_combined[col].astype(str))
    
    # Perbaikan: Scaling
    scaler = MinMaxScaler()
    df_combined[numerical] = scaler.fit_transform(df_combined[numerical])

    # Ambil baris terakhir (data yang diinput user)
    return df_combined.tail(1).to_numpy()

# --- FUNGSI PREDIKSI ---
def model_predict(data_processed):
    try:
        model = joblib.load('model_gb.joblib')
        return model.predict(data_processed)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

def main():
    # Setup Halaman
    st.set_page_config(page_title="HR Attrition Predictor", layout="wide")
    st.title("📊 Permasalahan Human Resource Dashboard")
    st.markdown("---")

    # --- INPUT USER ---
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

    # --- MAPPING DATA ---
    # Pastikan jumlah kolom dan urutannya sama persis dengan saat training
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

    columns_name = [
        'Age','BusinessTravel','DailyRate','Department','DistanceFromHome',
        'Education','EducationField','EnvironmentSatisfaction','Gender',
        'HourlyRate','JobInvolvement','JobLevel','JobRole','JobSatisfaction',
        'MaritalStatus','MonthlyIncome','MonthlyRate','NumCompaniesWorked',
        'OverTime','PercentSalaryHike','PerformanceRating',
        'RelationshipSatisfaction','StandardHours','StockOptionLevel',
        'TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance',
        'YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion',
        'YearsWithCurrManager'
    ]

    df_input = pd.DataFrame(data, columns=columns_name)

    st.markdown("---")
    if st.button("✨ Predict Attrition Status"):
        with st.spinner('Calculating...'):
            data_ready = data_preprocessing(df_input)
            prediction = model_predict(data_ready)

            if prediction is not None:
                if prediction[0] == 1:
                    st.error("### Result: Attrition - YES (Karyawan Berpotensi Keluar)")
                else:
                    st.success("### Result: Attrition - NO (Karyawan Berpotensi Bertahan)")

    # Footer
    year_now = datetime.date.today().year
    st.caption(f"Copyright © {year_now} Fahmi Fadillah")

if __name__ == "__main__":
    main()
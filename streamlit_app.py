import datetime
import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def data_preprocessing(data_input):
    df = pd.read_csv('employee_data_cleaned.csv')
    df = df.drop(columns=['EmployeeId', 'Attrition'], axis=1)
    df = pd.concat([df, data_input])

    numerical = df.select_dtypes(exclude='object').columns.tolist()
    categorical = df.select_dtypes(include='object').columns.tolist()

    df[categorical] = df[categorical].apply(LabelEncoder().fit_transform)
    df[numerical] = MinMaxScaler().fit_transform(df[numerical])

    return df.tail(1).to_numpy()


def model_predict(df):
    model = joblib.load('model_gb.joblib')
    return model.predict(df)


def prediction(output):
    if output == 1:
        st.error("Status Attrition: Yes")
    else:
        st.success("Status Attrition: No")


def main():

    st.title("Permasalahan Human Resource Dashboard")

    with st.container():
        col_gender, col_age, col_marital = st.columns(3)

        with col_gender:
            gender = st.radio("Gender", ["Male", "Female"])

        with col_age:
            age = st.number_input("Age", min_value=18, max_value=60)

        with col_marital:
            marital_status = st.selectbox(
                "Marital Status",
                ["Single", "Married", "Divorced"]
            )

    with st.container():

        col_education, col_edu_field = st.columns(2)

        with col_education:
            education = st.selectbox(
                "Education",
                ["Below College", "College", "Bachelor", "Master", "Doctor"]
            )

        with col_edu_field:
            education_field = st.selectbox(
                "Education Field",
                [
                    "Human Resources",
                    "Life Sciences",
                    "Marketing",
                    "Medical",
                    "Technical Degree",
                    "Other"
                ]
            )

    with st.container():

        col_distance, col_business_travel = st.columns(2)

        with col_distance:
            distance_from_home = st.number_input(
                "Distance From Home to Work (Km)", step=1
            )

        with col_business_travel:
            business_travel = st.selectbox(
                "Business Travel",
                ["Non-Travel", "Travel Rarely", "Travel Frequently"]
            )

    with st.container():

        col_dept, col_job_role, col_job_level = st.columns(3)

        with col_dept:
            department = st.selectbox(
                "Department",
                ["Human Resources", "Research & Development", "Sales"]
            )

        with col_job_role:
            job_role = st.selectbox(
                "Job Role",
                [
                    "Human Resources",
                    "Sales Executive",
                    "Sales Representative",
                    "Healthcare Representative",
                    "Research Scientist",
                    "Laboratory Technician",
                    "Manager",
                    "Manufacturing Director",
                    "Research Director"
                ]
            )

        with col_job_level:
            job_level = st.selectbox("Job Level", ["1", "2", "3", "4", "5"])

    with st.container():

        col_hourly_rate, col_daily_rate, col_monthly_rate = st.columns(3)

        with col_hourly_rate:
            hourly_rate = st.number_input("Hourly Rate", step=1)

        with col_daily_rate:
            daily_rate = st.number_input("Daily Rate", step=100)

        with col_monthly_rate:
            monthly_rate = st.number_input("Monthly Rate", step=1000)

    with st.container():

        col_monthly_income, col_percent_salary_hike = st.columns(2)

        with col_monthly_income:
            monthly_income = st.number_input("Monthly Income", step=100)

        with col_percent_salary_hike:
            percent_salary_hike = st.number_input(
                "Percent Salary Hike (%)", step=1
            )

    with st.container():

        col_standard_hours, col_over_time = st.columns(2)

        with col_standard_hours:
            standard_hours = st.number_input("Standard Hours", value=80)

        with col_over_time:
            over_time = "Yes" if st.checkbox("Over Time") else "No"

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

    df = pd.DataFrame(data, columns=[
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

        data_input = data_preprocessing(df)
        output = model_predict(data_input)

        prediction(output[0])

    year_now = datetime.date.today().year
    name = "Fahmi Fadillah"

    st.caption(f"Copyright © {year_now} {name}")


if __name__ == "__main__":
    main()
import pandas as pd
import streamlit as st
import plotly.express as px

# Konfigurasi Halaman
st.set_page_config(page_title="HR Employee Retention Dashboard", layout="wide")

# Load Data
df = pd.read_csv(r'C:\C:\Users\HP\OneDrive\Dokumen\dicoding\expert\test\data\employee_data.csv')

st.title("📊 HR Analytics Dashboard")
st.markdown("Dashboard ini menganalisis faktor utama penyebab karyawan keluar (Attrition).")

# Row 1: KPI
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Karyawan", len(df))
with col2:
    attrition_rate = (df['Attrition'].mean() * 100)
    st.metric("Attrition Rate", f"{attrition_rate:.2f}%")
with col3:
    st.metric("Rata-rata Gaji", f"${df['MonthlyIncome'].mean():.2f}")

# Row 2: Visualisasi
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Attrition berdasarkan Lembur (OverTime)")
    fig_ot = px.histogram(df, x="OverTime", color="Attrition", barmode="group")
    st.plotly_chart(fig_ot, use_container_width=True)

with col_right:
    st.subheader("Distribusi Gaji vs Attrition")
    fig_inc = px.box(df, x="Attrition", y="MonthlyIncome", color="Attrition")
    st.plotly_chart(fig_inc, use_container_width=True)

# Bagian Rekomendasi
st.info("💡 **Rekomendasi Action:** Perusahaan perlu meninjau beban kerja pada karyawan yang lembur karena memiliki korelasi tinggi dengan pengunduran diri.")
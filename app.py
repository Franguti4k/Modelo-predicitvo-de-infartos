
import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Cargar modelo
model = joblib.load("modelo_infarto.pkl")
scaler = joblib.load("scaler.pkl")

# Título
st.title("🩺 Predicción de Riesgo de Infarto")
st.write("Introduce los valores del paciente para estimar su riesgo de infarto.")

# Entradas
# Edad
Age = st.number_input(
    "Edad del paciente (años)",
    min_value=1,
    max_value=120,
    value=50
)
st.caption("Edad del paciente. El riesgo de enfermedad cardíaca aumenta con la edad.")

# Sexo
Sex = st.selectbox(
    "Sexo del paciente",
    ["Masculino", "Femenino"]
)
st.caption("Sexo del paciente. Los hombres suelen tener mayor riesgo en edades tempranas.")

# Tipo de dolor torácico
ChestPainType = st.selectbox(
    "Tipo de dolor torácico",
    ["TA - Angina típica", "ATA - Angina atípica", "NAP - Dolor no anginoso", "ASY - Asintomático"]
)
st.caption("El tipo de dolor puede indicar la severidad de la enfermedad cardíaca.")

# Presión arterial en reposo
RestingBP = st.number_input(
    "Presión arterial en reposo (mm Hg)",
    min_value=0,
    max_value=250,
    value=120
)
st.caption("La hipertensión es un factor de riesgo importante para enfermedad cardíaca.")

# Colesterol
Cholesterol = st.number_input(
    "Colesterol sérico (mg/dl)",
    min_value=0,
    max_value=600,
    value=200
)
st.caption("Colesterol alto puede causar aterosclerosis y aumentar riesgo de enfermedad cardíaca.")

# Glucemia en ayunas
FastingBS = st.selectbox(
    "¿Glucemia en ayunas > 120 mg/dl? (diabetes/prediabetes)",
    ["No", "Sí"]
)
st.caption("Valores altos de glucemia indican mayor riesgo de enfermedad cardíaca.")

# ECG en reposo
RestingECG = st.selectbox(
    "Resultados del electrocardiograma (ECG) en reposo",
    ["Normal", "ST - Anomalía ST-T", "LVH - Hipertrofia ventricular izquierda"]
)
st.caption("El ECG puede mostrar anomalías cardíacas estructurales o eléctricas.")

# Frecuencia cardíaca máxima
MaxHR = st.number_input(
    "Frecuencia cardíaca máxima alcanzada",
    min_value=60,
    max_value=220,
    value=150
)
st.caption("Valores anormales pueden indicar problemas de resistencia cardíaca.")

# Angina inducida por ejercicio
ExerciseAngina = st.selectbox(
    "¿Angina inducida por ejercicio?",
    ["No", "Sí"]
)
st.caption("Si el dolor aparece al hacer ejercicio, puede indicar enfermedad coronaria.")

# Oldpeak
Oldpeak = st.slider(
    "Depresión del segmento ST durante ejercicio (Oldpeak, mm)",
    min_value=0.0,
    max_value=6.0,
    value=0.0,
    step=0.1
)
st.caption("Mayor depresión ST indica mayor riesgo de enfermedad cardíaca.")

# ST_Slope
ST_Slope = st.selectbox(
    "Pendiente del segmento ST durante el ejercicio (ST_Slope)",
    ["Up - Ascendente", "Flat - Plano", "Down - Descendente"]
)
st.caption("Down indica mayor riesgo, Up indica menor riesgo.")

# Codificar variables categóricas
map_sexo = {
    "Masculino": 0,
    "Femenino": 1
}
map_chestPainType = {
    "ATA - Angina atípica": 0,
    "NAP - Dolor no anginoso": 1,
    "ASY - Asintomático": 2, 
    "TA - Angina típica": 3
}
map_azucar = {
    "No": 0,
    "Sí": 1
}
map_ecg = {
    "Normal": 0,
    "ST - Anomalía ST-T": 1,
    "LVH - Hipertrofia ventricular izquierda": 2
}
map_angina = {
    "No": 0,
    "Sí": 1
}

map_st_slope = {
    "Up - Ascendente": 0,
    "Flat - Plano": 1,
    "Down - Descendente": 2
}

Sex = map_sexo.get(Sex, -1)

ChestPainType = map_chestPainType.get(ChestPainType, -1)  

FastingBS = map_azucar.get(FastingBS, -1)  

RestingECG = map_ecg.get(RestingECG, -1)  

ExerciseAngina = map_angina.get(ExerciseAngina, -1)

ST_Slope = map_st_slope.get(ST_Slope, -1)       

# Columnas separadas segun tipo
columnas_num = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
columnas_cat = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# columnas en el orden exacto de entrenamiento
columnas_modelo = ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS',
                   'RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']

# Crear DataFrame con nombres de columnas
entrada_df = pd.DataFrame([[
    Age, RestingBP, Cholesterol, MaxHR, Oldpeak,
    Sex, ChestPainType, FastingBS, RestingECG, ExerciseAngina, ST_Slope
]], columns=columnas_num + columnas_cat)

# 6️⃣ Botón de predicción
if st.button("🔍 Predecir riesgo"):

    entrada_pred = entrada_df[columnas_modelo].copy()
    
    entrada_pred[columnas_num] = scaler.transform(entrada_pred[columnas_num])

    pred = model.predict(entrada_pred)[0]

    if pred == 1:
        st.error(f"⚠️ Alto riesgo de infarto. Le recomiendo que acuda a un médico.")
    else:
        st.success(f"✅ Bajo riesgo de infarto.")

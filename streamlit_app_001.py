import streamlit as st
import mlflow.sklearn
import numpy as np

st.title("Modelo de Predicción de Diabetes")
st.text("Este modelo se entrena con datos de la base de datos de Diabetes.")
st.text("Realizado por: David Burneo.")

# Configurar conexión con MLflow Tracking Server
mlflow.set_tracking_uri("http://localhost:9090") # cambiar en función de su servidor

# Cargar modelo desde el Model Registry, revise el ejemplo de flask
model = mlflow.sklearn.load_model("models:/diabetes_daburneo/1") # cambiar en función de su modelo

# Deslizadores para cada input del modelo
pregnancies = st.slider("Número de embarazos", 0, 20, 1)
glucose = st.slider("Nivel de glucosa", 0, 200, 100)
blood_pressure = st.slider("Presión arterial", 0, 122, 70)
skin_thickness = st.slider("Grosor del pliegue cutáneo", 0, 100, 20)
insulin = st.slider("Nivel de insulina", 0, 846, 80)
bmi = st.slider("Índice de masa corporal (BMI)", 0.0, 70.0, 25.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.slider("Edad", 10, 100, 30)

# Predicción
if st.button("Predecir"):
    entrada = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                         insulin, bmi, dpf, age]])
    pred = model.predict(entrada)[0]
    st.markdown("### Resultado:")
    st.success("Tiene diabetes" if pred == 1 else "No tiene diabetes")

st.text("Recuerde que los datos del presente modelo predictivo son de prueba.")

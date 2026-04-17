import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# configuración de la página
st.set_page_config(page_title="Prueba de Hipótesis", layout="wide")

# título de la página
st.title("Análisis de Datos")

# primer modulo: carga de datos
st.header("1 Generación de Datos")
st.markdown("Generar datos con distribución normal")

col1, col2, col3 = st.columns(3)
with col1:
    mean = st.number_input("Media poblacional", value=50.0)
with col2:
    std_dev = st.number_input("Desviación Estándar", value=10.0)
with col3: 
    n_samples = st.number_input("Tamaño de Muestras (n>=30)", min_value=30, value=1000)

if st.button("Generar Datos"):
    # muestra aleatoria usando variables
    data = np.random.normal(loc=mean, scale=std_dev, size=n_samples)
    # guarda para evitar que se borre al actualizar la página
    st.session_state['df'] = pd.DataFrame({'Variable': data})
    st.success(f"¡Se generaron {n_samples} datos correctamente!")

    # visualización
if 'df' in st.session_state:
    df = st.session_state['df']
    st.header("2 Visualización de la Distribución")
    
    # divide la pantalla en dos columnas para gráficas
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("Histograma y KDE")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['Variable'], kde=True, ax=ax1, color='skyblue')
        st.pyplot(fig1)
        
    with colB:
        st.subheader("Boxplot")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x=df['Variable'], ax=ax2, color='lightgreen')
        st.pyplot(fig2)
        
    # espacio para el análisis
    st.write("**Análisis del Estudiante:**")
    st.text_area("Observaciones:", 
                 placeholder="¿La distribución parece normal? ¿Hay sesgo o outliers? Escribe tu análisis aquí...")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import google.generativeai as genai # Importa la biblioteca de Gemini para análisis de datos

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

    # segundo modulo: visualización
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
    
# tercer modulo: prueba de hipótesis
if 'df' in st.session_state:
    df = st.session_state['df']
    st.header("3 Prueba de Hipótesis")
    st.markdown("Realizar una prueba de hipótesis para la media poblacional")
    
    # divide en tres columnas para los inputs
    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1:
        hypothesized_mean = st.number_input("Media Hipotética (H0)", value=50.0)
    with col_h2:
        alpha = st.number_input("Nivel de Significancia (α)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
    with col_h3:
        test_type = st.selectbox("Tipo de prueba", ["Bilateral", "Cola izquierda", "Cola derecha"])
        
    if st.button("Realizar Prueba de Hipótesis"):
        sample_mean = df['Variable'].mean()
        n = len(df)
        
        # cálculo del estadístico z con varianza conocida (std_dev del primer modulo)
        z_score = (sample_mean - hypothesized_mean) / (std_dev / np.sqrt(n))
        
        # ajuste del p-valor según el tipo de cola seleccionado
        if test_type == "Bilateral":
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        elif test_type == "Cola derecha":
            p_value = 1 - stats.norm.cdf(z_score)
        else: # cola izquierda
            p_value = stats.norm.cdf(z_score)
            
        reject_h0 = p_value < alpha
        
        st.write(f"**Media Muestral:** {sample_mean:.2f}")
        st.write(f"**Desviación Estándar (Poblacional):** {std_dev:.2f}")
        st.write(f"**Z-Score:** {z_score:.2f}")
        st.write(f"**P-Valor:** {p_value:.4f}")
        
        if reject_h0:
            st.error("Rechazamos la hipótesis nula (H0)")
        else:
            st.success("No se rechaza la hipótesis nula (H0)")
            
        # guarda diccionario con resultados para la api de gemini
        st.session_state['test_results'] = {
            'sample_mean': sample_mean, 'hypothesized_mean': hypothesized_mean, 'n': n, 
            'std_dev': std_dev, 'alpha': alpha, 'test_type': test_type, 
            'z_score': z_score, 'p_value': p_value, 'reject_h0': reject_h0
        }

# cuarto modulo: asistente de ia (gemini)
st.header("4 Asistente de IA")
st.markdown("Integración con Gemini para evaluar la decisión estadística")

# input seguro para la llave de la api
api_key = st.text_input("Ingresa tu API Key de Gemini:", type="password")

if st.button("Consultar a la IA"):
    # validacion para asegurar que exista la llave y los datos de la prueba
    if not api_key:
        st.warning("Por favor, ingresa tu API Key para continuar.")
    elif 'test_results' not in st.session_state:
        st.warning("Primero debes ejecutar la Prueba de Hipótesis en el módulo 3.")
    else:
        # configuracion de la api
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # recupera los resultados del modulo 3
        res = st.session_state['test_results']
        
        # armado del prompt usando el formato solicitado por el profesor
        prompt = f"""
        Se realizó una prueba Z con los siguientes parámetros:
        media muestral = {res['sample_mean']:.2f}, media hipotética = {res['hypothesized_mean']}, 
        n = {res['n']}, sigma = {res['std_dev']}, alpha = {res['alpha']}, tipo de prueba = {res['test_type']}.
        
        El estadístico Z fue = {res['z_score']:.4f} y el p-value = {res['p_value']:.4f}.
        
        ¿Se rechaza H0? Explica la decisión y si los supuestos de la prueba son razonables.
        """
        
        st.write("**Prompt enviado:**")
        st.info(prompt)
        
        # llamada a la api con indicador de carga
        with st.spinner("Gemini está analizando los resultados..."):
            try:
                respuesta = model.generate_content(prompt)
                st.write("**Respuesta de la IA:**")
                st.write(respuesta.text)
            except Exception as e:
                st.error(f"Error al conectar con la API: {e}")        
# 🌐 Dengue Prediction App

Una aplicación web basada en **Flask** para predecir casos de dengue y visualizar datos históricos y proyectados. ¡Ideal para profesionales de la salud pública y para la gestión proactiva de casos de dengue! 🎯

## 🚀 Funcionalidades Principales

### 🩺 **Predicción de Dengue**
- Introduce datos como edad, género, estrato y síntomas del paciente.
- Clasifica el caso en una de las siguientes categorías:
  - **0:** No aplica.
  - **1:** Dengue sin signos de alarma.
  - **2:** Dengue con signos de alarma.
  - **3:** Dengue grave.
- Ofrece recomendaciones personalizadas según la categoría del caso.

### 🛠️ **Entrenamiento del Modelo**
- Sube un archivo CSV con datos históricos para entrenar un modelo de clasificación.
- Manejo de valores faltantes mediante imputación o eliminación.
- Configuración personalizada de:
  - Capas ocultas del modelo.
  - Tasa de aprendizaje.
- Guarda el modelo entrenado para futuras predicciones.

### 📊 **Dashboard Interactivo**
- Visualiza tendencias y predicciones de casos de dengue:
  - Gráfica de tendencias con datos históricos y predicciones futuras.
  - Distribución por severidad de los casos (gráfica de pastel).
  - Mapa de calor mensual para casos esperados.
- Estadísticas clave:
  - Total de casos predichos.
  - Mes con mayor incidencia.
  - Nivel de riesgo promedio.

---

## 📊 **Datos Utilizados**

Este proyecto utiliza datos históricos de casos de dengue reportados en **Valledupar, Colombia**, disponibles públicamente a través de la plataforma del **Sistema Nacional de Vigilancia en Salud Pública (SIVIGILA)** del Ministerio de Salud y Protección Social de Colombia. Los datos abarcan el período desde **2016 hasta 2024** y están accesibles en el siguiente enlace oficial:

🔗 [Plataforma SIVIGILA - Casos de Dengue](https://www.minsalud.gov.co/sites/rid/paginas/freesearchresults.aspx?k=casos%20de%20dengue&scope=Todos)

---

## 🛠️ **Instalación**

### 1️⃣ Clona el repositorio
```bash
git clone https://github.com/Jassia627/prevencion-dengue.git
cd prevencion-dengue
```

### 2️⃣ Instala las dependencias
Asegúrate de tener Python 3.8+ instalado.
```bash
pip install -r requirements.txt
```

### 3️⃣ Ejecuta la aplicación
```bash
python app.py
```
Abre tu navegador y accede a: `http://127.0.0.1:5000/`


---

## 💻 **Uso de la Aplicación**

### 🌟 Predicción
1. Dirígete a la pestaña **Predict**.
2. Introduce los datos del paciente (edad, género, síntomas, etc.).
3. Obtén la predicción junto con recomendaciones específicas.

### 📈 Dashboard
1. Accede a la pestaña **Dashboard**.
2. Explora las tendencias y estadísticas relacionadas con los casos de dengue.

### 📚 Entrenamiento
1. Ve a la pestaña **Train**.
2. Sube un archivo CSV con datos históricos.
3. Configura las capas ocultas y la tasa de aprendizaje.
4. Entrena el modelo y guarda los resultados.

---

## 🧰 **Tecnologías Utilizadas**
- **Flask**: Framework web para Python.
- **Scikit-learn**: Modelado y preprocesamiento de datos.
- **Prophet**: Predicciones de series temporales.
- **Plotly**: Visualización interactiva de datos.
- **Joblib**: Serialización de modelos.

---


## ✉️ **creadores**
Si tienes alguna pregunta o sugerencia, ¡no dudes en contactarnos!
- **Juan Assia:** [GitHub](https://github.com/Jassia627)
- **Jose Chaparro:** [GitHub](https://github.com/Josechaparro09)

---

_“Predecir es prevenir. Con tecnología, podemos combatir el dengue de manera más eficiente.”_ 🌟

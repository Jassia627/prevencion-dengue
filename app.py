from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
import joblib
import json
import os

app = Flask(__name__)
app = Flask(__name__, static_url_path='/static', static_folder='imagenes')
app.secret_key = 'tu_clave_secreta_aqui'  # Necesario para flash messages
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit


# Asegurar que existe el directorio de uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class DenguePredictionSystem:
    def __init__(self):
        self.required_features = [
            'edad_', 'estrato_', 'sexo_', 'desplazami', 'famantdngu',
            'fiebre', 'cefalea', 'dolrretroo', 'malgias', 'artralgia',
            'erupcionr', 'dolor_abdo', 'vomito', 'diarrea', 'somnolenci'
        ]
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='most_frequent')
        self.numeric_imputer = SimpleImputer(strategy='mean')
        self.config_path = "model_config.json"
        self.load_model()

    def preprocess_data(self, X, training=True):
        """
        Preprocesa los datos manejando valores faltantes y codificación categórica
        """
        try:
            # Asegurar que todas las columnas requeridas estén presentes
            for feature in self.required_features:
                if feature not in X.columns:
                    if feature == 'sexo_':
                        X[feature] = 'M'  # valor por defecto
                    else:
                        X[feature] = '1'  # valor por defecto para variables binarias
            
            # Separar variables numéricas y categóricas
            numeric_cols = ['edad_', 'estrato_']
            categorical_cols = [col for col in X.columns if col not in numeric_cols]
            
            # Crear copias de los datos para cada tipo
            X_num = X[numeric_cols].copy()
            X_cat = X[categorical_cols].copy()
            
            if training:
                # Ajustar y transformar durante el entrenamiento
                X_num_imputed = self.numeric_imputer.fit_transform(X_num)
                X_cat_imputed = self.imputer.fit_transform(X_cat)
            else:
                # Solo transformar durante la predicción
                X_num_imputed = self.numeric_imputer.transform(X_num)
                X_cat_imputed = self.imputer.transform(X_cat)
            
            # Reconstruir el DataFrame
            X_num = pd.DataFrame(X_num_imputed, columns=numeric_cols, index=X.index)
            X_cat = pd.DataFrame(X_cat_imputed, columns=categorical_cols, index=X.index)
            
            # Combinar los DataFrames
            X_processed = pd.concat([X_num, X_cat], axis=1)
            
            # Codificación one-hot para variables categóricas
            X_encoded = pd.get_dummies(X_processed, columns=['sexo_'])
            
            if training:
                # Guardar las columnas para usar en predicción
                self.feature_columns = X_encoded.columns
            else:
                # Asegurar que todas las columnas del entrenamiento estén presentes
                for col in self.feature_columns:
                    if col not in X_encoded.columns:
                        X_encoded[col] = 0
                # Reordenar columnas para coincidir con el entrenamiento
                X_encoded = X_encoded[self.feature_columns]
            
            return X_encoded
            
        except Exception as e:
            raise Exception(f"Error en el preprocesamiento de datos: {str(e)}")

    def train_model(self, data, hidden_layers_str, learning_rate_str):
        try:
            self.data = data
            
            # Verificar si la columna clasfinal existe
            if 'clasfinal' not in self.data.columns:
                raise ValueError("La columna 'clasfinal' no existe en el dataset")
                
            # Mostrar información sobre valores faltantes antes de la limpieza
            total_rows = len(self.data)
            missing_rows = self.data['clasfinal'].isna().sum()
            
            # Eliminar filas con valores faltantes en clasfinal
            self.data = self.data.dropna(subset=['clasfinal'])
            
            # Si después de eliminar las filas vacías no quedan datos, lanzar error
            if len(self.data) == 0:
                raise ValueError("No hay datos válidos después de eliminar valores faltantes")
                
            # Convertir clasfinal a entero
            self.data['clasfinal'] = self.data['clasfinal'].astype(int)
            
            # Verificar valores únicos en clasfinal
            unique_values = self.data['clasfinal'].unique()
            if not all(val in [0, 1, 2, 3] for val in unique_values):
                invalid_values = [val for val in unique_values if val not in [0, 1, 2, 3]]
                raise ValueError(f"Valores no válidos encontrados en clasfinal: {invalid_values}. "
                               "Solo se permiten valores 0, 1, 2, 3")
            
            X = self.data[self.required_features].copy()
            y = self.data['clasfinal']

            X_processed = self.preprocess_data(X, training=True)
            X_scaled = self.scaler.fit_transform(X_processed)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            hidden_layers = tuple(map(int, hidden_layers_str.split(',')))
            learning_rate = float(learning_rate_str)

            self.model = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                learning_rate_init=learning_rate,
                max_iter=1000,
                random_state=42
            )
            
            self.model.fit(X_train, y_train)
            self.save_model()
            
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            return {
                'success': True,
                'train_score': train_score,
                'test_score': test_score,
                'total_rows': total_rows,
                'rows_removed': missing_rows,
                'rows_used': len(self.data),
                'message': f"Se eliminaron {missing_rows} filas con valores faltantes de un total de {total_rows} filas"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def predict(self, input_data):
        try:
            if self.model is None:
                raise ValueError("Modelo no entrenado")

            input_df = pd.DataFrame([input_data])
            input_processed = self.preprocess_data(input_df, training=False)
            input_scaled = self.scaler.transform(input_processed)
            prediction = self.model.predict(input_scaled)[0]
            
            prediction_map = {
                0: "No aplica",
                1: "Dengue sin signos de alarma",
                2: "Dengue con signos de alarma",
                3: "Dengue grave"
            }
            
            return {
                'success': True,
                'prediction': prediction_map.get(prediction, 'Desconocido')
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def save_model(self):
        try:
            # Save the model, scaler, and imputers
            joblib.dump(self.model, 'dengue_model.joblib')
            joblib.dump(self.scaler, 'scaler.joblib')
            joblib.dump(self.imputer, 'imputer.joblib')
            joblib.dump(self.numeric_imputer, 'numeric_imputer.joblib')
            
            # Save feature names and configuration
            config = {
                'feature_columns': list(self.feature_columns),
                'hidden_layers': self.hidden_layers_str if hasattr(self, 'hidden_layers_str') else '100,50',
                'learning_rate': self.learning_rate_str if hasattr(self, 'learning_rate_str') else '0.001'
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f)
                
        except Exception as e:
            raise Exception(f"Error al guardar el modelo: {str(e)}")

    def load_model(self):
        try:
            if (os.path.exists('dengue_model.joblib') and 
                os.path.exists('scaler.joblib') and 
                os.path.exists('imputer.joblib') and
                os.path.exists('numeric_imputer.joblib') and
                os.path.exists(self.config_path)):
                
                self.model = joblib.load('dengue_model.joblib')
                self.scaler = joblib.load('scaler.joblib')
                self.imputer = joblib.load('imputer.joblib')
                self.numeric_imputer = joblib.load('numeric_imputer.joblib')
                
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                self.feature_columns = config['feature_columns']
                self.hidden_layers_str = config.get('hidden_layers', '100,50')
                self.learning_rate_str = config.get('learning_rate', '0.001')
                
        except Exception as e:
            # Si hay error al cargar el modelo, inicializamos como None
            self.model = None
            print(f"No se pudo cargar el modelo: {str(e)}")

def get_recommendations(prediction):
    """
    Retorna recomendaciones específicas basadas en la predicción
    """
    general_recommendations = [
        "Mantener reposo en cama",
        "Tomar abundante líquido",
        "Usar mosquitero para evitar la transmisión",
        "Evitar la automedicación",
        "Seguir las indicaciones médicas"
    ]

    recommendations = {
        "No aplica": {
            "title": "Recomendaciones Preventivas",
            "severity": "info",
            "icon": "shield-alt",
            "specific": [
                "Mantenga la vigilancia de los síntomas",
                "Aplique medidas preventivas contra mosquitos",
                "Consulte si aparecen nuevos síntomas",
                "Mantenga limpio su entorno",
                "Use repelente regularmente"
            ]
        },
        "Dengue sin signos de alarma": {
            "title": "Cuidados para Dengue sin Signos de Alarma",
            "severity": "warning",
            "icon": "first-aid",
            "specific": [
                "Control diario de temperatura",
                "Paracetamol para la fiebre (NO aspirina)",
                "Hidratación oral frecuente",
                "Reposo absoluto",
                "Consulta médica de seguimiento en 48 horas"
            ]
        },
        "Dengue con signos de alarma": {
            "title": "Atención - Signos de Alarma",
            "severity": "danger",
            "icon": "exclamation-triangle",
            "specific": [
                "Busque atención médica inmediata",
                "No espere a que los síntomas empeoren",
                "Monitoreo constante de signos vitales",
                "Hidratación supervisada",
                "Posible necesidad de hospitalización"
            ]
        },
        "Dengue grave": {
            "title": "¡URGENTE! - Dengue Grave",
            "severity": "danger",
            "icon": "hospital",
            "specific": [
                "ACUDA INMEDIATAMENTE AL SERVICIO DE URGENCIAS",
                "Requiere hospitalización inmediata",
                "Necesita atención médica especializada",
                "Monitoreo intensivo necesario",
                "Tratamiento hospitalario urgente"
            ]
        }
    }

    return {
        "general": general_recommendations,
        "specific": recommendations.get(prediction, recommendations["No aplica"])
    }

# Instancia global del sistema de predicción
dengue_system = DenguePredictionSystem()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No se seleccionó ningún archivo')
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            flash('No se seleccionó ningún archivo')
            return redirect(request.url)

        if file:
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Leer el archivo CSV con manejo de errores
                try:
                    data = pd.read_csv(filepath)
                except Exception as e:
                    flash(f'Error al leer el archivo CSV: {str(e)}')
                    return redirect(request.url)
                
                hidden_layers = request.form.get('hidden_layers', '100,50')
                learning_rate = request.form.get('learning_rate', '0.001')
                
                result = dengue_system.train_model(data, hidden_layers, learning_rate)
                
                if result['success']:
                    flash(f'Modelo entrenado exitosamente.\n'
                          f'Precisión entrenamiento: {result["train_score"]:.2f}\n'
                          f'Precisión prueba: {result["test_score"]:.2f}\n'
                          f'Filas totales: {result["total_rows"]}\n'
                          f'Filas eliminadas: {result["rows_removed"]}\n'
                          f'Filas utilizadas: {result["rows_used"]}')
                else:
                    flash(f'Error en el entrenamiento: {result["error"]}')
                    
            except Exception as e:
                flash(f'Error: {str(e)}')
                
            finally:
                # Limpiar archivo temporal
                if os.path.exists(filepath):
                    os.remove(filepath)
                    
    return render_template('train.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            input_data = {
                'edad_': request.form.get('edad_'),
                'estrato_': request.form.get('estrato_'),
                'sexo_': request.form.get('sexo_'),
                'desplazami': request.form.get('desplazami'),
                'famantdngu': request.form.get('famantdngu'),
                'fiebre': request.form.get('fiebre'),
                'cefalea': request.form.get('cefalea'),
                'dolrretroo': request.form.get('dolrretroo'),
                'malgias': request.form.get('malgias'),
                'artralgia': request.form.get('artralgia'),
                'erupcionr': request.form.get('erupcionr'),
                'dolor_abdo': request.form.get('dolor_abdo'),
                'vomito': request.form.get('vomito'),
                'diarrea': request.form.get('diarrea'),
                'somnolenci': request.form.get('somnolenci')
            }
            
            result = dengue_system.predict(input_data)
            
            if result['success']:
                prediction = result['prediction']
                recommendations = get_recommendations(prediction)
                return render_template('predict.html', 
                                     show_result=True,
                                     prediction=prediction,
                                     recommendations=recommendations)
            else:
                flash(f'Error en la predicción: {result["error"]}')
                
        except Exception as e:
            flash(f'Error: {str(e)}')
            
    return render_template('predict.html', show_result=False)

if __name__ == '__main__':
    app.run(debug=True)
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
from flask import jsonify
import threading
import queue
import time
import speech_recognition as sr
from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
from plotly.subplots import make_subplots
import json

recording_threads = {}
audio_queues = {}
stop_flags = {}
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
@app.route('/get_microphones', methods=['GET'])
def get_microphones():
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        # Lista para almacenar los micrófonos filtrados
        mic_list = []
        seen_names = set()  # Para controlar duplicados
        
        # Palabras clave para identificar dispositivos de entrada válidos
        valid_keywords = ['mic', 'input', 'entrada', 'micrófono', 'microfono']
        # Palabras clave para excluir
        exclude_keywords = ['voicemeeter', 'virtual']
        
        for i in range(p.get_device_count()):
            try:
                device_info = p.get_device_info_by_index(i)
                device_name = device_info.get('name', '').lower()
                max_inputs = device_info.get('maxInputChannels', 0)
                
                # Solo incluir si:
                # 1. Tiene canales de entrada
                # 2. Contiene palabras clave de micrófono
                # 3. No contiene palabras clave de exclusión
                # 4. No es un duplicado
                if (max_inputs > 0 and 
                    any(keyword in device_name for keyword in valid_keywords) and
                    not any(keyword in device_name for keyword in exclude_keywords) and
                    device_name not in seen_names):
                    
                    # Limpiar y formatear el nombre del dispositivo
                    clean_name = (device_info.get('name', '')
                                .replace('Â', '')  # Eliminar caracteres especiales incorrectos
                                .replace('®', '')
                                .replace('©', '')
                                .strip())
                    
                    mic_list.append({
                        'index': i,
                        'name': clean_name
                    })
                    seen_names.add(device_name)
            
            except Exception as e:
                print(f"Error al procesar dispositivo {i}: {str(e)}")
        
        p.terminate()
        
        # Si no se encontraron micrófonos, incluir el dispositivo de entrada predeterminado
        if not mic_list:
            default_input = p.get_default_input_device_info()
            mic_list.append({
                'index': default_input['index'],
                'name': 'Micrófono predeterminado'
            })
        
        return jsonify({'microphones': mic_list})
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'microphones': [{'index': 0, 'name': 'Micrófono predeterminado'}]
        })

@app.route('/start_recording', methods=['POST'])
def start_recording():
    try:
        data = request.get_json()
        session_id = str(time.time())  # Identificador único para la sesión
        mic_index = data.get('microphoneIndex', 0)
        
        # Crear una cola para esta sesión
        audio_queue = queue.Queue()
        audio_queues[session_id] = audio_queue
        
        # Iniciar thread de grabación
        recording_thread = threading.Thread(
            target=record_audio,
            args=(mic_index, session_id, audio_queue)
        )
        recording_thread.daemon = True
        recording_threads[session_id] = recording_thread
        recording_thread.start()
        
        return jsonify({
            'success': True,
            'sessionId': session_id
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    try:
        data = request.get_json()
        session_id = data.get('sessionId')
        
        if session_id in recording_threads:
            del recording_threads[session_id]
        if session_id in audio_queues:
            del audio_queues[session_id]
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
        
@app.route('/get_transcription', methods=['POST'])
def get_transcription():
    try:
        data = request.get_json()
        session_id = data.get('sessionId')
        
        if session_id not in audio_queues:
            return jsonify({
                'success': False,
                'error': 'Sesión no encontrada'
            })
        
        audio_queue = audio_queues[session_id]
        
        # Verificar si hay audio en la cola
        if not audio_queue.empty():
            recognizer = sr.Recognizer()
            audio_data = audio_queue.get()
            
            try:
                text = recognizer.recognize_google(audio_data, language="es-ES")
                symptoms = analyze_symptoms(text.lower())
                
                return jsonify({
                    'success': True,
                    'text': text,
                    'symptoms': symptoms
                })
            except sr.UnknownValueError:
                return jsonify({
                    'success': True,
                    'text': '',
                    'symptoms': []
                })
        else:
            return jsonify({
                'success': True,
                'text': '',
                'symptoms': []
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def record_audio(mic_index, session_id, audio_queue):
    try:
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 200
        recognizer.pause_threshold = 0.8
        recognizer.phrase_threshold = 0.3
        recognizer.non_speaking_duration = 0.4
        
        with sr.Microphone(device_index=mic_index) as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            while session_id in recording_threads:
                try:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=None)
                    audio_queue.put(audio)
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    print(f"Error en grabación: {str(e)}")
                    break
    except Exception as e:
        print(f"Error iniciando grabación: {str(e)}")

def analyze_symptoms(text):
    # Definir las palabras clave para cada síntoma (similar al script original)
    symptom_keywords = {
            'Cefalea': [
                # Términos médicos
                'cefalea', 'migraña', 'jaqueca', 'dolor craneal',
                # Expresiones coloquiales
                'dolor de cabeza', 'me duele la cabeza', 'tengo la cabeza como un bombo',
                'siento que me va a estallar la cabeza', 'me está taladrando la cabeza',
                'me late la cabeza', 'siento pulsaciones en la cabeza',
                'parece que me pegaron en la cabeza', 'tengo la cabeza pesada',
                'me está matando la cabeza', 'no aguanto el dolor de cabeza',
                'se me parte la cabeza', 'tengo un dolor horrible de cabeza',
                'siento que me golpean la cabeza', 'me duele hasta el pelo',
                'tengo un martilleo en la cabeza', 'me va a explotar la cabeza',
                'me está reventando la cabeza', 'siento presión en la cabeza',
                'es como si tuviera la cabeza en una prensa', 'dolor de cráneo',
                'me duele toda la cabeza', 'tengo la cabeza hecha un lío',
                'no soporto el dolor de cabeza', 'me está dando migraña'
            ],
            
            'Dolor Retroocular': [
                # Términos médicos
                'dolor retroocular', 'dolor periorbitario', 'presión intraocular',
                # Expresiones coloquiales
                'me duelen los ojos', 'siento presión detrás de los ojos',
                'me duele atrás de los ojos', 'me arden los ojos con dolor',
                'siento como si me apretaran los ojos', 'me duele al mover los ojos',
                'siento los ojos pesados', 'me duele cuando muevo la vista',
                'siento como arena en los ojos', 'me duelen los ojos por dentro',
                'tengo dolor en el fondo de los ojos', 'me duele hasta parpadear',
                'siento presión en los globos oculares', 'me duele la parte de atrás de los ojos',
                'me lastima la luz', 'me duelen los ojos cuando los muevo',
                'siento que me aprietan los ojos', 'me duele hasta el movimiento de los ojos',
                'siento como si tuviera los ojos hinchados por dentro',
                'me duele la zona de los ojos', 'tengo dolor alrededor de los ojos',
                'siento punzadas en los ojos','dolor en los ojos'
            ],
            
            'Mialgias': [
                # Términos médicos
                'mialgia', 'dolor muscular', 'fatiga muscular',
                # Expresiones coloquiales
                'me duele todo el cuerpo', 'siento el cuerpo cortado',
                'tengo molido el cuerpo', 'me duelen todos los músculos',
                'siento como si me hubieran golpeado', 'estoy todo adolorido', 'tengo el cuerpo destrozado',
                'parece que me pasó un tren por encima', 'me duele todo',
                'siento el cuerpo pesado', 'tengo dolor hasta en los dedos',
                'me duele al moverme', 'estoy entumecido', 'todo me duele',
                'siento los músculos tensionados', 'tengo el cuerpo hecho polvo',
                'me duele hasta respirar', 'siento que me apalearon',
                'tengo todos los músculos adoloridos', 'no puedo ni moverme del dolor',
                'estoy todo contracturado', 'me duele cada parte del cuerpo',
                'siento como si hubiera corrido un maratón', 'tengo agujetas en todo el cuerpo',
                'estoy machacado', 'me duele hasta la punta del pelo'
            ],
            
            'Artralgias': [
                # Términos médicos
                'artralgia', 'dolor articular', 'inflamación articular',
                # Expresiones coloquiales
                'me duelen las articulaciones', 'me duelen las coyunturas',
                'no puedo doblar las rodillas', 'me duelen los huesos',
                'me crujen las articulaciones', 'tengo las articulaciones tiesas',
                'me duelen los nudos de los dedos', 'no puedo cerrar el puño',
                'me duelen las muñecas', 'siento las articulaciones inflamadas',
                'me duele al doblar los brazos', 'tengo las rodillas hinchadas',
                'me duelen las juntas', 'no puedo ni agarrar nada',
                'me duele al mover los dedos', 'siento rigidez en las articulaciones',
                'me duelen los tobillos', 'tengo dolor en todas las uniones',
                'me duelen hasta los dedos de los pies', 'no puedo flexionar las piernas',
                'siento las articulaciones calientes', 'me duele al hacer movimientos',
                'tengo las articulaciones como oxidadas', 'me duele al doblar cualquier parte'
            ],
            
            'Erupción/Rash': [
                # Términos médicos
                'erupción cutánea', 'rash', 'urticaria', 'dermatitis','manchas',
                # Expresiones coloquiales
                'tengo sarpullido', 'me salieron manchas', 'tengo la piel irritada',
                'me salieron ronchas', 'tengo granitos', 'me pica todo el cuerpo',
                'tengo la piel manchada', 'me salió alergia en la piel',
                'tengo brotes en la piel', 'me salieron puntos rojos',
                'tengo la piel con salpullido', 'me apareció un sarpullido',
                'tengo manchas que pican', 'me salieron granos por todos lados',
                'tengo la piel con erupciones', 'me salió una roncha',
                'tengo la piel irritada y con manchas', 'me pica y tengo manchas',
                'me salieron como picaduras', 'tengo todo el cuerpo con puntos',
                'parece que me picaron mosquitos', 'tengo la piel sensible y con manchas',
                'me salieron ampollas', 'tengo la piel enrojecida y con brotes'
            ],
            
            'Dolor Abdominal': [
                # Términos médicos
                'dolor abdominal', 'dolor epigástrico', 'malestar gastrointestinal',
                # Expresiones coloquiales
                'me duele la panza', 'tengo dolor de estómago', 'me duele la barriga',
                'tengo retortijones', 'siento punzadas en el estómago',
                'me duele aquí en la boca del estómago', 'tengo cólicos',
                'siento como si me apretaran el estómago', 'me duele todo el abdomen',
                'tengo el estómago revuelto', 'siento ardor en el estómago',
                'me duele después de comer', 'tengo gases con dolor',
                'siento como si tuviera piedras en el estómago', 'me duele al tacto la panza',
                'tengo el estómago inflamado', 'siento pinchazos en la barriga',
                'me duele aquí abajo', 'tengo dolor en la boca de la panza',
                'siento como si me hubieran pateado el estómago',
                'me duele todo el vientre', 'tengo torcijones',
                'siento náuseas y dolor de estómago', 'me duele hasta cuando respiro'
            ],
            
            'Vómito': [
                # Términos médicos
                'vómito', 'emesis', 'náusea', 'regurgitación',
                # Expresiones coloquiales
                'estoy vomitando', 'tengo ganas de vomitar', 'no paro de vomitar',
                'devuelvo todo lo que como', 'me dan náuseas', 'tengo ascos',
                'siento que voy a vomitar', 'me dan arcadas', 'estoy con náuseas',
                'no puedo retener nada', 'todo me da asco', 'vomito hasta el agua',
                'me dan ganas de devolver', 'tengo el estómago revuelto',
                'siento que se me revuelve el estómago', 'me dan heaves',
                'no puedo ni ver la comida', 'siento asco de todo',
                'cada vez que como vomito', 'me dan ganitas',
                'siento que se me viene todo', 'no puedo dejar de vomitar',
                'me dan náuseas todo el tiempo', 'tengo el estómago mal'
            ],
            
            'Diarrea': [
                # Términos médicos
                'diarrea', 'deposiciones líquidas', 'gastroenteritis','barriga','estomago',
                # Expresiones coloquiales
                'estoy suelto del estómago', 'tengo el estómago flojo',
                'voy mucho al baño', 'tengo el intestino suelto',
                'no puedo dejar de ir al baño', 'tengo colitis',
                'tengo mal de estómago', 'estoy descompuesto',
                'tengo el estómago revuelto', 'no puedo alejarme del baño',
                'tengo problemas estomacales', 'voy al baño a cada rato',
                'tengo diarrea fuerte', 'no me para el estómago',
                'tengo el estómago destemplado', 'estoy mal del estómago',
                'tengo que correr al baño', 'no me aguanto',
                'tengo urgencia por ir al baño', 'tengo el estómago descompuesto',
                'no puedo controlar el estómago', 'estoy con la fuente abierta',
                'tengo el intestino irritado', 'estoy con problemas intestinales'
            ],
            
            'Somnolencia': [
                # Términos médicos
                'somnolencia', 'fatiga', 'astenia', 'letargo','sueño',
                # Expresiones coloquiales
                'estoy muy cansado', 'me muero de sueño', 'no tengo energía',
                'estoy agotado', 'no me puedo mantener despierto',
                'me siento sin fuerzas', 'estoy que me caigo de sueño',
                'no doy más del cansancio', 'estoy exhausto',
                'no me puedo ni levantar', 'me siento débil',
                'no tengo ganas de nada', 'estoy todo el día con sueño',
                'me siento como zombie', 'no me puedo concentrar del sueño',
                'me pesan los párpados', 'siento el cuerpo pesado',
                'no tengo ánimo para nada', 'estoy muy decaído',
                'me siento sin energía', 'no puedo ni mantenerme en pie',
                'siento que podría dormir todo el día', 'estoy que me desmayo del cansancio',
                'no me puedo mantener activo', 'me siento como si no hubiera dormido'
            ],
            
            'Fiebre': [
                # Términos médicos
                'fiebre', 'hipertermia', 'pirexia', 'febrícula',
                # Expresiones coloquiales
                'tengo temperatura', 'estoy afiebrado', 'tengo calentura',
                'estoy ardiendo', 'me siento caliente', 'tengo el cuerpo caliente',
                'estoy que ardo', 'tengo escalofríos con fiebre',
                'me sube y baja la temperatura', 'estoy volando en fiebre',
                'tengo fiebre alta', 'me siento como si ardiera',
                'tengo el cuerpo hirviendo', 'estoy que quemo',
                'me siento acalorado', 'tengo temperatura alta',
                'siento que me quemo', 'estoy con mucha fiebre',
                'tengo décimas', 'me siento afiebrado',
                'estoy con calentura', 'siento el cuerpo muy caliente',
                'tengo fiebre con escalofríos', 'me sube la temperatura'
            ],
            
            'Hipotermia': [
                # Términos médicos
                'hipotermia', 'temperatura corporal baja', 'termometría baja',
                # Expresiones coloquiales
                'estoy helado', 'tengo mucho frío', 'no entro en calor',
                'siento el cuerpo frío', 'me tiembla todo del frío',
                'estoy congelado', 'no puedo entrar en calor',
                'tengo escalofríos', 'siento frío por dentro',
                'estoy temblando de frío', 'tengo el cuerpo frío',
                'me castañean los dientes', 'siento que me congelo',
                'no me puedo calentar', 'tengo un frío que me cala los huesos',
                'estoy que tiemblo', 'siento frío hasta en los huesos',
                'no hay manera de calentarme', 'me tiembla todo el cuerpo',
                'siento un frío interno', 'estoy más frío que un témpano',
                'tengo la piel fría', 'no logro entrar en calor',
                'siento que me muero de frío'
            ]
        }
    
    detected_symptoms = []
    for symptom, keywords in symptom_keywords.items():
        if any(keyword in text for keyword in keywords):
            detected_symptoms.append(symptom)
    
    return detected_symptoms

def load_and_process_data():
    try:
        # Leer el CSV con todas las columnas como tipo string inicialmente
        df = pd.read_csv('datos.csv', sep=';', dtype=str)
        
        # Convertir columnas numéricas explícitamente
        numeric_columns = ['semana', 'año', 'edad_', 'estrato_']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convertir fecha a datetime
        df['fec_not'] = pd.to_datetime(df['fec_not'], format='%d/%m/%Y', errors='coerce')
        
        # Convertir columnas booleanas
        bool_columns = ['famantdngu', 'fiebre', 'cefalea', 'dolrretroo', 'malgias', 
                       'artralgia', 'erupcionr', 'dolor_abdo', 'vomito', 'diarrea', 
                       'somnolenci', 'conducta']
        
        for col in bool_columns:
            df[col] = df[col].map({'true': True, 'false': False})
        
        return df
    
    except Exception as e:
        print(f"Error al cargar los datos: {str(e)}")
        # Devolver un DataFrame vacío o con estructura mínima en caso de error
        return pd.DataFrame(columns=['fec_not', 'semana', 'año', 'edad_', 'sexo_', 
                                   'bar_ver_', 'dir_res_', 'estrato_', 'tip_cas_'])

def create_kpis(df, year=None):
    try:
        if year:
            df = df[df['año'] == year]
        
        # Asegurarse de que edad_ es numérica
        df['edad_'] = pd.to_numeric(df['edad_'], errors='coerce')
        
        # KPIs básicos
        total_casos = len(df)
        casos_confirmados = len(df[df['tip_cas_'].str.contains('Confirmado', na=False)])
        edad_promedio = df['edad_'].mean()
        
        # Distribución por género
        distribucion_genero = df['sexo_'].value_counts(normalize=False).to_dict()
        porcentajes = df['sexo_'].value_counts(normalize=True).mul(100).round(1).to_dict()
        # Porcentaje de casos por tipo
        tipos_casos = df['tip_cas_'].value_counts().to_dict()
        
        clasfinal_distribucion = df['clasfinal'].value_counts()
        
        clasfinal_porcentajes = (clasfinal_distribucion / total_casos * 100).round(1)
        
        clasfinal_stats = {
            'distribucion': clasfinal_distribucion.to_dict(),
            'porcentajes': clasfinal_porcentajes.to_dict()
        }
        # Top 5 barrios
        top_barrios = df['bar_ver_'].value_counts().head(5).to_dict()
        
        # Casos por estrato
        casos_estrato = df['estrato_'].value_counts().sort_index().to_dict()
        
        # Nuevos KPIs
        casos_semanales_promedio = df.groupby('semana').size().mean()
        edad_mediana = df['edad_'].median()
        casos_pediatricos = len(df[df['edad_'] <= 18])
        porcentaje_pediatricos = (casos_pediatricos / total_casos * 100) if total_casos > 0 else 0
        
        return {
            'total_casos': total_casos,
            'casos_confirmados': casos_confirmados,
            'edad_promedio': round(edad_promedio, 1) if not pd.isna(edad_promedio) else 0,
            'edad_mediana': round(edad_mediana, 1) if not pd.isna(edad_mediana) else 0,
            'casos_semanales_promedio': round(casos_semanales_promedio, 1) if not pd.isna(casos_semanales_promedio) else 0,
            'distribucion_genero': distribucion_genero,
            'tipos_casos': tipos_casos,
            'casos_estrato': casos_estrato,
            'casos_pediatricos': casos_pediatricos,
            'porcentaje_pediatricos': round(porcentaje_pediatricos, 1),
            'top_barrios': top_barrios,
            'clasfinal_stats': clasfinal_stats
        }
        
    except Exception as e:
        print(f"Error al crear KPIs: {str(e)}")
        return {
            'total_casos': 0,
            'casos_confirmados': 0,
            'edad_promedio': 0,
            'edad_mediana': 0,
            'casos_semanales_promedio': 0,
            'distribucion_genero': {},
            'tipos_casos': {},
            'casos_estrato': {},
            'casos_pediatricos': 0,
            'porcentaje_pediatricos': 0,
            'top_barrios': {}
        }

def create_plots(df, year=None):
    try:
        if year:
            df = df[df['año'] == year]
        
        # Asegurarse de que edad_ es numérica
        df['edad_'] = pd.to_numeric(df['edad_'], errors='coerce')
        
        # Configuración de colores y layout base
        colors = ['#0d6efd',  # Bootstrap primary
                 '#6610f2',  # Bootstrap indigo
                 '#6f42c1',  # Bootstrap purple
                 '#d63384',  # Bootstrap pink
                 '#dc3545',  # Bootstrap red
                 '#fd7e14',  # Bootstrap orange
                 '#198754']  # Bootstrap green
                 
        layout_template = {
            'paper_bgcolor': '#f8f9fa',  # Fondo gris claro como el sitio
            'plot_bgcolor': '#ffffff',   # Fondo blanco para los gráficos
            'font': {'color': '#212529'},  # Color de texto Bootstrap
            'xaxis': {
                'gridcolor': '#e9ecef',  # Color de líneas de grilla Bootstrap
                'linecolor': '#dee2e6'    # Color de líneas de eje Bootstrap
            },
            'yaxis': {
                'gridcolor': '#e9ecef',
                'linecolor': '#dee2e6'
            }
        }
        
        # 1. Casos por semana
        casos_semana = df.groupby(['año', 'semana']).size().reset_index(name='casos')
        casos_semana['periodo'] = casos_semana.apply(lambda x: f"{x['año']}-S{x['semana']:02d}", axis=1)
        
        fig_casos_semana = px.line(
            casos_semana,
            x='periodo',
            y='casos',
            title='Casos por Semana Epidemiológica',
            template='plotly_white',
            color_discrete_sequence=[colors[0]]
        )
        fig_casos_semana.update_layout(**layout_template)
        fig_casos_semana.update_traces(line_width=3)
        
        # 2. Distribución por edad con segmentación por género
        fig_edad = px.histogram(
            df.dropna(subset=['edad_']),
            x='edad_',
            color='sexo_',
            nbins=20,
            title='Distribución por Edad y Género',
            template='plotly_white',
            color_discrete_sequence=[colors[1], colors[2]]
        )
        fig_edad.update_layout(**layout_template)
        fig_edad.update_traces(
            opacity=0.75,
            texttemplate='%{y}',
            textposition='outside'
        )

        # 3. Gráfico de torta para género
        fig_genero = px.pie(
            df['sexo_'].value_counts().reset_index(),
            values='count',
            names='sexo_',
            title='Casos de dengue por género',
            template='plotly_white',
            color_discrete_sequence=[colors[1], colors[2]],
            hole=0.4
        )
        fig_genero.update_layout(**layout_template)
        fig_genero.update_traces(
            textinfo='percent+label+value',
            textposition='outside',
            marker=dict(line=dict(color='#ffffff', width=2))
        )
        
        # 4. Clasificación Final
        fig_clasfinal = px.pie(
            df['clasfinal'].value_counts().reset_index(),
            values='count',
            names='clasfinal',
            title='Clasificación Final',
            template='plotly_white',
            color_discrete_sequence=colors,
            hole=0.4
        )
        fig_clasfinal.update_layout(**layout_template)
        fig_clasfinal.update_traces(
            textinfo='percent+label+value',
            textposition='outside',
            marker=dict(line=dict(color='#ffffff', width=2))
        )
        
        # 5. Top 5 Barrios
        top_barrios = df['bar_ver_'].value_counts().head(5).reset_index()
        fig_barrios = px.bar(
            top_barrios,
            y='bar_ver_',
            x='count',
            title='Top 5 Barrios con Más Casos',
            template='plotly_white',
            color_discrete_sequence=[colors[4]],
            orientation='h'
        )
        fig_barrios.update_layout(**layout_template)
        fig_barrios.update_traces(
            texttemplate='%{x}',
            textposition='outside',
            marker_line_color='#ffffff',
            marker_line_width=1,
            opacity=0.8
        )

        # 6. Grupos de edad
        df['grupo_edad'] = pd.cut(
            df['edad_'],
            bins=[0, 12, 17, 35, 59, float('inf')],
            labels=['Niñez', 'Adolescencia', 'Adulto Joven', 'Adulto', 'Mayor']
        )
        casos_grupos = df['grupo_edad'].value_counts().reset_index()
        casos_grupos.columns = ['grupo_edad', 'cantidad']
        fig_grupos_edad = px.bar(
            casos_grupos,
            x='grupo_edad',
            y='cantidad',
            title='Casos por Grupos de Edad',
            template='plotly_white',
            color_discrete_sequence=[colors[5]]
        )
        fig_grupos_edad.update_layout(**layout_template)
        fig_grupos_edad.update_traces(
            texttemplate='%{y}',
            textposition='outside',
            marker_line_color='#ffffff',
            marker_line_width=1,
            opacity=0.8
        )
        
        # 7. Frecuencia de síntomas
        sintomas_columns = [
            'famantdngu', 'fiebre', 'cefalea', 'dolrretroo', 'malgias', 
            'artralgia', 'erupcionr', 'dolor_abdo', 'vomito', 'diarrea', 
            'somnolenci', 'conducta'
        ]
        frecuencia_sintomas = df[sintomas_columns].sum().sort_values(ascending=False).head(5).reset_index()
        frecuencia_sintomas.columns = ['sintoma', 'frecuencia']
        
        fig_frecuencia_sintomas = px.bar(
            frecuencia_sintomas,
            x='sintoma',
            y='frecuencia',
            title='Top 5 síntomas más frecuentes',
            template='plotly_white',
            color_discrete_sequence=[colors[6]]
        )
        fig_frecuencia_sintomas.update_layout(**layout_template)
        fig_frecuencia_sintomas.update_traces(
            texttemplate='%{y}',
            textposition='outside',
            marker_line_color='#ffffff',
            marker_line_width=1,
            opacity=0.8
        )
        
        # Convertir las figuras a JSON
        plots = {
            'casos_semana': json.dumps(fig_casos_semana, cls=plotly.utils.PlotlyJSONEncoder),
            'edad': json.dumps(fig_edad, cls=plotly.utils.PlotlyJSONEncoder),
            'genero': json.dumps(fig_genero, cls=plotly.utils.PlotlyJSONEncoder),
            'barrios': json.dumps(fig_barrios, cls=plotly.utils.PlotlyJSONEncoder),
            'grupos_edad': json.dumps(fig_grupos_edad, cls=plotly.utils.PlotlyJSONEncoder),
            'frecuencia_sintomas': json.dumps(fig_frecuencia_sintomas, cls=plotly.utils.PlotlyJSONEncoder),
            'clasfinal_pie': json.dumps(fig_clasfinal, cls=plotly.utils.PlotlyJSONEncoder)
        }
        
        return plots
    
    except Exception as e:
        print(f"Error al crear gráficos: {str(e)}")
        return {}
    
    

@app.route('/dashboard')
def dashboard():
    try:
        df = load_and_process_data()
        # Obtener el año del query parameter
        year = request.args.get('year', type=int)
        years_available = sorted(df['año'].unique().tolist())

        # Si no hay año seleccionado, usar los datos completos
        if year is not None:
            df_filtered = df[df['año'] == year]
            kpis = create_kpis(df_filtered)
            plots = create_plots(df_filtered)
        else:
            kpis = create_kpis(df)
            plots = create_plots(df)

        return render_template('dashboard.html', 
                             kpis=kpis, 
                             plots=plots, 
                             years=years_available, 
                             selected_year=year)

    except Exception as e:
        print(f"Error en la ruta principal: {str(e)}")
        return "Error al cargar el dashboard"

if __name__ == '__main__':
    app.run(debug=True)
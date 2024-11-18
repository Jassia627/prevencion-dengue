from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
from plotly.subplots import make_subplots
import json
from datetime import datetime

app = Flask(__name__)

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
    
    

@app.route('/')
def dashboard():
    try:
        df = load_and_process_data()
        year = request.args.get('year', type=int)
        years_available = sorted(df['año'].unique().tolist())
        
        kpis = create_kpis(df, year)
        plots = create_plots(df, year)
        
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
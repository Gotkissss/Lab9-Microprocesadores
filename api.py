# LAB9-MICROPROCESADORES/api.py
# Backend Flask para Smart Agriculture Dashboard
# Desarrollado por: Diego (Persona 2)

from flask import Flask, jsonify, request
from flask_cors import CORS
import sqlite3
import os

app = Flask(__name__)
CORS(app)  # Permitir requests desde el navegador

# Ruta a la base de datos
DB_PATH = os.path.join('data', 'agriculture.db')

@app.route('/')
def home():
    return jsonify({
        'message': 'Smart Agriculture API',
        'endpoints': [
            '/api/sensors/latest',
            '/api/sensors/stats',
            '/api/alerts'
        ]
    })

@app.route('/api/sensors/latest', methods=['GET'])
def get_latest_sensors():
    """Obtiene los últimos 100 registros de sensores"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, humedad_suelo, temperatura, luz, humedad_aire, ph
            FROM sensor_readings
            ORDER BY id DESC
            LIMIT 100
        """)
        
        data = cursor.fetchall()
        conn.close()
        
        # Formatear respuesta (convertir a porcentaje donde aplique)
        result = [{
            'timestamp': row[0],
            'humedad_suelo': round(row[1] * 100, 1),
            'temperatura': round(row[2], 1),
            'luz': round(row[3], 1),
            'humedad_aire': round(row[4] * 100, 1),
            'ph': round(row[5], 2)
        } for row in data]
        
        # Invertir para mostrar del más antiguo al más reciente
        result.reverse()
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sensors/stats', methods=['GET'])
def get_sensor_stats():
    """Calcula estadísticas de los sensores"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Últimos 200 registros para estadísticas
        cursor.execute("""
            SELECT 
                AVG(humedad_suelo) as avg_hum_suelo,
                AVG(temperatura) as avg_temp,
                AVG(luz) as avg_luz,
                AVG(humedad_aire) as avg_hum_aire,
                AVG(ph) as avg_ph,
                MIN(temperatura) as min_temp,
                MAX(temperatura) as max_temp,
                MIN(humedad_suelo) as min_hum,
                MAX(humedad_suelo) as max_hum
            FROM (
                SELECT * FROM sensor_readings 
                ORDER BY id DESC 
                LIMIT 200
            )
        """)
        
        stats = cursor.fetchone()
        conn.close()
        
        return jsonify({
            'promedio': {
                'humedad_suelo': round(stats[0] * 100, 1),
                'temperatura': round(stats[1], 1),
                'luz': round(stats[2], 1),
                'humedad_aire': round(stats[3] * 100, 1),
                'ph': round(stats[4], 2)
            },
            'rangos': {
                'temperatura': {
                    'min': round(stats[5], 1), 
                    'max': round(stats[6], 1)
                },
                'humedad_suelo': {
                    'min': round(stats[7] * 100, 1), 
                    'max': round(stats[8] * 100, 1)
                }
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Detecta y devuelve alertas basadas en umbrales"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT humedad_suelo, temperatura, luz, humedad_aire, ph
            FROM sensor_readings
            ORDER BY id DESC
            LIMIT 1
        """)
        
        latest = cursor.fetchone()
        conn.close()
        
        if not latest:
            return jsonify([])
        
        alerts = []
        
        # Humedad del suelo crítica
        if latest[0] < 0.25:
            alerts.append({
                'level': 'CRÍTICO',
                'message': f'Humedad del suelo crítica: {latest[0]*100:.1f}%',
                'sensor': 'humedad_suelo',
                'value': round(latest[0] * 100, 1)
            })
        elif latest[0] < 0.35:
            alerts.append({
                'level': 'ADVERTENCIA',
                'message': f'Humedad del suelo baja: {latest[0]*100:.1f}%',
                'sensor': 'humedad_suelo',
                'value': round(latest[0] * 100, 1)
            })
        
        # Temperatura alta
        if latest[1] > 32:
            alerts.append({
                'level': 'ADVERTENCIA',
                'message': f'Temperatura alta: {latest[1]:.1f}°C',
                'sensor': 'temperatura',
                'value': round(latest[1], 1)
            })
        
        # pH fuera de rango
        if latest[4] < 5.5 or latest[4] > 7.5:
            alerts.append({
                'level': 'ADVERTENCIA',
                'message': f'pH fuera de rango óptimo: {latest[4]:.1f}',
                'sensor': 'ph',
                'value': round(latest[4], 2)
            })
        
        return jsonify(alerts)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Verificar que existe la base de datos
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: No se encontró {DB_PATH}")
        print("   Ejecuta 'python database.py' primero")
        exit(1)
    
    print("="*60)
    print("🚀 Smart Agriculture API Server")
    print("="*60)
    print(f"📊 Base de datos: {DB_PATH}")
    print(f"🌐 Servidor: http://localhost:5000")
    print("\n📡 Endpoints disponibles:")
    print("   - GET /api/sensors/latest  (últimos 100 registros)")
    print("   - GET /api/sensors/stats   (estadísticas)")
    print("   - GET /api/alerts          (alertas del sistema)")
    print("\n💡 Prueba: http://localhost:5000/api/sensors/latest")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')
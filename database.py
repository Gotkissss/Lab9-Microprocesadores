import sqlite3
import csv
import os

# --- Configuración ---
# Usamos las rutas sugeridas por la estructura del proyecto (data/agriculture.db y data/sensor_log.csv)
DATA_DIR = 'data'
DATABASE_FILE = os.path.join(DATA_DIR, 'agriculture.db')
CSV_FILE = os.path.join(DATA_DIR, 'sensor_log.csv')

# --- Funciones de Conexión y Setup ---

def get_db_connection():
    """establece la conexión a la base de datos."""
    # Asegura que el directorio 'data' exista
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    conn = sqlite3.connect(DATABASE_FILE)
    conn.isolation_level = None # Modo de autocommit para simplificar
    return conn

def create_tables():
    """
    crea todas las tablas necesarias en la base de datos.
    - sensor_readings: almacena las mediciones de los 5 canales de sensores.
    - actuator_events: almacena los eventos de control.
    - alerts: almacena las notificaciones de eventos críticos o de diagnóstico.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 1. tabla de Lecturas de Sensores
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                humedad_suelo REAL,
                temperatura REAL,
                luz REAL,
                humedad_aire REAL,
                ph REAL
            );
        ''')

        # 2. Tabla de Eventos de Actuadores (Riego, Ventilación)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS actuator_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                actuator TEXT NOT NULL,
                action TEXT NOT NULL, 
                value REAL,
                status TEXT
            );
        ''')

        # 3. Tabla de Alertas y Diagnósticos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL, -- 'INFO', 'WARNING', 'CRITICAL'
                message TEXT NOT NULL,
                source TEXT -- 'system', 'ph_range', 'temperature_range'
            );
        ''')
        
        print("Tablas creadas (sensor_readings, actuator_events, alerts) en 'agriculture.db'.")

    except sqlite3.Error as e:
        print(f"Error al crear tablas: {e}")
    finally:
        if conn:
            conn.close()

# --- Funciones de Inserción y Lectura de Datos ---

def insert_sensor_data_from_csv():
    """
    lee el archivo data/sensor_log.csv (generado por generate_sensors.py) 
    e inserta los datos en la tabla sensor_readings.
    """
    conn = None
    try:
        if not os.path.exists(CSV_FILE):
            print(f"Error: No se encontró el archivo CSV en '{CSV_FILE}'. Ejecute 'generate_sensors.py' primero.")
            return

        with open(CSV_FILE, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader) 
            
            # Aseguramos que el orden de las columnas del CSV coincida con la tabla
            # asumimos: timestamp, humedad_suelo, temperatura, luz, humedad_aire, ph
            data_to_insert = [row for row in reader]

        conn = get_db_connection()
        cursor = conn.cursor()

        sql_insert = '''
            INSERT INTO sensor_readings (timestamp, humedad_suelo, temperatura, luz, humedad_aire, ph)
            VALUES (?, ?, ?, ?, ?, ?)
        '''

        # ejecución de la inserción. sqlite3 convierte los strings del CSV
        cursor.executemany(sql_insert, data_to_insert)
        
        print(f"Insertados {len(data_to_insert)} registros de sensores en 'sensor_readings'.")

    except Exception as e:
        print(f"Error durante la inserción de datos: {e}")
    finally:
        if conn:
            conn.close()

def fetch_latest_readings(limit=5):
    """Consulta y muestra los registros más recientes de los sensores."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Consulta los últimos registros, ordenados por 'timestamp'
        cursor.execute(f'SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT {limit}')
        
        print(f"\n--- Últimos {limit} Registros de Sensores ---")
        # Imprime la cabecera de la tabla
        column_names = [description[0] for description in cursor.description]
        print(column_names)
        
        # Imprime las filas
        for row in cursor.fetchall():
            print(row)
        print("------------------------------------------")

    except sqlite3.Error as e:
        print(f"Error al leer datos de la DB: {e}")
    finally:
        if conn:
            conn.close()

# --- Punto de Entrada del Script ---

if __name__ == '__main__':
    # 1. Crear las tablas
    create_tables()
    
    # 2. Intentar insertar los datos del CSV
    # (Necesitas correr generate_sensors.py primero para que exista el archivo CSV)
    insert_sensor_data_from_csv()
    
    # 3. Leer los datos insertados para verificar
    fetch_latest_readings(limit=10)
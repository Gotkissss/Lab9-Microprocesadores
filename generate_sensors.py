import csv
import random
import os
import math
from datetime import datetime, timedelta

# ======================== CONFIGURACIÓN ========================
DATA_DIR = 'data'
CSV_FILE = os.path.join(DATA_DIR, 'sensor_log.csv')
NUM_RECORDS = 1000  # Cambia a 10000 o 100000 para más datos

# ======================== GENERACIÓN DE DATOS ========================

def generate_sensor_data(num_records):
    """
    Genera datos sintéticos de sensores para Smart Agriculture.
    Simula un ciclo diurno realista con variaciones naturales.
    
    Canales:
    - humedad_suelo: 0.15 a 0.70 (fracción, no porcentaje)
    - temperatura: 15 a 40°C
    - luz: 50 a 1000 lux
    - humedad_aire: 0.30 a 0.95 (fracción)
    - ph: 5.0 a 8.0
    """
    data = []
    start_time = datetime.now() - timedelta(hours=num_records//60)
    
    print(f"🌱 Generando {num_records} registros de sensores...")
    print(f"   Rango temporal: últimas {num_records//60} horas\n")
    
    for i in range(num_records):
        # Timestamp incremental (1 registro por minuto)
        timestamp = (start_time + timedelta(minutes=i)).isoformat()
        
        # Ciclo diurno (t = 0 a 1, representa 24 horas)
        t = (i % 1440) / 1440.0  # 1440 minutos = 24 horas
        
        # ---- HUMEDAD SUELO ----
        # Decrece durante el día (evaporación), se recupera en la noche
        base_humedad = 0.45 - 0.20 * t
        humedad_suelo = base_humedad + random.uniform(-0.05, 0.05)
        humedad_suelo = max(0.15, min(0.70, humedad_suelo))
        
        # ---- TEMPERATURA ----
        # Sigue ciclo sinusoidal (mínimo 6am, máximo 2pm)
        base_temp = 18.0 + 12.0 * math.sin(t * math.pi)
        temperatura = base_temp + random.uniform(-2.0, 2.0)
        temperatura = max(15.0, min(40.0, temperatura))
        
        # ---- LUZ ----
        # Ciclo solar (0 lux en la noche, máximo al mediodía)
        base_luz = 200.0 + 600.0 * max(0, math.sin(t * math.pi))
        luz = base_luz + random.uniform(-50, 50)
        luz = max(50.0, min(1000.0, luz))
        
        # ---- HUMEDAD AIRE ----
        # Inversamente proporcional a la temperatura
        base_hum_aire = 0.80 - 0.25 * (temperatura - 18.0) / 12.0
        humedad_aire = base_hum_aire + random.uniform(-0.05, 0.05)
        humedad_aire = max(0.30, min(0.95, humedad_aire))
        
        # ---- pH ----
        # Relativamente estable con pequeñas variaciones
        ph = 6.5 + random.uniform(-0.5, 0.5)
        ph = max(5.0, min(8.0, ph))
        
        # Agregar registro
        data.append([
            timestamp,
            round(humedad_suelo, 4),
            round(temperatura, 2),
            round(luz, 1),
            round(humedad_aire, 4),
            round(ph, 2)
        ])
        
        # Progreso cada 10%
        if (i + 1) % (num_records // 10) == 0:
            progress = ((i + 1) / num_records) * 100
            print(f"   Progreso: {progress:.0f}% ({i+1}/{num_records})")
    
    return data

def write_to_csv(data):
    """
    Escribe los datos en CSV con el formato esperado por database.py.
    Formato: timestamp, humedad_suelo, temperatura, luz, humedad_aire, ph
    """
    # Crear directorio 'data' si no existe
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 Creado directorio '{DATA_DIR}'\n")
    
    # Header debe coincidir EXACTAMENTE con database.py
    header = ['timestamp', 'humedad_suelo', 'temperatura', 'luz', 'humedad_aire', 'ph']
    
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    
    print(f"\n✅ Datos generados exitosamente!")
    print(f"   Archivo: {CSV_FILE}")
    print(f"   Registros: {len(data)}")
    print(f"   Rango temporal: {data[0][0]} a {data[-1][0]}")
    print(f"\n📊 Muestra de datos:")
    print(f"   Primer registro:")
    print(f"     - Humedad suelo: {data[0][1]*100:.1f}%")
    print(f"     - Temperatura: {data[0][2]}°C")
    print(f"     - Luz: {data[0][3]} lux")
    print(f"     - Humedad aire: {data[0][4]*100:.1f}%")
    print(f"     - pH: {data[0][5]}")
    print(f"\n   Último registro:")
    print(f"     - Humedad suelo: {data[-1][1]*100:.1f}%")
    print(f"     - Temperatura: {data[-1][2]}°C")
    print(f"     - Luz: {data[-1][3]} lux")
    print(f"     - Humedad aire: {data[-1][4]*100:.1f}%")
    print(f"     - pH: {data[-1][5]}")

def validate_data(data):
    """Valida que los datos generados estén en rangos correctos."""
    print("\n🔍 Validando datos generados...")
    
    errors = 0
    for i, row in enumerate(data):
        timestamp, hum_suelo, temp, luz, hum_aire, ph = row
        
        # Validar rangos
        if not (0.15 <= hum_suelo <= 0.70):
            print(f"⚠️  Registro {i}: Humedad suelo fuera de rango: {hum_suelo}")
            errors += 1
        if not (15.0 <= temp <= 40.0):
            print(f"⚠️  Registro {i}: Temperatura fuera de rango: {temp}")
            errors += 1
        if not (50.0 <= luz <= 1000.0):
            print(f"⚠️  Registro {i}: Luz fuera de rango: {luz}")
            errors += 1
        if not (0.30 <= hum_aire <= 0.95):
            print(f"⚠️  Registro {i}: Humedad aire fuera de rango: {hum_aire}")
            errors += 1
        if not (5.0 <= ph <= 8.0):
            print(f"⚠️  Registro {i}: pH fuera de rango: {ph}")
            errors += 1
    
    if errors == 0:
        print("✅ Todos los datos están en rangos válidos")
    else:
        print(f"⚠️  Se encontraron {errors} valores fuera de rango")
    
    return errors == 0

# ======================== MAIN ========================

if __name__ == '__main__':
    print("="*60)
    print("🌱 GENERADOR DE DATOS SINTÉTICOS - SMART AGRICULTURE")
    print("="*60)
    print()
    
    # Generar datos
    synthetic_data = generate_sensor_data(NUM_RECORDS)
    
    # Validar
    if validate_data(synthetic_data):
        # Escribir a CSV
        write_to_csv(synthetic_data)
        
        print("\n" + "="*60)
        print("✅ PROCESO COMPLETADO")
        print("="*60)
        print("\n📌 Siguiente paso: Ejecuta 'python database.py' para cargar")
        print("   los datos en la base de datos SQLite\n")
    else:
        print("\n❌ Error: Los datos generados tienen problemas")
        print("   Revisa los mensajes de error arriba\n")
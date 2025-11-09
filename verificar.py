import sqlite3

print("🔍 Verificando base de datos...\n")

# Conectar a la BD
conn = sqlite3.connect('data/agriculture.db')
cursor = conn.cursor()

# Contar registros
cursor.execute("SELECT COUNT(*) FROM sensor_readings")
total = cursor.fetchone()[0]
print(f"✅ Total registros en BD: {total}")

# Ver el primero
cursor.execute("SELECT * FROM sensor_readings ORDER BY id LIMIT 1")
primer = cursor.fetchone()
print(f"\n📍 Primer registro:")
print(f"   ID: {primer[0]}")
print(f"   Timestamp: {primer[1]}")
print(f"   Humedad suelo: {primer[2]*100:.1f}%")
print(f"   Temperatura: {primer[3]}°C")
print(f"   Luz: {primer[4]} lux")
print(f"   Humedad aire: {primer[5]*100:.1f}%")
print(f"   pH: {primer[6]}")

# Ver el último
cursor.execute("SELECT * FROM sensor_readings ORDER BY id DESC LIMIT 1")
ultimo = cursor.fetchone()
print(f"\n📍 Último registro:")
print(f"   ID: {ultimo[0]}")
print(f"   Timestamp: {ultimo[1]}")
print(f"   Humedad suelo: {ultimo[2]*100:.1f}%")
print(f"   Temperatura: {ultimo[3]}°C")
print(f"   Luz: {ultimo[4]} lux")
print(f"   Humedad aire: {ultimo[5]*100:.1f}%")
print(f"   pH: {ultimo[6]}")

# Estadísticas rápidas
cursor.execute("""
    SELECT 
        AVG(humedad_suelo) as avg_hum,
        AVG(temperatura) as avg_temp,
        AVG(luz) as avg_luz,
        MIN(temperatura) as min_temp,
        MAX(temperatura) as max_temp
    FROM sensor_readings
""")
stats = cursor.fetchone()
print(f"\n📊 Estadísticas generales:")
print(f"   Humedad promedio: {stats[0]*100:.1f}%")
print(f"   Temperatura promedio: {stats[1]:.1f}°C")
print(f"   Luz promedio: {stats[2]:.0f} lux")
print(f"   Temperatura mín/máx: {stats[3]:.1f}°C / {stats[4]:.1f}°C")

conn.close()

print("\n✅ Verificación completada!")
print("="*60)
print("🎉 LA BASE DE DATOS ESTÁ LISTA PARA USAR")
print("="*60)
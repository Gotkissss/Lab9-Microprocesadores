# Smart Agriculture Dashboard

## Descripción
Dashboard web para monitoreo en tiempo real del sistema Smart Agriculture.

## Características
- 5 canales de sensores monitoreados
- Actualización automática cada 5 segundos
- Sistema de alertas en 3 niveles
- Gráficas interactivas con Chart.js

## Requisitos
- Python 3.8+
- Flask 3.0.0
- flask-cors 4.0.0

## Instalación
```bash
pip install -r requirements.txt
```

## Uso
1. Ejecutar backend: `python api.py`
2. Abrir navegador: `http://localhost:5000`
3. Navegar al dashboard desde index.html

## Endpoints API
- `GET /api/sensors/latest` - Últimos 100 registros
- `GET /api/sensors/stats` - Estadísticas agregadas
- `GET /api/alerts` - Alertas activas

## Créditos
Desarrollado por: Diego (Persona 2)
Lab 9 - CC3086 Microprocesadores
Universidad del Valle de Guatemala
import pandas as pd
import sqlite3
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "autosense.db"


def create_sample_sensor_data():
    """Create sample sensor data for testing."""
    # Sample vehicle data
    vehicles = [
        {"id": 1, "vin": "2HGFC2F59JH000001", "make": "Honda", "model": "Civic", "year": 2018},
        {"id": 2, "vin": "1HGBH41JXMN109186", "make": "Honda", "model": "Civic", "year": 2017},
        {"id": 3, "vin": "3VWDX7AJ5DM123456", "make": "Volkswagen", "model": "Jetta", "year": 2013},
    ]
    
    # Sample sensor readings
    sensor_readings = []
    base_time = datetime.now()
    
    for vehicle in vehicles:
        # Generate 24 hours of sensor data
        for hour in range(24):
            timestamp = base_time - timedelta(hours=hour)
            
            # Engine temperature (normal range: 180-220°F)
            temp = 190 + (hour % 6) * 5  # Varying temperature
            
            # RPM (normal range: 600-3000 at idle/cruise)
            rpm = 800 + (hour % 4) * 200
            
            # Fuel level (0-100%)
            fuel = max(20, 100 - hour * 3)
            
            # Speed (0-70 mph)
            speed = (hour % 8) * 10
            
            # Oil pressure (20-60 psi)
            oil_pressure = 30 + (hour % 3) * 10
            
            # Battery voltage (12-14V)
            battery = 12.5 + (hour % 2) * 0.5
            
            # Add readings
            sensor_readings.extend([
                {"vehicle_id": vehicle["id"], "ts": timestamp, "sensor": "engine_temp", "value": temp},
                {"vehicle_id": vehicle["id"], "ts": timestamp, "sensor": "rpm", "value": rpm},
                {"vehicle_id": vehicle["id"], "ts": timestamp, "sensor": "fuel_level", "value": fuel},
                {"vehicle_id": vehicle["id"], "ts": timestamp, "sensor": "speed", "value": speed},
                {"vehicle_id": vehicle["id"], "ts": timestamp, "sensor": "oil_pressure", "value": oil_pressure},
                {"vehicle_id": vehicle["id"], "ts": timestamp, "sensor": "battery_voltage", "value": battery},
            ])
    
    return vehicles, sensor_readings


def load_kaggle_sensor_data(file_path: str) -> Optional[tuple]:
    """Load sensor data from Kaggle dataset."""
    try:
        if not os.path.exists(file_path):
            logger.warning(f"Kaggle dataset not found: {file_path}")
            return None
        
        # Read the dataset (assuming CSV or Parquet format)
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            logger.error(f"Unsupported file format: {file_path}")
            return None
        
        logger.info(f"Loaded Kaggle dataset with {len(df)} rows")
        
        # Transform to our schema
        vehicles = []
        sensor_readings = []
        
        # Extract unique vehicles
        if 'vin' in df.columns:
            unique_vins = df['vin'].unique()
            for i, vin in enumerate(unique_vins):
                vehicle_data = df[df['vin'] == vin].iloc[0]
                vehicles.append({
                    "id": i + 1,
                    "vin": str(vin),
                    "make": vehicle_data.get('make', 'Unknown'),
                    "model": vehicle_data.get('model', 'Unknown'),
                    "year": vehicle_data.get('year', 2020)
                })
        
        # Extract sensor readings
        for _, row in df.iterrows():
            # Map common sensor columns
            sensor_mappings = {
                'engine_temp': ['engine_temp', 'temperature', 'temp'],
                'rpm': ['rpm', 'engine_rpm'],
                'fuel_level': ['fuel_level', 'fuel', 'fuel_gauge'],
                'speed': ['speed', 'vehicle_speed', 'mph'],
                'oil_pressure': ['oil_pressure', 'oil'],
                'battery_voltage': ['battery_voltage', 'battery', 'voltage']
            }
            
            for sensor_name, possible_columns in sensor_mappings.items():
                for col in possible_columns:
                    if col in row and pd.notna(row[col]):
                        sensor_readings.append({
                            "vehicle_id": row.get('vehicle_id', 1),
                            "ts": pd.to_datetime(row.get('timestamp', datetime.now())),
                            "sensor": sensor_name,
                            "value": float(row[col])
                        })
                        break
        
        return vehicles, sensor_readings
        
    except Exception as e:
        logger.error(f"Error loading Kaggle dataset: {e}")
        return None


def save_sensor_data(vehicles: List[Dict[str, Any]], sensor_readings: List[Dict[str, Any]]):
    """Save sensor data to database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Save vehicles
        for vehicle in vehicles:
            cursor.execute(
                """INSERT OR REPLACE INTO vehicle (id, vin, make, model, year) 
                   VALUES (?, ?, ?, ?, ?)""",
                (vehicle["id"], vehicle["vin"], vehicle["make"], vehicle["model"], vehicle["year"])
            )
        
        # Save sensor readings
        for reading in sensor_readings:
            cursor.execute(
                """INSERT INTO sensor_reading (vehicle_id, ts, sensor, value) 
                   VALUES (?, ?, ?, ?)""",
                (reading["vehicle_id"], reading["ts"], reading["sensor"], reading["value"])
            )
        
        conn.commit()
        logger.info(f"Saved {len(vehicles)} vehicles and {len(sensor_readings)} sensor readings")
        
    except Exception as e:
        logger.error(f"Error saving sensor data: {e}")
        conn.rollback()
    finally:
        conn.close()


def get_sensor_analytics(vehicle_id: Optional[int] = None, sensor: Optional[str] = None):
    """Get sensor analytics and anomalies."""
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # Build query
        query = """
        SELECT 
            sr.vehicle_id,
            v.vin,
            sr.sensor,
            AVG(sr.value) as avg_value,
            MIN(sr.value) as min_value,
            MAX(sr.value) as max_value,
            COUNT(*) as reading_count
        FROM sensor_reading sr
        JOIN vehicle v ON sr.vehicle_id = v.id
        WHERE 1=1
        """
        params = []
        
        if vehicle_id:
            query += " AND sr.vehicle_id = ?"
            params.append(vehicle_id)
        
        if sensor:
            query += " AND sr.sensor = ?"
            params.append(sensor)
        
        query += " GROUP BY sr.vehicle_id, sr.sensor"
        
        df = pd.read_sql_query(query, conn, params=params)
        return df
        
    except Exception as e:
        logger.error(f"Error getting sensor analytics: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def detect_anomalies(vehicle_id: Optional[int] = None):
    """Detect sensor anomalies based on thresholds."""
    conn = sqlite3.connect(DB_PATH)
    
    # Define normal ranges for sensors
    thresholds = {
        'engine_temp': {'min': 160, 'max': 230, 'unit': '°F'},
        'rpm': {'min': 500, 'max': 3500, 'unit': 'RPM'},
        'fuel_level': {'min': 5, 'max': 100, 'unit': '%'},
        'speed': {'min': 0, 'max': 120, 'unit': 'mph'},
        'oil_pressure': {'min': 15, 'max': 70, 'unit': 'psi'},
        'battery_voltage': {'min': 11.5, 'max': 14.5, 'unit': 'V'}
    }
    
    anomalies = []
    
    try:
        for sensor, threshold in thresholds.items():
            query = """
            SELECT sr.*, v.vin
            FROM sensor_reading sr
            JOIN vehicle v ON sr.vehicle_id = v.id
            WHERE sr.sensor = ? AND (sr.value < ? OR sr.value > ?)
            """
            params = [sensor, threshold['min'], threshold['max']]
            
            if vehicle_id:
                query += " AND sr.vehicle_id = ?"
                params.append(vehicle_id)
            
            cursor = conn.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            for result in results:
                anomalies.append({
                    'vehicle_id': result[0],
                    'vin': result[4],
                    'sensor': result[2],
                    'value': result[3],
                    'timestamp': result[1],
                    'threshold_min': threshold['min'],
                    'threshold_max': threshold['max'],
                    'unit': threshold['unit']
                })
        
        return anomalies
        
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        return []
    finally:
        conn.close()


if __name__ == "__main__":
    # Try to load Kaggle dataset first
    kaggle_data = load_kaggle_sensor_data("data/automobile_telematics.csv")
    
    if kaggle_data:
        vehicles, sensor_readings = kaggle_data
        logger.info("Using Kaggle dataset")
    else:
        # Fall back to sample data
        vehicles, sensor_readings = create_sample_sensor_data()
        logger.info("Using sample sensor data")
    
    # Save to database
    save_sensor_data(vehicles, sensor_readings)
    
    # Show analytics
    print("\n=== Sensor Analytics ===")
    analytics = get_sensor_analytics()
    print(analytics)
    
    print("\n=== Detected Anomalies ===")
    anomalies = detect_anomalies()
    for anomaly in anomalies[:5]:  # Show first 5 anomalies
        print(f"Vehicle {anomaly['vin']}: {anomaly['sensor']} = {anomaly['value']} {anomaly['unit']} (normal: {anomaly['threshold_min']}-{anomaly['threshold_max']})") 
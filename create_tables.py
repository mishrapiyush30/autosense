import sqlite3

def create_tables():
    """Create missing tables in the database."""
    conn = sqlite3.connect("autosense.db")
    cursor = conn.cursor()
    
    # Create vehicle table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicle (
            id INTEGER PRIMARY KEY,
            vin TEXT UNIQUE,
            make TEXT,
            model TEXT,
            year INTEGER
        )
    ''')
    
    # Create sensor_reading table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_reading (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id INTEGER,
            ts TIMESTAMP,
            sensor TEXT,
            value REAL,
            FOREIGN KEY (vehicle_id) REFERENCES vehicle (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Tables created successfully!")

if __name__ == "__main__":
    create_tables() 
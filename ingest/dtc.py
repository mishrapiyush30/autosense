import csv
import psycopg
import os
import pathlib
from typing import Optional

DB = os.environ["DATABASE_URL"]
CSV = pathlib.Path("data/obd_codes.csv")


def load_dtc_codes(csv_path: Optional[pathlib.Path] = None) -> None:
    """Load OBD-II trouble codes from CSV file into Postgres."""
    if csv_path is None:
        csv_path = CSV
    
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        print("Please download the OBD-II codes CSV file to the data/ directory")
        return
    
    with psycopg.connect(DB) as conn, conn.cursor() as cur, open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # Skip header if present
        
        for row in reader:
            if len(row) >= 3:
                code, category, description = row[0], row[1], row[2]
                try:
                    cur.execute(
                        """INSERT INTO dtc (code, category, description) 
                           VALUES (%s, %s, %s) 
                           ON CONFLICT (code) DO UPDATE SET 
                           description = EXCLUDED.description,
                           category = EXCLUDED.category""",
                        (code, category, description)
                    )
                except Exception as e:
                    print(f"Error inserting DTC {code}: {e}")
        
        conn.commit()
        print(f"Loaded DTC codes from {csv_path}")


def create_sample_dtc_data() -> None:
    """Create sample DTC data for testing if CSV is not available."""
    sample_dtcs = [
        ("P0420", "Engine", "Catalyst System Efficiency Below Threshold (Bank 1)"),
        ("P0300", "Engine", "Random/Multiple Cylinder Misfire Detected"),
        ("P0171", "Engine", "System Too Lean (Bank 1)"),
        ("P0174", "Engine", "System Too Lean (Bank 2)"),
        ("P0128", "Engine", "Coolant Thermostat Temperature Below Regulating Temperature"),
        ("P0442", "Evaporative Emission Control", "Evaporative Emission Control System Leak Detected (Small Leak)"),
        ("P0455", "Evaporative Emission Control", "Evaporative Emission Control System Leak Detected (Gross Leak)"),
        ("P0506", "Engine", "Idle Control System RPM Lower Than Expected"),
        ("P0507", "Engine", "Idle Control System RPM Higher Than Expected"),
        ("P0700", "Transmission", "Transmission Control System Malfunction"),
    ]
    
    with psycopg.connect(DB) as conn, conn.cursor() as cur:
        for code, category, description in sample_dtcs:
            cur.execute(
                """INSERT INTO dtc (code, category, description) 
                   VALUES (%s, %s, %s) 
                   ON CONFLICT (code) DO UPDATE SET 
                   description = EXCLUDED.description,
                   category = EXCLUDED.category""",
                (code, category, description)
            )
        conn.commit()
        print(f"Created {len(sample_dtcs)} sample DTC codes")


if __name__ == "__main__":
    if CSV.exists():
        load_dtc_codes()
    else:
        print("CSV file not found, creating sample DTC data...")
        create_sample_dtc_data() 
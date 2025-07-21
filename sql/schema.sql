-- vehicles
CREATE TABLE vehicle (
    id SERIAL PRIMARY KEY,
    vin CHAR(17) UNIQUE NOT NULL,
    make TEXT,
    model TEXT,
    year INT
);

-- trouble codes
CREATE TABLE dtc (
    code VARCHAR(7) PRIMARY KEY,
    category TEXT,
    description TEXT
);

-- recalls
CREATE TABLE recall (
    id SERIAL PRIMARY KEY,
    nhtsa_id INT,
    vin CHAR(17),
    date DATE,
    summary TEXT
);

-- time‑series sensors (partition later if needed)
CREATE TABLE sensor_reading (
    id BIGSERIAL PRIMARY KEY,
    vehicle_id INT REFERENCES vehicle(id),
    ts TIMESTAMP,
    sensor TEXT,
    value DOUBLE PRECISION
);

-- Create indexes for better performance
CREATE INDEX idx_vehicle_vin ON vehicle(vin);
CREATE INDEX idx_recall_vin ON recall(vin);
CREATE INDEX idx_recall_date ON recall(date);
CREATE INDEX idx_sensor_vehicle_ts ON sensor_reading(vehicle_id, ts); 
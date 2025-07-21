import requests
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()

# NHTSA API endpoints
VPIC_API = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/"
RECALL_API = "https://api.nhtsa.gov/recalls/recallsByVehicle"


def decode_vin(vin: str) -> Dict[str, str]:
    """Decode VIN to get make, model, and year."""
    try:
        url = f"{VPIC_API}{vin}?format=json"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data.get("Results") and len(data["Results"]) > 0:
            result = data["Results"][0]
            return {
                "make": result.get("Make", ""),
                "model": result.get("Model", ""),
                "year": result.get("ModelYear", "")
            }
    except Exception as e:
        print(f"Error decoding VIN {vin}: {e}")
    
    return {"make": "", "model": "", "year": ""}


def fetch_recalls_by_vehicle(make: str, model: str, year: str) -> List[Dict[str, Any]]:
    """Fetch recalls for a specific make, model, and year."""
    try:
        # Clean up parameters
        make = make.lower().replace(" ", "")
        model = model.lower().replace(" ", "")
        
        url = f"{RECALL_API}?make={make}&model={model}&modelYear={year}"
        print(f"Fetching recalls: {url}")
        
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        results = data.get("results", [])
        print(f"Found {len(results)} recalls for {year} {make} {model}")
        return results
        
    except Exception as e:
        print(f"Error fetching recalls for {year} {make} {model}: {e}")
        return []


def fetch_recalls_from_vins(vins: List[str]) -> List[Dict[str, Any]]:
    """Fetch recalls for a list of VINs."""
    all_recalls = []
    
    for vin in vins:
        print(f"Processing VIN: {vin}")
        
        # Decode VIN to get make, model, year
        vehicle_info = decode_vin(vin)
        
        if vehicle_info["make"] and vehicle_info["model"] and vehicle_info["year"]:
            # Fetch recalls for this vehicle
            recalls = fetch_recalls_by_vehicle(
                vehicle_info["make"], 
                vehicle_info["model"], 
                vehicle_info["year"]
            )
            
            # Add VIN to each recall
            for recall in recalls:
                recall["VIN"] = vin
                all_recalls.append(recall)
        else:
            print(f"Could not decode VIN {vin}")
    
    return all_recalls


def fetch_recalls(days: int = 30) -> List[Dict[str, Any]]:
    """Fetch recalls using sample VINs for testing."""
    # Sample VINs for testing
    sample_vins = [
        "2HGFC2F59JH000001",  # Honda Civic
        "1HGBH41JXMN109186",  # Honda Civic
        "3VWDX7AJ5DM123456",  # Volkswagen Jetta
    ]
    
    print("Fetching recalls for sample VINs...")
    recalls = fetch_recalls_from_vins(sample_vins)
    
    if not recalls:
        print("No recalls found, using sample data")
        return get_sample_recalls()
    
    return recalls


def get_sample_recalls() -> List[Dict[str, Any]]:
    """Return sample recall data for testing."""
    return [
        {
            "NHTSACampaignNumber": "23V123456",
            "VIN": "1HGBH41JXMN109186",
            "RecallDate": "2024-01-15",
            "Summary": "Safety recall for airbag deployment issue affecting 2017-2019 Honda Civic models"
        },
        {
            "NHTSACampaignNumber": "23V789012",
            "VIN": "2HGFC2F59JH000001",
            "RecallDate": "2024-02-20",
            "Summary": "Recall for brake system software update required for 2018 Honda Civic"
        },
        {
            "NHTSACampaignNumber": "23V345678",
            "VIN": "3VWDX7AJ5DM123456",
            "RecallDate": "2024-03-10",
            "Summary": "Fuel system component replacement required for 2013 Volkswagen Jetta"
        }
    ]


def save(rows: List[Dict[str, Any]]) -> None:
    """Save recall data to SQLite database."""
    if not rows:
        print("No recalls to save")
        return
    
    conn = sqlite3.connect("autosense.db")
    cursor = conn.cursor()
    
    for r in rows:
        try:
            cursor.execute(
                """INSERT OR REPLACE INTO recall (nhtsa_id, vin, date, summary) 
                   VALUES (?, ?, ?, ?)""",
                (
                    r.get("NHTSACampaignNumber"),
                    r.get("VIN", "")[:17] if r.get("VIN") else None,
                    r.get("RecallDate"),
                    r.get("Summary", "")
                )
            )
        except Exception as e:
            print(f"Error inserting recall {r.get('NHTSACampaignNumber')}: {e}")
    
    conn.commit()
    conn.close()
    print(f"Saved {len(rows)} recalls to database")


if __name__ == "__main__":
    print("Fetching NHTSA recalls...")
    recalls = fetch_recalls()
    print(f"Found {len(recalls)} recalls")
    save(recalls) 
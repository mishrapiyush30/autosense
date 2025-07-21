#!/usr/bin/env python3
"""
AutoSense Setup Script

This script helps you set up the AutoSense project by:
1. Creating necessary directories
2. Setting up environment variables
3. Loading sample data
4. Building the vector index
5. Starting the services
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Optional


def run_command(command: str, description: str, check: bool = True) -> bool:
    """Run a shell command with error handling."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False


def check_prerequisites() -> bool:
    """Check if all prerequisites are installed."""
    print("🔍 Checking prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        return False
    
    # Check Docker
    if not run_command("docker --version", "Checking Docker", check=False):
        print("❌ Docker is not installed or not running")
        return False
    
    # Check Docker Compose
    if not run_command("docker compose version", "Checking Docker Compose", check=False):
        print("❌ Docker Compose is not installed")
        return False
    
    print("✅ All prerequisites are satisfied")
    return True


def create_env_file() -> bool:
    """Create .env file if it doesn't exist."""
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    print("📝 Creating .env file...")
    env_content = """# AutoSense Environment Configuration

# Database
DATABASE_URL=postgresql+psycopg://postgres:example@localhost:5432/postgres

# Vector Database
QDRANT_URL=http://localhost:6333

# OpenAI (optional - add your key for LLM features)
# OPENAI_API_KEY=your_openai_api_key_here

# API Configuration
API_BASE_URL=http://localhost:8000
"""
    
    try:
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ .env file created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False


def start_services() -> bool:
    """Start Docker services."""
    print("🚀 Starting Docker services...")
    
    # Start PostgreSQL and Qdrant
    if not run_command("docker compose up -d postgres qdrant", "Starting PostgreSQL and Qdrant"):
        return False
    
    # Wait for services to be ready
    print("⏳ Waiting for services to be ready...")
    time.sleep(10)
    
    return True


def install_dependencies() -> bool:
    """Install Python dependencies."""
    print("📦 Installing Python dependencies...")
    
    # Upgrade pip
    run_command("python -m pip install --upgrade pip", "Upgrading pip", check=False)
    
    # Install project dependencies
    if not run_command("pip install -e .", "Installing project dependencies"):
        return False
    
    return True


def initialize_database() -> bool:
    """Initialize the database schema."""
    print("🗄️ Initializing database...")
    
    # Set environment variables
    os.environ["DATABASE_URL"] = "postgresql+psycopg://postgres:example@localhost:5432/postgres"
    
    # Wait for PostgreSQL to be ready
    print("⏳ Waiting for PostgreSQL to be ready...")
    time.sleep(5)
    
    # Run schema initialization
    if not run_command("psql -h localhost -U postgres -d postgres -f sql/schema.sql", "Initializing database schema", check=False):
        print("⚠️ Database schema initialization failed, but continuing...")
    
    return True


def load_sample_data() -> bool:
    """Load sample data into the database."""
    print("📥 Loading sample data...")
    
    # Set environment variables
    os.environ["DATABASE_URL"] = "postgresql+psycopg://postgres:example@localhost:5432/postgres"
    
    # Load DTC codes
    if not run_command("python ingest/dtc.py", "Loading DTC codes"):
        print("⚠️ DTC codes loading failed, but continuing...")
    
    # Load recalls
    if not run_command("python ingest/recalls.py", "Loading recalls"):
        print("⚠️ Recalls loading failed, but continuing...")
    
    return True


def build_vector_index() -> bool:
    """Build the vector index."""
    print("🔍 Building vector index...")
    
    # Set environment variables
    os.environ["DATABASE_URL"] = "postgresql+psycopg://postgres:example@localhost:5432/postgres"
    os.environ["QDRANT_URL"] = "http://localhost:6333"
    
    # Wait for Qdrant to be ready
    print("⏳ Waiting for Qdrant to be ready...")
    time.sleep(5)
    
    if not run_command("python index/build_index.py", "Building vector index"):
        print("⚠️ Vector index building failed, but continuing...")
    
    return True


def run_tests() -> bool:
    """Run the test suite."""
    print("🧪 Running tests...")
    
    # Set environment variables
    os.environ["DATABASE_URL"] = "postgresql+psycopg://postgres:example@localhost:5432/postgres"
    os.environ["QDRANT_URL"] = "http://localhost:6333"
    
    # Start API server in background
    print("🚀 Starting API server for testing...")
    api_process = subprocess.Popen(
        "uvicorn api:app --host 0.0.0.0 --port 8000",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Wait for server to start
    time.sleep(10)
    
    # Run tests
    test_success = run_command("pytest tests/ -v", "Running tests")
    
    # Stop API server
    api_process.terminate()
    api_process.wait()
    
    return test_success


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 AutoSense setup completed successfully!")
    print("="*60)
    print("\n📋 Next steps:")
    print("1. Start the FastAPI server:")
    print("   uvicorn api:app --reload --host 0.0.0.0 --port 8000")
    print("\n2. Start the Streamlit UI (in another terminal):")
    print("   streamlit run ui/app.py")
    print("\n3. Open your browser to:")
    print("   - API docs: http://localhost:8000/docs")
    print("   - Streamlit UI: http://localhost:8501")
    print("\n4. Try some example queries:")
    print("   - 'P0420 catalyst efficiency below threshold'")
    print("   - 'Engine misfiring and rough idle'")
    print("   - '2HGFC2F59JH000001 recalls'")
    print("\n🔧 Useful commands:")
    print("   - Run tests: pytest tests/ -v")
    print("   - Run evaluation: python eval/run_eval.py")
    print("   - Stop services: docker compose down")
    print("\n📚 Documentation: README.md")
    print("="*60)


def main():
    """Main setup function."""
    print("🚗 AutoSense Setup Script")
    print("="*40)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Setup failed: Prerequisites not met")
        sys.exit(1)
    
    # Create .env file
    if not create_env_file():
        print("\n❌ Setup failed: Could not create .env file")
        sys.exit(1)
    
    # Start services
    if not start_services():
        print("\n❌ Setup failed: Could not start services")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed: Could not install dependencies")
        sys.exit(1)
    
    # Initialize database
    if not initialize_database():
        print("\n⚠️ Warning: Database initialization had issues")
    
    # Load sample data
    if not load_sample_data():
        print("\n⚠️ Warning: Sample data loading had issues")
    
    # Build vector index
    if not build_vector_index():
        print("\n⚠️ Warning: Vector index building had issues")
    
    # Run tests
    if not run_tests():
        print("\n⚠️ Warning: Tests had issues")
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main() 
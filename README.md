# 🚗 AutoSense - AI Diagnostic Platform for Connected Cars

AutoSense is an end-to-end AI diagnostic platform that helps drivers and fleet operators understand cryptic OBD-II fault codes, recall notices, and sensor anomalies. Built with modern AI/ML technologies, it provides intelligent automotive diagnostics through a ReAct-style agentic RAG system.

## 🌟 Features

- **🔍 Intelligent Search**: Vector-based semantic search for DTC codes, recalls, and diagnostic information
- **🤖 ReAct Agent**: AI-powered reasoning agent that provides comprehensive diagnostic analysis
- **📊 Real-time Data**: Integration with NHTSA recalls API and OBD-II trouble codes
- **🎯 Edge Case Handling**: Robust validation and error handling for malformed inputs
- **📈 Evaluation Suite**: Comprehensive testing with MRR@10, ROUGE-L, and robustness metrics
- **🚀 Modern UI**: Beautiful Streamlit interface with real-time diagnostics
- **🔧 CI/CD Pipeline**: Automated testing, security scanning, and deployment

## 🏗️ Architecture

```mermaid
flowchart TD
    User[👤 User] -->|HTTP| UI[🎨 Streamlit UI]
    UI -->|REST API| APP[⚡ FastAPI + ReAct Agent]
    APP -->|Vector Search| QDRANT[🗄️ Qdrant Vector DB]
    APP -->|SQL Queries| PG[(🐘 PostgreSQL)]
    
    ETL[📥 Data Ingestion] -->|NHTSA API| PG
    ETL -->|OBD-II Codes| PG
    ETL -->|Embeddings| QDRANT
    
    EVAL[📊 Evaluation] -->|Test Suite| APP
    CI[🔄 CI/CD] -->|Automated Testing| EVAL
    
    subgraph "Data Sources"
        NHTSA[NHTSA Recalls API]
        OBD[OBD-II Trouble Codes]
        SENSORS[Vehicle Telematics]
    end
    
    NHTSA --> ETL
    OBD --> ETL
    SENSORS --> ETL
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- OpenAI API key (optional, for LLM features)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd AutoSense
```

### 2. Environment Configuration

Create a `.env` file:

```bash
# Database
DATABASE_URL=postgresql+psycopg://postgres:example@localhost:5432/postgres

# Vector Database
QDRANT_URL=http://localhost:6333

# OpenAI (optional)
OPENAI_API_KEY=your_openai_api_key_here

# API Configuration
API_BASE_URL=http://localhost:8000
```

### 3. Start Services

```bash
# Start all services with Docker Compose
docker compose up -d

# Or start services individually
docker compose up -d postgres qdrant
```

### 4. Install Dependencies

```bash
pip install -e .
```

### 5. Initialize Data

```bash
# Load sample DTC codes
python ingest/dtc.py

# Fetch recent recalls from NHTSA
python ingest/recalls.py

# Build vector index
python index/build_index.py
```

### 6. Start the Application

```bash
# Start FastAPI server
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Start Streamlit UI (in another terminal)
streamlit run ui/app.py
```

## 📖 Usage Examples

### API Usage

```python
import httpx
import asyncio

async def search_dtc():
    async with httpx.AsyncClient() as client:
        # Search for DTC code
        response = await client.post("http://localhost:8000/search", json={
            "query": "P0420 catalyst efficiency",
            "k": 5
        })
        results = response.json()
        print(f"Found {len(results['results'])} results")

# Run the search
asyncio.run(search_dtc())
```

### Agent Usage

```python
from agent.core import AutoSenseAgent

async def diagnose_problem():
    agent = AutoSenseAgent()
    result = await agent.react("My car is showing P0420 error code", vin="2HGFC2F59JH000001")
    print(result["answer"])
    await agent.close()

# Run diagnosis
asyncio.run(diagnose_problem())
```

### Streamlit UI

1. Open your browser to `http://localhost:8501`
2. Enter a diagnostic query (e.g., "P0420 catalyst efficiency below threshold")
3. Optionally provide a VIN for vehicle-specific recalls
4. Click "🤖 AI Diagnosis" for comprehensive analysis

## 🧪 Testing

### Run Unit Tests

```bash
pytest tests/ -v
```

### Run Evaluation Suite

```bash
python eval/run_eval.py
```

### Run Full CI Pipeline Locally

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linting
ruff check .
ruff format --check .

# Run type checking
mypy api.py agent/ eval/ ingest/ index/ ui/

# Run tests
pytest tests/ -v
```

## 📊 Evaluation Metrics

The system is evaluated using:

- **MRR@10**: Mean Reciprocal Rank for retrieval quality
- **ROUGE-L**: Longest common subsequence for answer quality
- **Robustness**: Success rate against adversarial inputs

Example evaluation results:
```
Retrieval MRR@10: 0.847
Answer Quality ROUGE-L: 0.723
Robustness Success Rate: 0.900
Overall Score: 0.823
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `QDRANT_URL` | Qdrant vector database URL | `http://localhost:6333` |
| `OPENAI_API_KEY` | OpenAI API key for LLM features | Optional |
| `API_BASE_URL` | FastAPI server URL | `http://localhost:8000` |

### Docker Configuration

The `docker-compose.yml` file includes:

- **PostgreSQL 16**: Primary database for structured data
- **Qdrant v1.7.3**: Vector database for embeddings
- **FastAPI App**: Main application server

## 🏗️ Project Structure

```
AutoSense/
├── api.py                 # FastAPI application
├── docker-compose.yml     # Docker services configuration
├── Dockerfile            # Container definition
├── pyproject.toml        # Python dependencies
├── README.md             # This file
├── .gitignore           # Git ignore rules
├── .github/
│   └── workflows/
│       └── ci.yml       # CI/CD pipeline
├── agent/
│   ├── core.py          # ReAct agent implementation
│   └── errors.py        # Edge case handling
├── eval/
│   └── run_eval.py      # Evaluation harness
├── ingest/
│   ├── dtc.py           # OBD-II codes loader
│   └── recalls.py       # NHTSA recalls loader
├── index/
│   └── build_index.py   # Vector index builder
├── sql/
│   └── schema.sql       # Database schema
├── tests/
│   └── test_search.py   # Unit tests
└── ui/
    └── app.py           # Streamlit UI
```

## 🔒 Security Features

- Input validation and sanitization
- SQL injection prevention
- XSS protection
- Rate limiting (configurable)
- Secure API endpoints

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker compose up --build -d

# Or build individual image
docker build -t autosense .
docker run -p 8000:8000 autosense
```

### Cloud Deployment

The project includes CI/CD pipelines for:

- **GitHub Actions**: Automated testing and deployment
- **Docker Registry**: Container image management
- **Environment Management**: Staging and production deployments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run tests before committing
pytest tests/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NHTSA**: For providing the recalls API
- **OBD-II Community**: For maintaining trouble code databases
- **OpenAI**: For GPT models powering the ReAct agent
- **Qdrant**: For the vector database technology
- **FastAPI**: For the modern web framework
- **Streamlit**: For the beautiful UI framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation**: [Wiki](https://github.com/your-repo/wiki)

---

**AutoSense** - Making automotive diagnostics intelligent and accessible! 🚗✨ 
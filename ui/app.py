import streamlit as st
import httpx
import asyncio
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
import time

# Page configuration
st.set_page_config(
    page_title="AutoSense - AI Diagnostic Platform",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .result-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .error-card {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .success-card {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")


class AutoSenseUI:
    """Streamlit UI for AutoSense diagnostic platform."""
    
    def __init__(self):
        self.api_base_url = API_BASE_URL
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def check_health(self) -> Dict[str, Any]:
        """Check API health status."""
        try:
            response = await self.http_client.get(f"{self.api_base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def search(self, query: str, vin: Optional[str] = None, k: int = 5) -> Dict[str, Any]:
        """Search for diagnostic information."""
        try:
            payload = {"query": query, "k": k}
            if vin:
                payload["vin"] = vin
            
            response = await self.http_client.post(f"{self.api_base_url}/search", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def get_dtc_info(self, code: str) -> Optional[Dict[str, Any]]:
        """Get detailed DTC information."""
        try:
            response = await self.http_client.get(f"{self.api_base_url}/dtc/{code}")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    async def get_recalls(self, vin: str) -> List[Dict[str, Any]]:
        """Get recalls for a VIN."""
        try:
            response = await self.http_client.get(f"{self.api_base_url}/recalls/{vin}")
            if response.status_code == 200:
                data = response.json()
                return data.get("recalls", [])
            return []
        except Exception:
            return []
    
    async def run_agent_diagnosis(self, query: str, vin: Optional[str] = None) -> Dict[str, Any]:
        """Run the ReAct agent for comprehensive diagnosis."""
        try:
            # Import the agent here to avoid circular imports
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            from agent.core import AutoSenseAgent
            from agent.errors import ensure_valid_input
            
            # Validate input
            validation = ensure_valid_input(query, vin)
            if not validation.get("is_valid", True):
                return {
                    "error": validation.get("error", "Validation failed"),
                    "suggestion": validation.get("suggestion", "Please check your input")
                }
            
            # Run agent
            agent = AutoSenseAgent(self.api_base_url)
            result = await agent.react(query, vin)
            await agent.close()
            
            return result
            
        except Exception as e:
            return {"error": f"Agent diagnosis failed: {str(e)}"}
    
    def render_header(self):
        """Render the main header."""
        st.markdown('<h1 class="main-header">🚗 AutoSense</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI Diagnostic Platform for Connected Cars</p>', unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with configuration options."""
        st.sidebar.title("⚙️ Configuration")
        
        # API Status
        st.sidebar.subheader("API Status")
        if st.button("Check Health"):
            health_status = asyncio.run(self.check_health())
            if health_status.get("status") == "healthy":
                st.sidebar.success("✅ API Healthy")
            else:
                st.sidebar.error("❌ API Unhealthy")
        
        # Search Configuration
        st.sidebar.subheader("Search Settings")
        k_results = st.sidebar.slider("Number of results", 1, 20, 5)
        
        # Advanced Options
        st.sidebar.subheader("Advanced Options")
        show_debug = st.sidebar.checkbox("Show Debug Info", False)
        
        return {
            "k_results": k_results,
            "show_debug": show_debug
        }
    
    def render_main_interface(self, config: Dict[str, Any]):
        """Render the main diagnostic interface."""
        st.subheader("🔍 Diagnostic Query")
        
        # Input form
        with st.form("diagnostic_form"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                query = st.text_area(
                    "Describe the problem or enter a DTC code",
                    placeholder="e.g., 'My car is showing P0420 error code' or 'Engine misfiring'",
                    height=100
                )
            
            with col2:
                vin = st.text_input(
                    "VIN (optional)",
                    placeholder="17-character VIN",
                    max_chars=17
                )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                search_button = st.form_submit_button("🔍 Search", use_container_width=True)
            with col2:
                diagnose_button = st.form_submit_button("🤖 AI Diagnosis", use_container_width=True)
            with col3:
                clear_button = st.form_submit_button("🗑️ Clear", use_container_width=True)
        
        # Handle form submission
        if search_button and query:
            self.handle_search(query, vin, config)
        elif diagnose_button and query:
            self.handle_diagnosis(query, vin, config)
        elif clear_button:
            st.rerun()
    
    def handle_search(self, query: str, vin: Optional[str], config: Dict[str, Any]):
        """Handle search request."""
        with st.spinner("Searching for diagnostic information..."):
            results = asyncio.run(self.search(query, vin, config["k_results"]))
        
        if "error" in results:
            st.error(f"Search failed: {results['error']}")
            return
        
        # Display results
        st.subheader(f"📋 Search Results ({len(results['results'])} found)")
        
        for i, result in enumerate(results['results'], 1):
            with st.expander(f"Result {i} (Score: {result['score']:.3f})"):
                if result.get("type") == "dtc":
                    st.markdown(f"**DTC Code:** {result.get('code', 'N/A')}")
                    st.markdown(f"**Category:** {result.get('category', 'N/A')}")
                    st.markdown(f"**Description:** {result.get('description', 'N/A')}")
                elif result.get("type") == "recall":
                    st.markdown(f"**Recall ID:** {result.get('rid', 'N/A')}")
                    st.markdown(f"**Date:** {result.get('date', 'N/A')}")
                    st.markdown(f"**Summary:** {result.get('summary', 'N/A')}")
        
        if config["show_debug"]:
            st.json(results)
    
    def handle_diagnosis(self, query: str, vin: Optional[str], config: Dict[str, Any]):
        """Handle AI diagnosis request."""
        with st.spinner("Running AI diagnosis..."):
            result = asyncio.run(self.run_agent_diagnosis(query, vin))
        
        if "error" in result:
            st.error(f"Diagnosis failed: {result['error']}")
            return
        
        # Display diagnosis
        st.subheader("🤖 AI Diagnosis")
        
        # Main answer
        st.markdown("### Analysis")
        st.markdown(result.get("answer", "No analysis available"))
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")
        with col2:
            st.metric("Search Results", len(result.get("search_results", [])))
        with col3:
            st.metric("Recalls Found", len(result.get("recalls", [])))
        
        # Detailed information
        if result.get("search_results"):
            st.subheader("📋 Supporting Information")
            for i, search_result in enumerate(result["search_results"][:3], 1):
                with st.expander(f"Source {i}"):
                    if search_result.get("type") == "dtc":
                        st.markdown(f"**DTC:** {search_result.get('code')}")
                        st.markdown(f"**Description:** {search_result.get('description')}")
                    else:
                        st.markdown(f"**Recall:** {search_result.get('rid')}")
                        st.markdown(f"**Summary:** {search_result.get('summary')}")
        
        if result.get("recalls"):
            st.subheader("⚠️ Active Recalls")
            for recall in result["recalls"][:3]:
                st.warning(f"**Recall {recall.get('nhtsa_id')}:** {recall.get('summary')}")
        
        if config["show_debug"]:
            st.subheader("🔧 Debug Information")
            st.json(result)
    
    def render_examples(self):
        """Render example queries."""
        st.subheader("💡 Example Queries")
        
        examples = [
            "P0420 catalyst efficiency below threshold",
            "Engine misfiring and rough idle",
            "Check engine light is on",
            "2HGFC2F59JH000001 recalls",
            "P0300 random misfire detected",
            "System too lean bank 1"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, key=f"example_{i}"):
                    st.session_state.example_query = example
                    st.rerun()
    
    def render_footer(self):
        """Render the footer."""
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; color: #666; font-size: 0.9rem;">
                <p>AutoSense - AI Diagnostic Platform for Connected Cars</p>
                <p>Built with FastAPI, Qdrant, and Streamlit</p>
            </div>
            """,
            unsafe_allow_html=True
        )


def main():
    """Main Streamlit application."""
    ui = AutoSenseUI()
    
    # Render header
    ui.render_header()
    
    # Render sidebar
    config = ui.render_sidebar()
    
    # Check for example query
    if "example_query" in st.session_state:
        st.text_area("Query", value=st.session_state.example_query, key="query_input")
        del st.session_state.example_query
    
    # Render main interface
    ui.render_main_interface(config)
    
    # Render examples
    ui.render_examples()
    
    # Render footer
    ui.render_footer()


if __name__ == "__main__":
    main() 
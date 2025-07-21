from typing import List, Dict, Any, Optional
import httpx
import openai
import os
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ReAct prompt template
REACT_PROMPT = """You are AutoSense, an expert automotive diagnostic agent. Your job is to help diagnose car problems using a systematic approach.

Follow this ReAct pattern:
1. **Thought**: Analyze the problem and determine what information you need
2. **Action**: Take a specific action (search, lookup, etc.)
3. **Observation**: Record what you found
4. **Thought**: Reflect on the information and decide next steps
5. **Answer**: Provide a comprehensive diagnosis with repair steps

Available actions:
- search: Search for relevant diagnostic information
- lookup_dtc: Get detailed information about a specific DTC code
- lookup_recalls: Check for recalls related to a VIN

Current query: {query}
VIN: {vin}

Start your reasoning process:"""


class AutoSenseAgent:
    """ReAct-style agent for automotive diagnostics."""
    
    def __init__(self, api_base_url: str = API_BASE_URL, openai_api_key: Optional[str] = None):
        self.api_base_url = api_base_url
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        
        if not self.openai_api_key:
            logger.warning("No OpenAI API key provided. Agent will use fallback responses.")
        
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def search(self, query: str, vin: Optional[str] = None, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant diagnostic information."""
        try:
            payload = {"query": query, "k": k}
            if vin:
                payload["vin"] = vin
            
            response = await self.http_client.post(f"{self.api_base_url}/search", json=payload)
            response.raise_for_status()
            return response.json()["results"]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def lookup_dtc(self, code: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a DTC code."""
        try:
            response = await self.http_client.get(f"{self.api_base_url}/dtc/{code}")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"DTC lookup failed for {code}: {e}")
            return None
    
    async def lookup_recalls(self, vin: str) -> List[Dict[str, Any]]:
        """Get recalls for a specific VIN."""
        try:
            response = await self.http_client.get(f"{self.api_base_url}/recalls/{vin}")
            if response.status_code == 200:
                data = response.json()
                return data.get("recalls", [])
            return []
        except Exception as e:
            logger.error(f"Recall lookup failed for VIN {vin}: {e}")
            return []
    
    def _extract_dtc_code(self, query: str) -> Optional[str]:
        """Extract DTC code from query if present."""
        import re
        # Look for patterns like P0420, C1234, etc.
        dtc_pattern = r'\b[PBCU][0-9]{4}\b'
        match = re.search(dtc_pattern, query.upper())
        return match.group() if match else None
    
    def _extract_vin(self, query: str) -> Optional[str]:
        """Extract VIN from query if present."""
        import re
        # Look for 17-character VIN pattern
        vin_pattern = r'\b[A-HJ-NPR-Z0-9]{17}\b'
        match = re.search(vin_pattern, query.upper())
        return match.group() if match else None
    
    async def react(self, query: str, vin: Optional[str] = None) -> Dict[str, Any]:
        """Execute the ReAct reasoning loop."""
        start_time = datetime.now()
        thoughts = []
        actions = []
        observations = []
        
        try:
            # Extract VIN and DTC from query if not provided
            extracted_vin = vin or self._extract_vin(query)
            extracted_dtc = self._extract_dtc_code(query)
            
            # Initial thought
            thoughts.append("Analyzing the automotive diagnostic query to understand the problem and determine required information.")
            
            # Search for relevant information
            actions.append("search")
            search_results = await self.search(query, extracted_vin, k=5)
            observations.append(f"Found {len(search_results)} relevant results")
            
            # If DTC code found, get detailed information
            if extracted_dtc:
                actions.append("lookup_dtc")
                dtc_info = await self.lookup_dtc(extracted_dtc)
                if dtc_info:
                    observations.append(f"Retrieved detailed DTC information for {extracted_dtc}")
                else:
                    observations.append(f"No detailed information found for DTC {extracted_dtc}")
            
            # If VIN provided, check for recalls
            if extracted_vin:
                actions.append("lookup_recalls")
                recalls = await self.lookup_recalls(extracted_vin)
                observations.append(f"Found {len(recalls)} recalls for VIN {extracted_vin}")
            
            # Generate final answer using LLM
            thoughts.append("Synthesizing all gathered information to provide a comprehensive diagnosis.")
            
            if self.openai_api_key:
                answer = await self._generate_llm_response(query, search_results, extracted_dtc, extracted_vin, recalls)
            else:
                answer = self._generate_fallback_response(query, search_results, extracted_dtc, extracted_vin, recalls)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            return {
                "query": query,
                "vin": extracted_vin,
                "dtc_code": extracted_dtc,
                "answer": answer,
                "thoughts": thoughts,
                "actions": actions,
                "observations": observations,
                "search_results": search_results,
                "recalls": recalls if extracted_vin else [],
                "processing_time": processing_time,
                "timestamp": start_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"ReAct execution failed: {e}")
            return {
                "query": query,
                "error": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": start_time.isoformat()
            }
    
    async def _generate_llm_response(
        self, 
        query: str, 
        search_results: List[Dict[str, Any]], 
        dtc_code: Optional[str], 
        vin: Optional[str], 
        recalls: List[Dict[str, Any]]
    ) -> str:
        """Generate response using OpenAI API."""
        try:
            # Prepare context for LLM
            context_parts = []
            
            if search_results:
                context_parts.append("Relevant diagnostic information:")
                for i, result in enumerate(search_results[:3], 1):
                    context_parts.append(f"{i}. {result.get('description', result.get('summary', 'No description'))}")
            
            if dtc_code:
                context_parts.append(f"\nDTC Code {dtc_code} detected in query.")
            
            if recalls:
                context_parts.append(f"\nRecalls found for VIN {vin}:")
                for recall in recalls[:3]:
                    context_parts.append(f"- {recall.get('summary', 'No summary')}")
            
            context = "\n".join(context_parts)
            
            messages = [
                {
                    "role": "system",
                    "content": """You are AutoSense, an expert automotive diagnostic assistant. 
                    Provide clear, actionable diagnostic advice based on the information provided. 
                    Include specific repair steps when possible, and always mention safety considerations."""
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nContext:\n{context}\n\nProvide a comprehensive diagnostic response:"
                }
            ]
            
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            return self._generate_fallback_response(query, search_results, dtc_code, vin, recalls)
    
    def _generate_fallback_response(
        self, 
        query: str, 
        search_results: List[Dict[str, Any]], 
        dtc_code: Optional[str], 
        vin: Optional[str], 
        recalls: List[Dict[str, Any]]
    ) -> str:
        """Generate fallback response without LLM."""
        response_parts = [f"Diagnostic analysis for: {query}"]
        
        if dtc_code:
            response_parts.append(f"\n**DTC Code Detected**: {dtc_code}")
        
        if search_results:
            response_parts.append("\n**Relevant Information Found**:")
            for result in search_results[:3]:
                if result.get("type") == "dtc":
                    response_parts.append(f"- DTC {result.get('code')}: {result.get('description', 'No description')}")
                elif result.get("type") == "recall":
                    response_parts.append(f"- Recall {result.get('rid')}: {result.get('summary', 'No summary')}")
        
        if recalls:
            response_parts.append(f"\n**Recalls for VIN {vin}**:")
            for recall in recalls[:2]:
                response_parts.append(f"- {recall.get('summary', 'No summary')}")
        
        response_parts.append("\n**Recommendation**: Please consult with a qualified automotive technician for proper diagnosis and repair.")
        
        return "\n".join(response_parts)
    
    async def close(self):
        """Close the HTTP client."""
        await self.http_client.aclose()


# Convenience function for easy usage
async def react(query: str, vin: Optional[str] = None) -> str:
    """Simple interface for ReAct agent."""
    agent = AutoSenseAgent()
    try:
        result = await agent.react(query, vin)
        return result.get("answer", "Unable to generate response")
    finally:
        await agent.close()


if __name__ == "__main__":
    import asyncio
    
    async def test_agent():
        agent = AutoSenseAgent()
        result = await agent.react("My car is showing P0420 error code")
        print(json.dumps(result, indent=2))
        await agent.close()
    
    asyncio.run(test_agent()) 
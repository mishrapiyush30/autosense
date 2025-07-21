import json
import httpx
import pandas as pd
import tqdm
import numpy as np
import evaluate
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize metrics
mrr = evaluate.load("mean_reciprocal_rank")
rouge = evaluate.load("rouge")


class AutoSenseEvaluator:
    """Evaluation harness for AutoSense agentic RAG system."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def evaluate_retrieval(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate retrieval performance using MRR@10."""
        scores = []
        
        for row in tqdm.tqdm(test_data, desc="Evaluating retrieval"):
            try:
                response = await self.http_client.post(
                    f"{self.api_base_url}/search",
                    json={"query": row["query"], "k": 10}
                )
                response.raise_for_status()
                hits = response.json()["results"]
                
                # Calculate relevance scores
                gold_codes = set(row.get("gold_codes", []))
                gold_recalls = set(row.get("gold_recalls", []))
                
                hit_scores = []
                for hit in hits:
                    score = 0
                    if hit.get("type") == "dtc" and hit.get("code") in gold_codes:
                        score = 1
                    elif hit.get("type") == "recall" and hit.get("rid") in gold_recalls:
                        score = 1
                    hit_scores.append(score)
                
                # Calculate MRR
                if any(hit_scores):
                    mrr_score = max(1 / (i + 1) for i, score in enumerate(hit_scores) if score)
                else:
                    mrr_score = 0
                
                scores.append(mrr_score)
                
            except Exception as e:
                logger.error(f"Error evaluating retrieval for query '{row['query']}': {e}")
                scores.append(0)
        
        return {
            "mrr@10": np.mean(scores),
            "std_mrr": np.std(scores),
            "total_queries": len(scores)
        }
    
    async def evaluate_answer_quality(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate answer quality using ROUGE-L."""
        predictions = []
        references = []
        
        for row in tqdm.tqdm(test_data, desc="Evaluating answer quality"):
            try:
                # Get search results
                response = await self.http_client.post(
                    f"{self.api_base_url}/search",
                    json={"query": row["query"], "k": 3}
                )
                response.raise_for_status()
                hits = response.json()["results"]
                
                # Create prediction from top result
                if hits:
                    top_hit = hits[0]
                    if top_hit.get("type") == "dtc":
                        prediction = top_hit.get("description", "")
                    else:
                        prediction = top_hit.get("summary", "")
                else:
                    prediction = "No relevant information found"
                
                predictions.append(prediction)
                references.append(row.get("reference", ""))
                
            except Exception as e:
                logger.error(f"Error evaluating answer quality for query '{row['query']}': {e}")
                predictions.append("Error occurred")
                references.append(row.get("reference", ""))
        
        # Calculate ROUGE-L
        rouge_scores = rouge.compute(predictions=predictions, references=references)
        
        return {
            "rouge_l": rouge_scores["rougeL"],
            "rouge_1": rouge_scores["rouge1"],
            "rouge_2": rouge_scores["rouge2"],
            "total_queries": len(predictions)
        }
    
    async def evaluate_agent_robustness(self, adversarial_queries: List[str]) -> Dict[str, Any]:
        """Evaluate agent robustness against adversarial inputs."""
        results = []
        
        for query in tqdm.tqdm(adversarial_queries, desc="Testing robustness"):
            try:
                response = await self.http_client.post(
                    f"{self.api_base_url}/search",
                    json={"query": query, "k": 5}
                )
                
                result = {
                    "query": query,
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "error": None
                }
                
                if response.status_code == 200:
                    data = response.json()
                    result["results_count"] = len(data.get("results", []))
                else:
                    result["error"] = response.text
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    "query": query,
                    "status_code": None,
                    "success": False,
                    "error": str(e),
                    "results_count": 0
                })
        
        success_rate = sum(1 for r in results if r["success"]) / len(results)
        
        return {
            "success_rate": success_rate,
            "total_queries": len(results),
            "failed_queries": sum(1 for r in results if not r["success"]),
            "detailed_results": results
        }
    
    async def run_full_evaluation(self, test_data_path: Optional[str] = None) -> Dict[str, Any]:
        """Run complete evaluation suite."""
        logger.info("Starting full evaluation suite...")
        
        # Load test data
        if test_data_path and Path(test_data_path).exists():
            with open(test_data_path, 'r') as f:
                test_data = json.load(f)
        else:
            test_data = self._create_sample_test_data()
        
        # Create adversarial queries
        adversarial_queries = self._create_adversarial_queries()
        
        # Run evaluations
        retrieval_results = await self.evaluate_retrieval(test_data)
        quality_results = await self.evaluate_answer_quality(test_data)
        robustness_results = await self.evaluate_agent_robustness(adversarial_queries)
        
        # Compile results
        full_results = {
            "retrieval": retrieval_results,
            "answer_quality": quality_results,
            "robustness": robustness_results,
            "summary": {
                "overall_score": (retrieval_results["mrr@10"] + quality_results["rouge_l"] + robustness_results["success_rate"]) / 3,
                "timestamp": pd.Timestamp.now().isoformat()
            }
        }
        
        logger.info("Evaluation completed!")
        logger.info(f"MRR@10: {retrieval_results['mrr@10']:.3f}")
        logger.info(f"ROUGE-L: {quality_results['rouge_l']:.3f}")
        logger.info(f"Robustness: {robustness_results['success_rate']:.3f}")
        
        return full_results
    
    def _create_sample_test_data(self) -> List[Dict[str, Any]]:
        """Create sample test data for evaluation."""
        return [
            {
                "query": "P0420 catalyst efficiency below threshold",
                "gold_codes": ["P0420"],
                "gold_recalls": [],
                "reference": "P0420 indicates catalyst system efficiency below threshold. This typically means the catalytic converter is not working properly and may need replacement."
            },
            {
                "query": "P0300 random misfire detected",
                "gold_codes": ["P0300"],
                "gold_recalls": [],
                "reference": "P0300 indicates random/multiple cylinder misfire detected. This can be caused by various issues including spark plugs, ignition coils, or fuel system problems."
            },
            {
                "query": "P0171 system too lean bank 1",
                "gold_codes": ["P0171"],
                "gold_recalls": [],
                "reference": "P0171 indicates system too lean on bank 1. This means the engine is running with too much air and not enough fuel."
            },
            {
                "query": "2HGFC2F59JH000001 recalls",
                "gold_codes": [],
                "gold_recalls": [],
                "reference": "Checking for recalls associated with VIN 2HGFC2F59JH000001."
            }
        ]
    
    def _create_adversarial_queries(self) -> List[str]:
        """Create adversarial queries for robustness testing."""
        return [
            "",  # Empty query
            "A" * 600,  # Very long query
            "P9999",  # Invalid DTC code
            "123456789012345678",  # Invalid VIN (18 chars)
            "P0420<script>alert('xss')</script>",  # XSS attempt
            "P0420' OR 1=1--",  # SQL injection attempt
            "🚗🚙🚕",  # Unicode emojis
            "P0420" + "\x00" * 10,  # Null bytes
            "P0420" + " " * 100,  # Excessive whitespace
            "P0420\nP0300\nP0171",  # Multiple lines
        ]
    
    async def close(self):
        """Close the HTTP client."""
        await self.http_client.aclose()


async def main():
    """Main evaluation function."""
    evaluator = AutoSenseEvaluator()
    
    try:
        # Run evaluation
        results = await evaluator.run_full_evaluation()
        
        # Save results
        output_path = "eval/results.json"
        Path(output_path).parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nEvaluation results saved to {output_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Retrieval MRR@10: {results['retrieval']['mrr@10']:.3f}")
        print(f"Answer Quality ROUGE-L: {results['answer_quality']['rouge_l']:.3f}")
        print(f"Robustness Success Rate: {results['robustness']['success_rate']:.3f}")
        print(f"Overall Score: {results['summary']['overall_score']:.3f}")
        print("="*50)
        
    finally:
        await evaluator.close()


if __name__ == "__main__":
    asyncio.run(main()) 
"""
BIS Standards RAG Engine - Evaluation Script
This script evaluates the RAG engine performance
"""

import json
import time
import argparse
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ground_truth(gt_path: str) -> Dict:
    """Load ground truth data"""
    with open(gt_path, 'r') as f:
        return json.load(f)


def calculate_hit_rate(predictions: List[Dict], ground_truth: Dict, k: int = 3) -> float:
    """Calculate Hit Rate @ k"""
    hits = 0
    total = 0
    
    for pred in predictions:
        query_id = pred["id"]
        
        if query_id not in ground_truth:
            continue
            
        gt_standards = set(ground_truth[query_id])
        pred_standards = [
            s["standard"] for s in pred["retrieved_standards"][:k]
        ]
        
        total += 1
        if any(s in gt_standards for s in pred_standards):
            hits += 1
    
    return hits / total if total > 0 else 0.0


def calculate_mrr(predictions: List[Dict], ground_truth: Dict, k: int = 5) -> float:
    """Calculate Mean Reciprocal Rank @ k"""
    rr_sum = 0.0
    total = 0
    
    for pred in predictions:
        query_id = pred["id"]
        
        if query_id not in ground_truth:
            continue
            
        gt_standards = set(ground_truth[query_id])
        pred_standards = [
            s["standard"] for s in pred["retrieved_standards"][:k]
        ]
        
        total += 1
        
        # Find first relevant standard
        for i, pred_std in enumerate(pred_standards):
            if pred_std in gt_standards:
                rr_sum += 1.0 / (i + 1)
                break
    
    return rr_sum / total if total > 0 else 0.0


def calculate_latency(predictions: List[Dict]) -> Dict:
    """Calculate average latency"""
    latencies = [p["latency_seconds"] for p in predictions]
    
    return {
        "avg": sum(latencies) / len(latencies) if latencies else 0,
        "min": min(latencies) if latencies else 0,
        "max": max(latencies) if latencies else 0
    }


def evaluate(predictions_path: str, ground_truth_path: str = None) -> Dict:
    """Run evaluation"""
    
    # Load predictions
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    
    logger.info(f"Loaded {len(predictions)} predictions")
    
    results = {
        "num_predictions": len(predictions),
        "latency": calculate_latency(predictions)
    }
    
    # If ground truth provided, calculate retrieval metrics
    if ground_truth_path:
        ground_truth = load_ground_truth(ground_truth_path)
        
        results["hit_rate@3"] = calculate_hit_rate(predictions, ground_truth, k=3)
        results["hit_rate@5"] = calculate_hit_rate(predictions, ground_truth, k=5)
        results["mrr@5"] = calculate_mrr(predictions, ground_truth, k=5)
        
        logger.info(f"Hit Rate @3: {results['hit_rate@3']:.4f}")
        logger.info(f"Hit Rate @5: {results['hit_rate@5']:.4f}")
        logger.info(f"MRR @5: {results['mrr@5']:.4f}")
    
    logger.info(f"Average latency: {results['latency']['avg']:.4f}s")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG Engine")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON")
    parser.add_argument("--ground-truth", help="Path to ground truth JSON")
    
    args = parser.parse_args()
    
    results = evaluate(args.predictions, args.ground_truth)
    
    print("\n=== Evaluation Results ===")
    print(json.dumps(results, indent=2))
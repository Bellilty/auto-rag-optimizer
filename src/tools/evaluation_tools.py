"""
Evaluation Tools
================

Tools for evaluating RAG performance and comparing configurations.
"""

import numpy as np
from typing import List, Dict, Any, Optional


def calculate_average_score(evaluations: List[Dict]) -> float:
    """
    Calculate average evaluation score.
    
    Args:
        evaluations: List of evaluation dictionaries with 'score' field
        
    Returns:
        Average score
    """
    if not evaluations:
        return 0.0
    
    scores = [eval_dict.get('score', 0) for eval_dict in evaluations]
    return float(np.mean(scores))


def calculate_win_rate(
    baseline_scores: List[float],
    optimized_scores: List[float]
) -> Dict[str, Any]:
    """
    Calculate win rate of optimized vs baseline.
    
    Args:
        baseline_scores: Scores from baseline configuration
        optimized_scores: Scores from optimized configuration
        
    Returns:
        Dictionary with win rate statistics
    """
    if len(baseline_scores) != len(optimized_scores):
        raise ValueError("Score lists must have same length")
    
    if not baseline_scores:
        return {
            "win_rate": 0,
            "loss_rate": 0,
            "tie_rate": 0
        }
    
    wins = sum(1 for b, o in zip(baseline_scores, optimized_scores) if o > b)
    losses = sum(1 for b, o in zip(baseline_scores, optimized_scores) if o < b)
    ties = len(baseline_scores) - wins - losses
    
    total = len(baseline_scores)
    
    return {
        "win_rate": wins / total,
        "loss_rate": losses / total,
        "tie_rate": ties / total,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "total": total
    }


def calculate_improvement_metrics(
    baseline_scores: List[float],
    optimized_scores: List[float]
) -> Dict[str, Any]:
    """
    Calculate improvement metrics.
    
    Args:
        baseline_scores: Baseline scores
        optimized_scores: Optimized scores
        
    Returns:
        Improvement statistics
    """
    baseline_avg = np.mean(baseline_scores) if baseline_scores else 0
    optimized_avg = np.mean(optimized_scores) if optimized_scores else 0
    
    absolute_improvement = optimized_avg - baseline_avg
    relative_improvement = (absolute_improvement / baseline_avg * 100) if baseline_avg > 0 else 0
    
    # Per-query improvements
    per_query_improvements = [
        o - b for b, o in zip(baseline_scores, optimized_scores)
    ]
    
    return {
        "baseline_avg": float(baseline_avg),
        "optimized_avg": float(optimized_avg),
        "absolute_improvement": float(absolute_improvement),
        "relative_improvement_pct": float(relative_improvement),
        "improvement_std": float(np.std(per_query_improvements)) if per_query_improvements else 0,
        "max_improvement": float(max(per_query_improvements)) if per_query_improvements else 0,
        "min_improvement": float(min(per_query_improvements)) if per_query_improvements else 0
    }


def detect_evaluation_issues(
    evaluations: List[Dict],
    score_threshold: float = 5.0
) -> List[str]:
    """
    Detect issues in evaluation results.
    
    Args:
        evaluations: List of evaluation dictionaries
        score_threshold: Threshold for considering a score "low"
        
    Returns:
        List of detected issues
    """
    issues = []
    
    if not evaluations:
        issues.append("No evaluations available")
        return issues
    
    scores = [eval_dict.get('score', 0) for eval_dict in evaluations]
    
    # Check for consistently low scores
    low_scores = [s for s in scores if s < score_threshold]
    if len(low_scores) > len(scores) * 0.7:
        issues.append(f"High proportion of low scores: {len(low_scores)}/{len(scores)} below {score_threshold}")
    
    # Check for very low variance
    if len(scores) > 1:
        score_std = np.std(scores)
        if score_std < 0.5:
            issues.append("Very low score variance - evaluation may not be discriminative")
    
    # Check for errors in evaluations
    errors = [e for e in evaluations if 'error' in e]
    if errors:
        issues.append(f"{len(errors)} evaluation(s) returned errors")
    
    return issues


def summarize_evaluations(
    evaluations: List[Dict],
    config_name: str = "Configuration"
) -> Dict[str, Any]:
    """
    Create a summary of evaluations.
    
    Args:
        evaluations: List of evaluation dictionaries
        config_name: Name of the configuration
        
    Returns:
        Summary dictionary
    """
    if not evaluations:
        return {
            "config_name": config_name,
            "num_evaluations": 0,
            "avg_score": 0,
            "score_std": 0
        }
    
    scores = [eval_dict.get('score', 0) for eval_dict in evaluations]
    
    # Count queries by score range
    excellent = sum(1 for s in scores if s >= 8)
    good = sum(1 for s in scores if 6 <= s < 8)
    fair = sum(1 for s in scores if 4 <= s < 6)
    poor = sum(1 for s in scores if s < 4)
    
    return {
        "config_name": config_name,
        "num_evaluations": len(evaluations),
        "avg_score": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_median": float(np.median(scores)),
        "score_min": float(min(scores)),
        "score_max": float(max(scores)),
        "score_distribution": {
            "excellent (8-10)": excellent,
            "good (6-8)": good,
            "fair (4-6)": fair,
            "poor (0-4)": poor
        },
        "percentiles": {
            "p25": float(np.percentile(scores, 25)),
            "p50": float(np.percentile(scores, 50)),
            "p75": float(np.percentile(scores, 75)),
            "p90": float(np.percentile(scores, 90))
        }
    }


def create_comparison_report(
    baseline_evals: List[Dict],
    optimized_evals: List[Dict]
) -> Dict[str, Any]:
    """
    Create a comprehensive comparison report.
    
    Args:
        baseline_evals: Baseline evaluations
        optimized_evals: Optimized evaluations
        
    Returns:
        Comparison report
    """
    baseline_scores = [e.get('score', 0) for e in baseline_evals]
    optimized_scores = [e.get('score', 0) for e in optimized_evals]
    
    baseline_summary = summarize_evaluations(baseline_evals, "Baseline")
    optimized_summary = summarize_evaluations(optimized_evals, "Optimized")
    
    win_rate = calculate_win_rate(baseline_scores, optimized_scores)
    improvement = calculate_improvement_metrics(baseline_scores, optimized_scores)
    
    return {
        "baseline_summary": baseline_summary,
        "optimized_summary": optimized_summary,
        "win_rate": win_rate,
        "improvement": improvement,
        "recommendation": _generate_recommendation(improvement, win_rate)
    }


def _generate_recommendation(
    improvement: Dict[str, Any],
    win_rate: Dict[str, Any]
) -> str:
    """
    Generate recommendation based on improvement metrics.
    
    Args:
        improvement: Improvement metrics
        win_rate: Win rate statistics
        
    Returns:
        Recommendation string
    """
    rel_improvement = improvement['relative_improvement_pct']
    win_pct = win_rate['win_rate'] * 100
    
    if rel_improvement > 10 and win_pct > 60:
        return "Strong recommendation to adopt optimized configuration - significant improvement observed."
    elif rel_improvement > 5 and win_pct > 50:
        return "Moderate recommendation to adopt optimized configuration - clear improvement in most queries."
    elif rel_improvement > 0 and win_pct >= 50:
        return "Slight improvement observed. Consider adopting if improvements align with priorities."
    elif abs(rel_improvement) < 2:
        return "No significant difference between configurations. Baseline configuration is adequate."
    else:
        return "Optimized configuration performed worse. Stick with baseline or try different optimization approach."


if __name__ == "__main__":
    # Test evaluation tools
    print("=== Testing Evaluation Tools ===\n")
    
    # Sample evaluations
    baseline = [
        {'score': 6.5, 'query': 'q1'},
        {'score': 7.0, 'query': 'q2'},
        {'score': 5.5, 'query': 'q3'},
    ]
    
    optimized = [
        {'score': 7.5, 'query': 'q1'},
        {'score': 8.0, 'query': 'q2'},
        {'score': 6.0, 'query': 'q3'},
    ]
    
    summary = summarize_evaluations(baseline, "Baseline")
    print("Baseline summary:")
    print(summary)
    print()
    
    baseline_scores = [e['score'] for e in baseline]
    optimized_scores = [e['score'] for e in optimized]
    
    win_rate = calculate_win_rate(baseline_scores, optimized_scores)
    print("Win rate:")
    print(win_rate)
    print()
    
    improvement = calculate_improvement_metrics(baseline_scores, optimized_scores)
    print("Improvement metrics:")
    print(improvement)
    print()
    
    report = create_comparison_report(baseline, optimized)
    print("Comparison report recommendation:")
    print(report['recommendation'])


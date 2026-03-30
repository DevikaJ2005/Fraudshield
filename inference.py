#!/usr/bin/env python3
"""
FraudShield Inference - Kaggle Edition
Tests the environment with Kaggle real data and LLM agent

Usage:
    # First time: download data
    python download_kaggle_data.py
    
    # Then run inference
    python inference.py
    
    # Or with LLM:
    export HF_TOKEN="your_token"
    python inference_llm.py
"""

import os
import sys
import json
import logging
from typing import List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import components
from models import DecisionEnum, FraudCheckAction
from fraudshield_env import FraudShieldEnvironment
from graders import FraudShieldGrader
from llm_agent import LLMFraudDetectionAgent
from data_loader import KaggleDataLoader


class SimpleHeuristicAgent:
    """Rule-based fraud detection agent using real Kaggle patterns"""

    def decide(self, observation):
        """Make fraud decision based on transaction features"""
        data = observation.transaction_data
        
        # Rule 1: New seller + high amount
        if data.seller_account_age_days < 7 and data.amount > data.item_price * 1.5:
            return FraudCheckAction(
                transaction_id=observation.transaction_id,
                decision=DecisionEnum.FRAUD,
                confidence=0.92,
                reasoning="New seller with unusually high transaction amount relative to item price."
            )
        
        # Rule 2: New seller + risky country
        risky_countries = ["NG", "RU", "CN", "KP"]
        if data.seller_account_age_days < 14 and data.shipping_address in risky_countries:
            return FraudCheckAction(
                transaction_id=observation.transaction_id,
                decision=DecisionEnum.FRAUD,
                confidence=0.88,
                reasoning="New seller shipping to high-risk country detected."
            )
        
        # Rule 3: Previous fraud flags
        if data.previous_fraud_flags > 0:
            return FraudCheckAction(
                transaction_id=observation.transaction_id,
                decision=DecisionEnum.FRAUD,
                confidence=0.85,
                reasoning=f"Account flagged {data.previous_fraud_flags} times previously."
            )
        
        # Rule 4: Device mismatch + high amount
        if data.device_country != data.shipping_address and data.amount > 1000:
            return FraudCheckAction(
                transaction_id=observation.transaction_id,
                decision=DecisionEnum.FRAUD,
                confidence=0.75,
                reasoning="Device location mismatch with shipping location and high amount."
            )
        
        # Rule 5: Low rating sellers
        if data.seller_avg_rating < 2.0 and data.num_seller_reviews < 10:
            return FraudCheckAction(
                transaction_id=observation.transaction_id,
                decision=DecisionEnum.FRAUD,
                confidence=0.70,
                reasoning="Low-rated seller with few reviews."
            )
        
        # Default: Legitimate
        return FraudCheckAction(
            transaction_id=observation.transaction_id,
            decision=DecisionEnum.LEGITIMATE,
            confidence=0.75,
            reasoning="No major fraud indicators detected."
        )


def run_task_with_agent(
    env: FraudShieldEnvironment,
    agent,
    task_name: str,
    agent_name: str = "Rule-Based"
) -> tuple:
    """Run agent on a single task"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Running {task_name.upper()} task with {agent_name} Agent...")
    logger.info(f"{'='*70}")
    
    # Reset
    reset_result = env.reset(task_name)
    logger.info(f"Episode: {env.episode_id}")
    logger.info(f"Transactions: {len(env.current_transactions)}")
    
    predictions = []
    confidences = []
    step_count = 0
    
    # Run
    while not env.is_done:
        observation = env._get_observation() if step_count == 0 else step_result.observation
        
        try:
            action = agent.decide(observation)
        except Exception as e:
            logger.warning(f"Agent error: {e}, using fallback")
            from models import FraudCheckAction
            action = FraudCheckAction(
                transaction_id=observation.transaction_id,
                decision=DecisionEnum.LEGITIMATE,
                confidence=0.5,
                reasoning="Agent error"
            )
        
        predictions.append(action.decision.value)
        confidences.append(action.confidence)
        
        step_result = env.step(action)
        step_count += 1
        
        # Log progress
        if step_count % 20 == 0 or step_count == 1:
            logger.info(
                f"  Step {step_count:3d}: {action.decision.value:10s} "
                f"({action.confidence:.2f}) → {step_result.reward.value:+.2f}"
            )
    
    logger.info(f"✓ {task_name.upper()} complete: {step_count} steps")
    ground_truth = env.ground_truth_labels
    
    return predictions, ground_truth, confidences


def main():
    """Main inference"""
    logger.info("\n" + "="*70)
    logger.info("🚀 FraudShield Inference - Kaggle Real Data Edition")
    logger.info("="*70)
    
    # Check data
    data_loader = KaggleDataLoader(data_path="data")
    if not data_loader.load_data():
        logger.error("❌ Data not found!")
        logger.error("Run: python download_kaggle_data.py")
        sys.exit(1)
    
    logger.info("✓ Kaggle data loaded successfully")
    
    # Initialize
    env = FraudShieldEnvironment(data_path="data", seed=42)
    env.load_kaggle_data()
    agent = SimpleHeuristicAgent()
    
    logger.info("✓ Environment and agent initialized")
    
    # Run tasks
    logger.info("\n📋 Running all 3 tasks with real Kaggle data...")
    
    try:
        easy_pred, easy_truth, easy_conf = run_task_with_agent(env, agent, "easy")
        medium_pred, medium_truth, medium_conf = run_task_with_agent(env, agent, "medium")
        hard_pred, hard_truth, hard_conf = run_task_with_agent(env, agent, "hard")
    except Exception as e:
        logger.error(f"Execution error: {e}", exc_info=True)
        sys.exit(1)
    
    # Grade
    logger.info("\n" + "="*70)
    logger.info("📊 GRADING RESULTS")
    logger.info("="*70)
    
    grading_result = FraudShieldGrader.grade_all_tasks(
        easy_pred, easy_truth, easy_conf,
        medium_pred, medium_truth, medium_conf,
        hard_pred, hard_truth, hard_conf,
    )
    
    # Display results
    logger.info(f"\n📊 EASY:   {grading_result['easy']['score']:.4f}")
    logger.info(f"   Acc: {grading_result['easy']['metrics']['accuracy']:.4f} | "
                f"F1: {grading_result['easy']['metrics']['f1_score']:.4f}")
    
    logger.info(f"\n📊 MEDIUM: {grading_result['medium']['score']:.4f}")
    logger.info(f"   Precision: {grading_result['medium']['metrics']['precision']:.4f} | "
                f"F1: {grading_result['medium']['metrics']['f1_score']:.4f}")
    
    logger.info(f"\n📊 HARD:   {grading_result['hard']['score']:.4f}")
    logger.info(f"   Recall: {grading_result['hard']['metrics']['recall']:.4f} | "
                f"F1: {grading_result['hard']['metrics']['f1_score']:.4f}")
    
    logger.info("\n" + "="*70)
    logger.info(f"🏆 FINAL SCORE: {grading_result['final_score']:.4f}")
    logger.info("="*70)
    
    # Save
    output_file = "fraudshield_kaggle_results.json"
    with open(output_file, "w") as f:
        json.dump(grading_result, f, indent=2)
    logger.info(f"\n✓ Results saved to: {output_file}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("REAL DATA PERFORMANCE (Kaggle Credit Card Dataset)")
    logger.info("="*70)
    logger.info(f"Easy Score:     {grading_result['easy']['score']:.4f}")
    logger.info(f"Medium Score:   {grading_result['medium']['score']:.4f}")
    logger.info(f"Hard Score:     {grading_result['hard']['score']:.4f}")
    logger.info(f"Final Score:    {grading_result['final_score']:.4f}")
    logger.info("="*70)
    
    return grading_result


if __name__ == "__main__":
    try:
        result = main()
        logger.info("\n✓ Inference completed successfully!\n")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n✗ Inference failed: {e}", exc_info=True)
        sys.exit(1)

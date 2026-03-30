#!/usr/bin/env python3
"""
FraudShield LLM Inference Script
Tests the environment with an LLM-powered agent

Usage:
    export HF_TOKEN="your_huggingface_token_here"
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

# Import environment components
from models import DecisionEnum
from fraudshield_env import FraudShieldEnvironment
from graders import FraudShieldGrader
from llm_agent import LLMFraudDetectionAgent


def run_task(
    env: FraudShieldEnvironment,
    agent: LLMFraudDetectionAgent,
    task_name: str
) -> tuple:
    """Run LLM agent on a single task"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Running {task_name.upper()} task with LLM Agent...")
    logger.info(f"{'='*70}")
    
    # Reset environment
    reset_result = env.reset(task_name)
    logger.info(f"Started episode: {env.episode_id}")
    logger.info(f"Max steps: {env.max_steps[env.current_task]}")
    logger.info(f"Transactions to evaluate: {len(env.current_transactions)}")
    
    predictions = []
    confidences = []
    step_count = 0
    
    # Run episode
    while not env.is_done:
        # Get current observation
        observation = env._get_observation() if step_count == 0 else step_result.observation
        
        # LLM agent decides
        try:
            action = agent.decide(observation)
        except Exception as e:
            logger.error(f"LLM agent error: {e}, using fallback")
            # Fallback decision if LLM fails
            from models import FraudCheckAction
            action = FraudCheckAction(
                transaction_id=observation.transaction_id,
                decision=DecisionEnum.LEGITIMATE,
                confidence=0.5,
                reasoning="LLM unavailable"
            )
        
        predictions.append(action.decision.value)
        confidences.append(action.confidence)
        
        # Take step
        step_result = env.step(action)
        step_count += 1
        
        # Log progress
        if step_count % 10 == 0 or step_count == 1:
            logger.info(
                f"  Step {step_count:3d}: "
                f"Decision={action.decision.value:10s} "
                f"Confidence={action.confidence:.2f} "
                f"Reward={step_result.reward.value:+.2f} "
                f"Cumulative={env.cumulative_reward:+.2f}"
            )
    
    logger.info(f"\n✓ {task_name.upper()} task complete!")
    logger.info(f"  Total steps: {step_count}")
    logger.info(f"  Final cumulative reward: {env.cumulative_reward:+.2f}")
    logger.info(f"  Fraud predictions: {sum(1 for p in predictions if p == 'fraud')}")
    logger.info(f"  Legitimate predictions: {sum(1 for p in predictions if p == 'legitimate')}")
    
    # Get ground truth
    ground_truth = env.ground_truth_labels
    
    return predictions, ground_truth, confidences


def main():
    """Main LLM inference script"""
    logger.info("\n")
    logger.info("=" * 70)
    logger.info("🚀 FraudShield LLM Inference")
    logger.info("=" * 70)
    
    # Check for HF token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("\n❌ HF_TOKEN not found!")
        logger.error("Set it with: export HF_TOKEN='your_token_here'")
        logger.error("Get token from: https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    logger.info("\n✓ HF_TOKEN found")
    
    # Initialize environment and agent
    # Initialize environment and agent
    try:
        env = FraudShieldEnvironment(data_path="data", seed=42)
        env.load_kaggle_data()
        agent = LLMFraudDetectionAgent(hf_token=hf_token)
        logger.info("✓ Environment initialized with seed=42")
        logger.info("✓ Agent: LLMFraudDetectionAgent (Mistral-7B via HuggingFace)")
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        sys.exit(1)
    
    # Run all 3 tasks
    logger.info("\n📋 Running all 3 tasks...")
    
    try:
        easy_predictions, easy_ground_truth, easy_confidences = run_task(env, agent, "easy")
        medium_predictions, medium_ground_truth, medium_confidences = run_task(env, agent, "medium")
        hard_predictions, hard_ground_truth, hard_confidences = run_task(env, agent, "hard")
    except Exception as e:
        logger.error(f"Task execution error: {e}", exc_info=True)
        sys.exit(1)
    
    # Grade all tasks
    logger.info("\n" + "=" * 70)
    logger.info("📊 GRADING RESULTS")
    logger.info("=" * 70)
    
    grading_result = FraudShieldGrader.grade_all_tasks(
        easy_predictions, easy_ground_truth, easy_confidences,
        medium_predictions, medium_ground_truth, medium_confidences,
        hard_predictions, hard_ground_truth, hard_confidences,
    )
    
    # Print detailed results
    logger.info(f"\n📊 EASY Task Score: {grading_result['easy']['score']:.4f}")
    logger.info(f"   Accuracy:  {grading_result['easy']['metrics']['accuracy']:.4f}")
    logger.info(f"   Precision: {grading_result['easy']['metrics']['precision']:.4f}")
    logger.info(f"   Recall:    {grading_result['easy']['metrics']['recall']:.4f}")
    logger.info(f"   F1-Score:  {grading_result['easy']['metrics']['f1_score']:.4f}")
    
    logger.info(f"\n📊 MEDIUM Task Score: {grading_result['medium']['score']:.4f}")
    logger.info(f"   Precision: {grading_result['medium']['metrics']['precision']:.4f}")
    logger.info(f"   Recall:    {grading_result['medium']['metrics']['recall']:.4f}")
    logger.info(f"   F1-Score:  {grading_result['medium']['metrics']['f1_score']:.4f}")
    logger.info(f"   ROC-AUC:   {grading_result['medium']['metrics']['roc_auc']:.4f}")
    
    logger.info(f"\n📊 HARD Task Score: {grading_result['hard']['score']:.4f}")
    logger.info(f"   Precision: {grading_result['hard']['metrics']['precision']:.4f}")
    logger.info(f"   Recall:    {grading_result['hard']['metrics']['recall']:.4f}")
    logger.info(f"   F1-Score:  {grading_result['hard']['metrics']['f1_score']:.4f}")
    
    # Final score
    logger.info("\n" + "=" * 70)
    logger.info(f"🏆 FINAL SCORE: {grading_result['final_score']:.4f}")
    logger.info("=" * 70)
    
    # Save results to JSON
    output_file = "fraudshield_llm_results.json"
    with open(output_file, "w") as f:
        json.dump(grading_result, f, indent=2)
    logger.info(f"\n✓ Results saved to: {output_file}")
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Easy Score:     {grading_result['easy']['score']:.4f}")
    logger.info(f"Medium Score:   {grading_result['medium']['score']:.4f}")
    logger.info(f"Hard Score:     {grading_result['hard']['score']:.4f}")
    logger.info(f"Final Score:    {grading_result['final_score']:.4f}")
    logger.info("=" * 70)
    
    return grading_result


if __name__ == "__main__":
    try:
        result = main()
        logger.info("\n✓ LLM Inference completed successfully!\n")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n✗ Inference failed: {e}", exc_info=True)
        sys.exit(1)

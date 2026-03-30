"""
FraudShield LLM Agent
Uses Hugging Face Inference API for intelligent fraud detection
"""

import os
import logging
from typing import Optional
import json
import requests

from models import FraudCheckAction, DecisionEnum

logger = logging.getLogger(__name__)


class LLMFraudDetectionAgent:
    """
    AI-powered fraud detection agent using Hugging Face Inference API
    
    Uses a language model to analyze transactions and make intelligent
    fraud/legitimate decisions based on transaction features.
    """

    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize LLM agent
        
        Args:
            hf_token: HuggingFace API token (from env if not provided)
        """
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        
        if not self.hf_token:
            raise ValueError(
                "HF_TOKEN not provided. Set environment variable or pass token to __init__"
            )
        
        # Use Mistral-7B (fast and accurate)
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        self.headers = {"Authorization": f"Bearer {self.hf_token}"}
        
        logger.info(f"LLM Agent initialized with model: {self.model_id}")

    def _build_prompt(self, observation) -> str:
        """Build prompt for LLM to analyze transaction"""
        data = observation.transaction_data
        
        prompt = f"""You are a fraud detection expert analyzing an e-commerce transaction.

TRANSACTION DATA:
- Amount: ${data.amount:.2f}
- Item Price: ${data.item_price:.2f}
- Item Category: {data.item_category}
- Seller Account Age: {data.seller_account_age_days} days
- Buyer Account Age: {data.buyer_account_age_days} days
- Seller Rating: {data.seller_avg_rating}/5.0 ({data.num_seller_reviews} reviews)
- Seller Previous Fraud Flags: {data.previous_fraud_flags}
- Shipping Address: {data.shipping_address}
- Device Country: {data.device_country}
- Payment Method: {data.payment_method}
- Is Repeat Buyer: {data.is_repeat_buyer}
- Timestamp: {data.timestamp}

FRAUD INDICATORS TO CONSIDER:
1. New sellers (< 7 days) with high purchases are risky
2. Shipping to high-risk countries (NG, RU, CN, KP) is suspicious
3. Device location different from shipping location can indicate fraud
4. Accounts with previous fraud flags are risky
5. Very high amounts for new sellers are red flags
6. Repeat buyers are generally trustworthy
7. Established sellers with many reviews are trustworthy

TASK: Analyze this transaction and decide if it's FRAUD or LEGITIMATE.

Respond in this EXACT format:
DECISION: [FRAUD or LEGITIMATE]
CONFIDENCE: [0.0 to 1.0]
REASONING: [Brief explanation]"""
        
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call Hugging Face Inference API"""
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 150,
                    "temperature": 0.3,  # Lower temp = more consistent
                }
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.warning(f"HF API error: {response.status_code}")
                return None
            
            result = response.json()
            
            # Extract generated text
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            
            return None
            
        except requests.exceptions.Timeout:
            logger.warning("HF API timeout - using fallback")
            return None
        except Exception as e:
            logger.warning(f"HF API error: {e}")
            return None

    def _parse_response(self, response_text: str) -> tuple:
        """
        Parse LLM response to extract decision, confidence, reasoning
        
        Returns:
            (decision, confidence, reasoning)
        """
        try:
            if not response_text:
                return None, 0.5, "LLM unavailable"
            
            lines = response_text.split('\n')
            
            decision = None
            confidence = 0.5
            reasoning = "Unable to parse response"
            
            for line in lines:
                if "DECISION:" in line:
                    decision_text = line.split("DECISION:")[-1].strip().upper()
                    if "FRAUD" in decision_text:
                        decision = "fraud"
                    elif "LEGITIMATE" in decision_text:
                        decision = "legitimate"
                
                elif "CONFIDENCE:" in line:
                    try:
                        conf_text = line.split("CONFIDENCE:")[-1].strip()
                        confidence = float(conf_text.split()[0])
                        confidence = max(0.0, min(1.0, confidence))
                    except:
                        pass
                
                elif "REASONING:" in line:
                    reasoning = line.split("REASONING:")[-1].strip()
            
            return decision, confidence, reasoning
            
        except Exception as e:
            logger.warning(f"Parse error: {e}")
            return None, 0.5, "Parse error"

    def decide(self, observation) -> FraudCheckAction:
        """
        Make fraud/legitimate decision using LLM
        
        Args:
            observation: FraudCheckObservation
            
        Returns:
            FraudCheckAction with decision and confidence
        """
        # Build prompt
        prompt = self._build_prompt(observation)
        
        # Call LLM
        logger.info(f"Calling LLM for transaction {observation.transaction_id}")
        response = self._call_llm(prompt)
        
        # Parse response
        decision, confidence, reasoning = self._parse_response(response)
        
        # Fallback if parsing fails
        if decision is None:
            logger.warning(f"Using fallback for {observation.transaction_id}")
            decision = "legitimate"
            confidence = 0.5
            reasoning = "LLM unavailable, defaulting to legitimate"
        
        # Ensure reasoning is long enough
        if len(reasoning) < 10:
            reasoning = f"LLM analysis: {reasoning}"
        
        # Create action
        action = FraudCheckAction(
            transaction_id=observation.transaction_id,
            decision=DecisionEnum(decision),
            confidence=confidence,
            reasoning=reasoning[:500]  # Truncate if too long
        )
        
        logger.info(
            f"LLM Decision: {decision} "
            f"(confidence: {confidence:.2f}) "
            f"for {observation.transaction_id}"
        )
        
        return action

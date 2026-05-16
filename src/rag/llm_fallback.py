"""
LLM fallback for low-confidence predictions using Groq (free tier, fast online).
"""
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMFallback:
    """
    Lightweight LLM client for re-classifying or explaining tickets.
    Uses Groq (free, fast). Requires GROQ_API_KEY environment variable.
    """
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model or "llama-3.1-8b-instant"  # fast, free, good accuracy
        self.client = None
        self._init_client(api_key)

    def _init_client(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            logger.warning("GROQ_API_KEY not set. LLM fallback will not work.")
            self.client = None
            return
        try:
            from groq import Groq
            self.client = Groq(api_key=key)
            logger.info(f"Initialized Groq client with model {self.model}")
        except ImportError:
            logger.warning("Groq library not installed. Run: pip install groq")
            self.client = None

    def is_available(self) -> bool:
        """Check if Groq client is ready."""
        return self.client is not None

    def classify_ticket(self, text: str) -> Tuple[str, float]:
        """
        Classify a single ticket using LLM.
        Returns (category, confidence).
        """
        if not self.is_available():
            logger.warning("LLM not available, returning fallback category")
            return "General Inquiry", 0.5

        categories = ['Account', 'Billing', 'Fraud', 'General Inquiry', 'Technical']
        prompt = f"""Classify the following support ticket into exactly one of these categories: {', '.join(categories)}.
Ticket: "{text}"
Answer only with the category name. Do not add any extra text. If unsure, answer 'General Inquiry'."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            category = response.choices[0].message.content.strip()

            # Validate and clean category
            if category not in categories:
                for cat in categories:
                    if cat.lower() in category.lower():
                        category = cat
                        break
                else:
                    category = "General Inquiry"

            # Simple confidence heuristic (can be improved with logprobs later)
            confidence = 0.85
            return category, confidence

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return "General Inquiry", 0.5

    def explain_prediction(self, text: str, model_category: str, similar_tickets: list = None) -> str:
        """
        Generate a natural language explanation for the model's prediction.
        Optionally includes similar tickets.
        """
        if not self.is_available():
            return f"This ticket was classified as {model_category} by the model (LLM not available)."

        similar_text = ""
        if similar_tickets:
            similar_text = "\nSimilar past tickets:\n"
            for i, t in enumerate(similar_tickets[:3], 1):
                similar_text += f"{i}. [{t['metadata']['category']}] {t['metadata']['clean_text'][:150]}...\n"

        prompt = f"""You are a support analyst. The classification model predicted that the following ticket belongs to '{model_category}'.

Ticket: "{text}"
{similar_text}
Provide a short, helpful explanation of why this ticket likely belongs to '{model_category}' and, if applicable, suggest a next action. Keep it under 100 words.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            explanation = response.choices[0].message.content.strip()
            return explanation
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return f"This ticket was classified as {model_category} by the model."


if __name__ == "__main__":
    # Quick test (requires GROQ_API_KEY environment variable)
    if os.environ.get("GROQ_API_KEY"):
        llm = LLMFallback()
        test_ticket = "Someone stole my credit card and made unauthorized purchases"
        cat, conf = llm.classify_ticket(test_ticket)
        print(f"Test ticket: {test_ticket}")
        print(f"LLM classification: {cat} (confidence {conf})")
    else:
        print("Set GROQ_API_KEY to test Groq.")
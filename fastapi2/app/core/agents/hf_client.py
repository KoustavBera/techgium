"""
Hugging Face Inference Client

Client for Hugging Face Inference API to access medical LLMs.
Supports MedGemma, OpenBioLLM, and other medical models.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import os
import json
import asyncio
from datetime import datetime
import time

from app.utils import get_logger

logger = get_logger(__name__)

# HuggingFace Hub InferenceClient import
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("huggingface-hub not installed - using mock responses")


class HFModel(str, Enum):
    """Available Hugging Face models for medical validation."""
    # Primary medical models
    GPT_OSS_120B = "openai/gpt-oss-120b:groq"  # GPT-OSS-120B via Groq
    II_MEDICAL_8B = "Intelligent-Internet/II-Medical-8B-1706:featherless-ai"  # II-Medical-8B via Featherless
    
    # Legacy models (kept for compatibility)
    MEDGEMMA_4B = "google/medgemma-4b-it"
    OPENBIOLLM_8B = "aaditya/OpenBioLLM-Llama3-8B"


@dataclass
class HFConfig:
    """Configuration for Hugging Face client."""
    api_key: Optional[str] = None
    
    # Request settings
    timeout_seconds: int = 60
    max_retries: int = 3
    
    # Generation parameters
    max_tokens: int = 256
    temperature: float = 0.5  # Lower for medical consistency
    
    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")


@dataclass
class HFResponse:
    """Response from Hugging Face inference."""
    text: str
    model: str
    is_mock: bool = False
    latency_ms: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "is_mock": self.is_mock,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error
        }


class HuggingFaceClient:
    """
    Client for Hugging Face Inference API.
    
    Used for medical validation agents - NON-DECISIONAL.
    """
    
    def __init__(self, config: Optional[HFConfig] = None):
        """Initialize Hugging Face client."""
        self.config = config or HFConfig()
        self._request_count = 0
        self._last_request_time: Optional[float] = None
        
        # Check availability
        self._available = HF_AVAILABLE and self.config.api_key is not None
        
        if self._available:
            # Initialize InferenceClient
            self._client = InferenceClient(api_key=self.config.api_key)
            logger.info("HuggingFaceClient initialized with InferenceClient")
        else:
            self._client = None
            logger.info("HuggingFaceClient in mock mode (no API key or InferenceClient unavailable)")
    
    @property
    def is_available(self) -> bool:
        """Check if client is available for real requests."""
        return self._available
    
    def generate(
        self,
        prompt: str,
        model: Union[HFModel, str] = HFModel.GPT_OSS_120B,
        system_prompt: Optional[str] = None
    ) -> HFResponse:
        """
        Generate text using HuggingFace InferenceClient.
        
        Args:
            prompt: User prompt
            model: Model to use
            system_prompt: Optional system instruction
            
        Returns:
            HFResponse with generated text
        """
        model_id = model.value if isinstance(model, HFModel) else model
        
        if not self.is_available:
            return self._mock_response(prompt, model_id)
        
        start_time = time.time()
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Make request with retries
        for attempt in range(self.config.max_retries):
            try:
                completion = self._client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                
                latency = (time.time() - start_time) * 1000
                self._request_count += 1
                self._last_request_time = time.time()
                
                # Extract response text
                response_message = completion.choices[0].message
                text = response_message.content if hasattr(response_message, 'content') else str(response_message)
                
                return HFResponse(
                    text=text,
                    model=model_id,
                    is_mock=False,
                    latency_ms=latency
                )
                
            except Exception as e:
                logger.warning(f"HF API error (attempt {attempt + 1}/{self.config.max_retries}): {str(e)[:200]}")
                if attempt == self.config.max_retries - 1:
                    return self._mock_response(prompt, model_id, str(e))
                time.sleep(1.0 * (attempt + 1))  # Exponential backoff
        
        return self._mock_response(prompt, model_id, "Max retries exceeded")
    
    def _mock_response(
        self,
        prompt: str,
        model_id: str,
        error: Optional[str] = None
    ) -> HFResponse:
        """Generate mock response when API unavailable."""
        
        # Context-aware mock responses for medical validation
        if "plausibility" in prompt.lower() or "valid" in prompt.lower():
            mock_text = (
                "VALIDATION RESULT: PLAUSIBLE\n\n"
                "The biomarker values and risk assessments appear to be within "
                "physiologically reasonable ranges. The measurements are consistent "
                "with typical clinical observations.\n\n"
                "CONFIDENCE: Moderate\n"
                "CAVEATS: This is an automated preliminary check. Clinical validation required.\n\n"
                "[MOCK RESPONSE - HuggingFace API unavailable]"
            )
        elif "consistency" in prompt.lower() or "coherent" in prompt.lower():
            mock_text = (
                "CONSISTENCY CHECK: PASSED\n\n"
                "The health screening results show internal consistency. "
                "Biomarker patterns across systems are coherent and do not "
                "exhibit contradictory indicators.\n\n"
                "No major inconsistencies detected.\n\n"
                "[MOCK RESPONSE - HuggingFace API unavailable]"
            )
        else:
            mock_text = (
                "MEDICAL VALIDATION: PRELIMINARY CHECK COMPLETE\n\n"
                "The submitted health data has been reviewed. "
                "Results appear reasonable for screening purposes.\n\n"
                "IMPORTANT: This is a screening tool only. "
                "Professional medical evaluation is recommended.\n\n"
                "[MOCK RESPONSE - HuggingFace API unavailable]"
            )
        
        return HFResponse(
            text=mock_text,
            model=model_id,
            is_mock=True,
            latency_ms=10.0,
            error=error
        )
    
    async def generate_async(
        self,
        prompt: str,
        model: Union[HFModel, str] = HFModel.GPT_OSS_120B,
        system_prompt: Optional[str] = None
    ) -> HFResponse:
        """Async version of generate."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(prompt, model, system_prompt)
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "is_available": self.is_available,
            "request_count": self._request_count,
            "last_request": self._last_request_time
        }

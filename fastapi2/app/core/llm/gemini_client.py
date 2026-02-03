"""
Gemini API Client

Wrapper for Google Gemini 1.5 Flash API with rate limiting and error handling.
For health screening interpretation ONLY - non-decisional.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import os
import json
import asyncio
from datetime import datetime

from app.utils import get_logger

logger = get_logger(__name__)

# LangChain Gemini import
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("langchain-google-genai not installed - using mock responses")


class GeminiModel(str, Enum):
    """Available Gemini models."""
    FLASH_2_5 = "gemini-2.5-flash"  # Latest stable
    FLASH_1_5 = "gemini-1.5-flash-latest"  # Previous version
    PRO_1_5 = "gemini-1.5-pro-latest"


@dataclass
class GeminiConfig:
    """Configuration for Gemini client."""
    api_key: Optional[str] = None
    model: GeminiModel = GeminiModel.FLASH_2_5
    temperature: float = 1.0  # Gemini 2.5+ defaults to 1.0
    max_output_tokens: int = 2048
    top_p: float = 0.8
    top_k: int = 40
    
    # Safety settings for medical content
    safety_threshold: str = "BLOCK_ONLY_HIGH"
    
    # Rate limiting
    max_requests_per_minute: int = 60
    request_timeout_seconds: int = 30
    
    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


@dataclass
class GeminiResponse:
    """Structured response from Gemini."""
    text: str
    model: str
    finish_reason: str = "STOP"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    is_mock: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "is_mock": self.is_mock
        }


class GeminiClient:
    """
    Client for Google Gemini API.
    
    Used for explaining health screening results - NOT for diagnosis.
    """
    
    def __init__(self, config: Optional[GeminiConfig] = None):
        """
        Initialize Gemini client.
        
        Args:
            config: Optional configuration, uses defaults if not provided
        """
        self.config = config or GeminiConfig()
        self._model = None
        self._request_count = 0
        self._last_request_time = None
        self._initialized = False
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the Gemini model using LangChain."""
        if not GEMINI_AVAILABLE:
            logger.info("LangChain Gemini not available - mock mode enabled")
            self._initialized = False
            return
        
        if not self.config.api_key:
            logger.warning("No Gemini API key provided - mock mode enabled")
            self._initialized = False
            return
        
        try:
            # Initialize LangChain ChatGoogleGenerativeAI
            self._llm = ChatGoogleGenerativeAI(
                model=self.config.model.value,
                temperature=self.config.temperature,
                max_tokens=self.config.max_output_tokens,
                timeout=self.config.request_timeout_seconds,
                max_retries=2,
                google_api_key=self.config.api_key
            )
            
            self._initialized = True
            logger.info(f"LangChain Gemini client initialized with model: {self.config.model.value}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangChain Gemini: {e}")
            self._initialized = False
    
    @property
    def is_available(self) -> bool:
        """Check if Gemini is available for use."""
        return self._initialized
    
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> GeminiResponse:
        """
        Generate a response from Gemini using LangChain.
        
        Args:
            prompt: The user prompt
            system_instruction: Optional system instruction
            
        Returns:
            GeminiResponse with generated text
        """
        start_time = datetime.now()
        
        if not self.is_available:
            return self._mock_response(prompt)
        
        try:
            # Build full prompt with system instruction if provided
            if system_instruction:
                full_prompt = f"{system_instruction}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # Invoke LangChain model
            response = self._llm.invoke(full_prompt)
            
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            # Extract text from LangChain response
            text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract token usage if available
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, 'usage_metadata'):
                prompt_tokens = response.usage_metadata.get('input_tokens', 0)
                completion_tokens = response.usage_metadata.get('output_tokens', 0)
            
            self._request_count += 1
            self._last_request_time = datetime.now()
            
            return GeminiResponse(
                text=text,
                model=self.config.model.value,
                finish_reason="STOP",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency,
                is_mock=False
            )
            
        except Exception as e:
            logger.error(f"LangChain Gemini generation failed: {e}")
            return self._mock_response(prompt, error=str(e))
    
    async def generate_async(
        self,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> GeminiResponse:
        """Async version of generate."""
        # For now, run sync in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.generate(prompt, system_instruction)
        )
    
    def _mock_response(self, prompt: str, error: Optional[str] = None) -> GeminiResponse:
        """Generate a mock response when Gemini is unavailable."""
        if error:
            mock_text = f"[MOCK RESPONSE - Error: {error}]\n\n"
        else:
            mock_text = "[MOCK RESPONSE - Gemini unavailable]\n\n"
        
        # Generate contextual mock response based on prompt keywords
        if "risk" in prompt.lower():
            mock_text += (
                "Based on the provided health screening data, the following observations can be made:\n\n"
                "**Summary**: The risk assessment indicates areas requiring attention. "
                "Individual biomarkers have been analyzed and weighted according to their "
                "physiological significance.\n\n"
                "**Recommendations**: Consult with a healthcare professional for a complete "
                "evaluation. This screening provides preliminary indicators only.\n\n"
                "*Note: This is a simulated response for demonstration purposes.*"
            )
        elif "explain" in prompt.lower():
            mock_text += (
                "The health screening system has analyzed multiple physiological parameters. "
                "The results reflect biomarker measurements from various body systems. "
                "Each measurement has been validated for plausibility and cross-system consistency.\n\n"
                "*Note: This is a simulated response for demonstration purposes.*"
            )
        else:
            mock_text += (
                "Health screening analysis complete. Results have been processed through "
                "the validation pipeline and risk scoring engine.\n\n"
                "*Note: This is a simulated response for demonstration purposes.*"
            )
        
        return GeminiResponse(
            text=mock_text,
            model="mock",
            finish_reason="MOCK",
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(mock_text.split()),
            latency_ms=10.0,
            is_mock=True
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "is_available": self.is_available,
            "model": self.config.model.value,
            "request_count": self._request_count,
            "last_request": self._last_request_time.isoformat() if self._last_request_time else None
        }

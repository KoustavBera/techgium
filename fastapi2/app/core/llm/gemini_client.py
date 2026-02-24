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
    """Available Gemini models in 2026."""
    FLASH_3_0 = "gemini-3.0-flash"  # Newest state-of-the-art
    FLASH_3_0_PREVIEW = "gemini-3.0-flash-preview"
    FLASH_2_5 = "gemini-2.5-flash"  # Stable standard
    FLASH_2_5_LITE = "gemini-2.5-flash-lite"
    PRO_3_0 = "gemini-3.0-pro"  # High reasoning
    FLASH_2_0 = "gemini-2.0-flash"  # Legacy stable
    FLASH_1_5_LEGACY = "gemini-1.5-flash"  # Retired September 2025



@dataclass
class GeminiConfig:
    """Configuration for Gemini client."""
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    model: GeminiModel = GeminiModel.FLASH_2_5
    temperature: float = 0.5  # Gemini 2.5/3.0 defaults

    max_output_tokens: int = 2048
    top_p: float = 0.8
    top_k: int = 40
    
    # Safety settings for medical content
    safety_threshold: str = "BLOCK_ONLY_HIGH"
    
    # Structured Output
    response_mime_type: Optional[str] = None
    response_schema: Optional[Dict[str, Any]] = None
    
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
        self._cache = {}  # In-memory cache: {cache_key: (timestamp, response_text)}
        self._cache_ttl_seconds = 900  # 15 minutes — matches typical screening session length
        
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
            # Safely resolve model name — config.model may be a GeminiModel enum
            # member OR a plain string (e.g. when constructed from settings).
            model_name = (
                self._model_name
                if hasattr(self.config.model, "value")
                else str(self.config.model)
            )

            self._llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                timeout=self.config.request_timeout_seconds,
                max_retries=2,
                google_api_key=self.config.api_key,
                response_mime_type=self.config.response_mime_type,
                response_schema=self.config.response_schema
            )

            self._initialized = True
            logger.info(f"LangChain Gemini client initialized with model: {model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize LangChain Gemini: {e}")
            self._initialized = False

    @property
    def is_available(self) -> bool:
        """Check if Gemini is available for use."""
        return self._initialized

    @property
    def _model_name(self) -> str:
        """Safely resolve model name whether config.model is an enum or a plain string."""
        m = self.config.model
        return m.value if hasattr(m, "value") else str(m)

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        use_cache: bool = True
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
        
        # Check cache if enabled
        cache_key = None
        if use_cache:
            cache_key = self._get_cache_key(prompt, system_instruction)
            cached = self._get_from_cache(cache_key)
            if cached:
                logger.info(f"Cache hit for prompt hash {hash(cache_key) % 10000}")
                return GeminiResponse(
                    text=cached,
                    model=f"{self._model_name} (cached)",
                    finish_reason="CACHED",
                    latency_ms=1.0,
                    is_mock=False
                )
        
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
            
            # Store in cache
            if use_cache and cache_key:
                self._add_to_cache(cache_key, text)
            
            return GeminiResponse(
                text=text,
                model=self._model_name,
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
        system_instruction: Optional[str] = None,
        use_cache: bool = True
    ) -> GeminiResponse:
        """
        True async generate using LangChain's ainvoke.

        Unlike run_in_executor, ainvoke yields control to the event loop
        during the HTTP round-trip, supporting real concurrency in FastAPI.
        """
        if not self.is_available:
            return self._mock_response(prompt)

        # Check cache first (same logic as sync path)
        cache_key = None
        if use_cache:
            cache_key = self._get_cache_key(prompt, system_instruction)
            cached = self._get_from_cache(cache_key)
            if cached:
                logger.info(f"Async cache hit for prompt hash {hash(cache_key) % 10000}")
                return GeminiResponse(
                    text=cached,
                    model=f"{self._model_name} (cached)",
                    finish_reason="CACHED",
                    latency_ms=1.0,
                    is_mock=False
                )

        start_time = datetime.now()
        try:
            full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt

            # True async — yields event loop during I/O, no thread blocking
            response = await self._llm.ainvoke(full_prompt)

            latency = (datetime.now() - start_time).total_seconds() * 1000
            text = response.content if hasattr(response, "content") else str(response)

            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, "usage_metadata"):
                prompt_tokens = response.usage_metadata.get("input_tokens", 0)
                completion_tokens = response.usage_metadata.get("output_tokens", 0)

            self._request_count += 1
            self._last_request_time = datetime.now()

            if use_cache and cache_key:
                self._add_to_cache(cache_key, text)

            return GeminiResponse(
                text=text,
                model=self._model_name,
                finish_reason="STOP",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency,
                is_mock=False
            )

        except Exception as e:
            logger.error(f"Async Gemini generation failed: {e}")
            return self._mock_response(prompt, error=str(e))
    
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
    
    def _get_cache_key(self, prompt: str, system_instruction: Optional[str]) -> str:
        """
        Generate cache key that is robust to minor floating-point noise.

        Raw prompts embed live risk scores like 42.731 or confidence 0.8312,
        which differ slightly each session even for identical health profiles.
        We normalise these to 1 decimal place before hashing so that a score
        of 42.7 and 42.8 can share a cache entry (within normal rounding margin).
        """
        import hashlib, re
        # Round all floats to 1 decimal to absorb sensor-level noise
        def _round_float(m: re.Match) -> str:
            try:
                return f"{round(float(m.group()), 1)}"
            except ValueError:
                return m.group()

        normalized = re.sub(r"\d+\.\d+", _round_float, prompt)
        # Also collapse whitespace / newlines for stability
        normalized = " ".join(normalized.split())
        content = f"{system_instruction or ''}|||{normalized}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """Retrieve from cache if still valid (15-minute TTL)."""
        if cache_key in self._cache:
            cached_time, cached_text = self._cache[cache_key]
            age = (datetime.now() - cached_time).total_seconds()
            if age < self._cache_ttl_seconds:
                return cached_text
            else:
                del self._cache[cache_key]
        return None

    def _add_to_cache(self, cache_key: str, text: str):
        """Add response to cache with a max size of 500 entries."""
        self._cache[cache_key] = (datetime.now(), text)
        if len(self._cache) > 500:
            # Evict the oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]

    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "is_available": self.is_available,
            "model": self._model_name,
            "request_count": self._request_count,
            "last_request": self._last_request_time.isoformat() if self._last_request_time else None
        }

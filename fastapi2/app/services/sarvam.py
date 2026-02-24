import os
import httpx
import logging
import hashlib
from typing import Optional
from diskcache import Cache

logger = logging.getLogger(__name__)

# Initialize persistent cache (store in .sarvam_cache in the same dir)
cache_dir = os.path.join(os.path.dirname(__file__), ".sarvam_cache")
cache = Cache(cache_dir)

# Map between our session language codes (from frontend overlay) and Sarvam API codes
SUPPORTED_LANGUAGES = {
    "hi-IN": "hi-IN",
    "bn-IN": "bn-IN",
    "kn-IN": "kn-IN",
    "en-IN": "en-IN",
    "ta-IN": "ta-IN",
    "te-IN": "te-IN",
    "ml-IN": "ml-IN",
    "mr-IN": "mr-IN",
    "gu-IN": "gu-IN",
    "pa-IN": "pa-IN",
    "od-IN": "od-IN",
}

# Map langdetect/Sarvam 2-char codes back to our full codes
LANG_CODE_MAP = {
    "hi": "hi-IN",
    "bn": "bn-IN",
    "kn": "kn-IN",
    "en": "en-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "gu": "gu-IN",
    "pa": "pa-IN",
    "or": "od-IN",
}

# Language-specific TTS speakers (all Indian female voices)
SPEAKERS_BY_LANG = {
    "hi-IN": "priya",    # Hindi
    "bn-IN": "simran",   # Bengali
    "kn-IN": "kavya",    # Kannada
    "en-IN": "shreya",   # English-India
    "ta-IN": "ananya",   # Tamil
    "te-IN": "aruna",    # Telugu
    "ml-IN": "harini",   # Malayalam
    "mr-IN": "meera",    # Marathi
    "gu-IN": "divya",    # Gujarati
    "pa-IN": "priya",    # Punjabi (fallback)
    "od-IN": "priya",    # Odia (fallback)
}


class SarvamAIService:
    """
    Service wrapper for Sarvam AI APIs.
    
    Implements the Smart Indian Language Pipeline:
    1. detect_language()  — Uses Sarvam /text-lid to detect what language user typed
    2. translate_text()   — Bidirectional translation (any Indian lang ↔ English)
    3. generate_tts()     — bulbul:v3 TTS with language-specific Indian voice
    """

    def __init__(self):
        self.api_key = os.getenv("SARVAM_API_KEY")
        self.base_url = "https://api.sarvam.ai"

        if not self.api_key:
            logger.warning("SARVAM_API_KEY is not set. Sarvam AI integration will be disabled.")

    async def detect_language(self, text: str) -> str:
        """
        Detects the language of the input text.
        OPTIMIZED: Uses local `langdetect` first (Zero Cost). 
        Falls back to Sarvam /text-lid only if confidence is low.
        """
        if not text or not text.strip():
            return "en-IN"

        # 1. Primary: Local langdetect (Free & Fast)
        try:
            from langdetect import detect_langs
            langs = detect_langs(text)
            if langs:
                best_match = langs[0]
                # High confidence threshold
                if best_match.prob > 0.8:
                    detected = LANG_CODE_MAP.get(best_match.lang, "en-IN")
                    logger.info(f"langdetect primary succeeded: {best_match.lang} (conf: {best_match.prob:.2f}) → {detected}")
                    return detected
                else:
                    logger.info(f"langdetect confidence too low ({best_match.prob:.2f}), falling back to Sarvam API.")
        except Exception as e:
            logger.warning(f"langdetect primary failed: {e}. Falling back to Sarvam API.")

        # 2. Fallback: Sarvam API
        if self.api_key:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/text-lid",
                        json={"input": text.strip()[:500]},
                        headers={
                            "api-subscription-key": self.api_key,
                            "Content-Type": "application/json"
                        },
                        timeout=5.0
                    )
                    if response.status_code == 200:
                        lang_code = response.json().get("language_code", "en")
                        detected = LANG_CODE_MAP.get(lang_code, "en-IN")
                        logger.info(f"Sarvam fallback detected language: {lang_code} → {detected}")
                        return detected
            except Exception as e:
                logger.warning(f"Sarvam language detection failed: {e}")

        return "en-IN"  # Default fallback

    async def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translates text via Sarvam /translate with Diskcache to eliminate redundant API cost.
        """
        if not self.api_key or not text.strip():
            return text

        if source_lang == target_lang:
            return text

        s_lang = SUPPORTED_LANGUAGES.get(source_lang)
        t_lang = SUPPORTED_LANGUAGES.get(target_lang)
        if not s_lang or not t_lang:
            return text

        # Check Cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_key = f"trans_{s_lang}_{t_lang}_{text_hash}"
        cached_val = cache.get(cache_key)
        if cached_val:
            logger.info(f"Translation CACHE HIT: {s_lang} → {t_lang}")
            return cached_val

        payload = {
            "input": text,
            "source_language_code": s_lang,
            "target_language_code": t_lang,
            "speaker_gender": "Female",
            "mode": "formal",
            "model": "sarvam-translate:v1"
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/translate",
                    json=payload,
                    headers={
                        "api-subscription-key": self.api_key,
                        "Content-Type": "application/json"
                    },
                    timeout=15.0
                )
                if response.status_code == 200:
                    translated = response.json().get("translated_text", text)
                    # Cache result for 30 days
                    cache.set(cache_key, translated, expire=86400 * 30)
                    return translated
                else:
                    logger.error(f"Sarvam Translate error: {response.text}")
                    return text
        except Exception as e:
            logger.error(f"Sarvam Translate request failed: {e}")
            return text

    async def generate_tts(self, text: str, target_lang: str) -> Optional[str]:
        """
        Generates TTS audio via Sarvam bulbul:v3 with Diskcache to prevent huge repeated TTS costs.
        """
        if not self.api_key or not text.strip():
            return None

        t_lang = SUPPORTED_LANGUAGES.get(target_lang)
        if not t_lang:
            return None

        speaker = SPEAKERS_BY_LANG.get(target_lang, "priya")

        # Check Cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_key = f"tts_{t_lang}_{speaker}_{text_hash}"
        cached_audio = cache.get(cache_key)
        if cached_audio:
            logger.info(f"TTS CACHE HIT: {t_lang} with {speaker}")
            return cached_audio

        payload = {
            "text": text,
            "target_language_code": t_lang,
            "speaker": speaker,
            "pace": 1.0,
            "enable_preprocessing": True,
            "model": "bulbul:v3"
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/text-to-speech",
                    json=payload,
                    headers={
                        "api-subscription-key": self.api_key,
                        "Content-Type": "application/json"
                    },
                    timeout=25.0
                )
                if response.status_code == 200:
                    audios = response.json().get("audios", [])
                    if audios:
                        audio_b64 = audios[0]
                        # Cache audio for 30 days
                        cache.set(cache_key, audio_b64, expire=86400 * 30)
                        return audio_b64
                else:
                    logger.error(f"Sarvam TTS error: {response.text}")
                    return None
        except Exception as e:
            logger.error(f"Sarvam TTS request failed: {e}")
            return None


# Singleton
sarvam_service = SarvamAIService()

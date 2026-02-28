import os
import httpx
import logging
import hashlib
from typing import Optional, List
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

# Sarvam translate API max character limit per request
# sarvam-translate:v1 supports up to 2000 chars; we stay well under to be safe
_TRANSLATE_CHUNK_SIZE = 1800


def _split_into_chunks(text: str, max_chars: int = _TRANSLATE_CHUNK_SIZE) -> List[str]:
    """
    Split text into chunks that respect the Sarvam API character limit.
    Tries to split on paragraph/sentence boundaries to preserve meaning.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    # Try paragraph splits first
    paragraphs = text.split("\n\n")
    current_chunk = ""

    for para in paragraphs:
        # If a single paragraph exceeds max, split by sentences
        if len(para) > max_chars:
            sentences = para.replace(". ", ".\n").split("\n")
            for sent in sentences:
                if len(current_chunk) + len(sent) + 2 > max_chars:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sent
                else:
                    current_chunk = (current_chunk + " " + sent).strip() if current_chunk else sent
        else:
            if len(current_chunk) + len(para) + 2 > max_chars:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk = (current_chunk + "\n\n" + para).strip() if current_chunk else para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


class SarvamAIService:
    """
    Service wrapper for Sarvam AI APIs.

    Implements the Smart Indian Language Pipeline:
    1. detect_language()  — Uses Sarvam /text-lid to detect what language user typed
    2. translate_text()   — Bidirectional translation (any Indian lang ↔ English)
                            Automatically chunks long text to stay within API limits.
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
                # Lowered threshold slightly — short non-English texts often have
                # lower confidence but are still correct. Use 0.6 instead of 0.8.
                if best_match.prob > 0.6:
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

    async def _translate_single_chunk(
        self,
        client: httpx.AsyncClient,
        text: str,
        s_lang: str,
        t_lang: str,
        cache_key: str,
    ) -> str:
        """
        Translate a single chunk of text (must be within API character limits).
        Returns translated text or original on failure.
        """
        payload = {
            "input": text,
            "source_language_code": s_lang,
            "target_language_code": t_lang,
            "speaker_gender": "Female",
            "mode": "formal",
            "model": "sarvam-translate:v1",
        }

        try:
            response = await client.post(
                f"{self.base_url}/translate",
                json=payload,
                headers={
                    "api-subscription-key": self.api_key,
                    "Content-Type": "application/json",
                },
                timeout=20.0,
            )
            if response.status_code == 200:
                data = response.json()
                translated = data.get("translated_text", "").strip()
                if not translated:
                    logger.warning(f"Sarvam returned empty translated_text for chunk, using original.")
                    return text
                # Guard: Sarvam sometimes returns the input unchanged (silent failure)
                if translated == text.strip():
                    logger.warning(
                        f"Sarvam returned TEXT UNCHANGED (silent failure). "
                        f"source={s_lang}, target={t_lang}, len={len(text)}. "
                        f"NOT caching — will retry on next request."
                    )
                    return text
                # Cache this chunk's translation
                cache.set(cache_key, translated, expire=86400 * 30)
                logger.info(f"Sarvam translated chunk: {len(text)} → {len(translated)} chars ({s_lang}→{t_lang})")
                return translated
            else:
                # Log the full error response for debugging
                error_body = response.text
                logger.error(
                    f"Sarvam Translate error {response.status_code}: {error_body} "
                    f"| source={s_lang}, target={t_lang}, len={len(text)}"
                )
                return text  # Fallback: return original chunk untranslated
        except Exception as e:
            logger.error(f"Sarvam Translate request failed: {e}")
            return text

    async def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translates text via Sarvam /translate with Diskcache to eliminate redundant API cost.

        Automatically splits text longer than 1800 characters into chunks so that
        each API call stays within the sarvam-translate:v1 limit (2000 chars).
        Translated chunks are rejoined with double newlines.
        """
        if not self.api_key or not text.strip():
            return text

        if source_lang == target_lang:
            return text

        s_lang = SUPPORTED_LANGUAGES.get(source_lang)
        t_lang = SUPPORTED_LANGUAGES.get(target_lang)
        if not s_lang or not t_lang:
            logger.warning(f"Unsupported language pair: {source_lang} → {target_lang}")
            return text

        # Split into chunks that fit within the API limit
        chunks = _split_into_chunks(text, _TRANSLATE_CHUNK_SIZE)
        logger.info(
            f"Translating {s_lang} → {t_lang}: {len(text)} chars, "
            f"{len(chunks)} chunk(s)"
        )

        translated_chunks: List[str] = []
        async with httpx.AsyncClient() as client:
            for i, chunk in enumerate(chunks):
                # Check cache per-chunk
                text_hash = hashlib.md5(chunk.encode()).hexdigest()
                cache_key = f"trans_{s_lang}_{t_lang}_{text_hash}"
                cached_val = cache.get(cache_key)
                if cached_val:
                    logger.info(f"Translation CACHE HIT: chunk {i+1}/{len(chunks)}, {s_lang} → {t_lang}")
                    translated_chunks.append(cached_val)
                else:
                    translated = await self._translate_single_chunk(
                        client, chunk, s_lang, t_lang, cache_key
                    )
                    translated_chunks.append(translated)

        # Rejoin all translated chunks
        result = "\n\n".join(translated_chunks)
        logger.info(
            f"Translation complete: {len(text)} chars → {len(result)} chars "
            f"({s_lang} → {t_lang})"
        )
        return result

    async def generate_tts(self, text: str, target_lang: str) -> Optional[str]:
        """
        Generates TTS audio via Sarvam bulbul:v3 with Diskcache to prevent huge repeated TTS costs.
        Only uses the first 500 characters of text to avoid TTS length limits.
        """
        if not self.api_key or not text.strip():
            return None

        t_lang = SUPPORTED_LANGUAGES.get(target_lang)
        if not t_lang:
            return None

        speaker = SPEAKERS_BY_LANG.get(target_lang, "priya")

        # Truncate to first 500 chars for TTS (voice is a summary, not full content)
        tts_text = text[:500].strip()

        # Check Cache first
        text_hash = hashlib.md5(tts_text.encode()).hexdigest()
        cache_key = f"tts_{t_lang}_{speaker}_{text_hash}"
        cached_audio = cache.get(cache_key)
        if cached_audio:
            logger.info(f"TTS CACHE HIT: {t_lang} with {speaker}")
            return cached_audio

        payload = {
            "text": tts_text,
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

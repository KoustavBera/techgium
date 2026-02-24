import os
import asyncio
from dotenv import load_dotenv
import httpx

async def test_translate():
    load_dotenv()
    api_key = os.getenv("SARVAM_API_KEY")
    base_url = "https://api.sarvam.ai"

    text = "I'm here for you, and it's great that you reached out 🌟 Could you share how long you've been experiencing these symptoms? Also, on a scale of 1 to 10, how would you rate their severity? Lastly, have you noticed any other symptoms or triggers that seem to be related? 🌿 This will help me understand your situation better and support you more effectively."
    s_lang = "en-IN"
    t_lang = "bn-IN"

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
                f"{base_url}/translate",
                json=payload,
                headers={
                    "api-subscription-key": api_key,
                    "Content-Type": "application/json"
                },
                timeout=15.0
            )
            print("Status:", response.status_code)
            print("Response:", response.text)
    except Exception as e:
        print("Error", e)

if __name__ == "__main__":
    asyncio.run(test_translate())

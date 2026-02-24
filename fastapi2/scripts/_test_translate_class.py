import os
import asyncio
from dotenv import load_dotenv

async def test_translate():
    load_dotenv()
    from app.services.sarvam import sarvam_service

    text = "I'm here for you, and it's great that you reached out 🌟 Could you share how long you've been experiencing these symptoms? Also, on a scale of 1 to 10, how would you rate their severity? Lastly, have you noticed any other symptoms or triggers that seem to be related? 🌿 This will help me understand your situation better and support you more effectively."
    
    print(f"Translating: {text}")
    translated = await sarvam_service.translate_text(
        text=text,
        source_lang="en-IN",
        target_lang="bn-IN"
    )
    print("Result:", translated)

if __name__ == "__main__":
    asyncio.run(test_translate())

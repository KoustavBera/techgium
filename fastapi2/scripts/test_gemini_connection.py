import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

def test_gemini():
    """Simple script to test if the Gemini 2.5 Flash API is working."""
    print("Loading environment variables...")
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ ERROR: No GEMINI_API_KEY found in .env file.")
        return
        
    print(f"✅ Found API Key: {api_key[:5]}...{api_key[-4:] if len(api_key) > 9 else ''}")
    
    # Try the 2.5 Flash model
    model_name = "gemini-2.5-flash"
    print(f"\nInitializing LangChain Gemini client with model: {model_name}")
    
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.7,
            google_api_key=api_key
        )
        
        print(f"\nSending a test prompt to {model_name}...")
        prompt = "Hi, are you working correctly? Please reply with a short medical fact."
        print(f"Prompt: '{prompt}'")
        
        # Test standard text prompt
        messages = [
            HumanMessage(content=prompt)
        ]
        
        response = llm.invoke(messages)
        
        print("\n" + "="*50)
        print("🤖 GEMINI RESPONSE:")
        print("="*50)
        print(response.content)
        print("="*50)
        print("\n✅ SUCCESS: Model connection is working perfectly!")
        
    except Exception as e:
        print("\n" + "="*50)
        print("❌ ERROR: Connection failed!")
        print("="*50)
        print(f"Error details:\n{str(e)}")
        print("\nThis usually means:")
        print("1. Your API key might be invalid or expired")
        print("2. The model name might not be available in your region yet")
        print("3. You might have hit a rate limit")

if __name__ == "__main__":
    test_gemini()

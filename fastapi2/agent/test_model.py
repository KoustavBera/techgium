from llama_cpp import Llama
import os

# Path to the model
model_path = r"c:\Users\KOUSTAV BERA\OneDrive\Desktop\chiranjeevi\fastapi2\agent\llama-3-8b.Q4_K_M.gguf"

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    exit(1)

print(f"Loading model from {model_path}...")

try:
    # Initialize the model
    # n_gpu_layers=-1 to offload all layers to GPU if available, or set to 0 to run on CPU
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,  # Context window
        n_gpu_layers=-1, # Try to use GPU, change to 0 if no GPU
        verbose=True
    )

    print("Model loaded successfully!")
    
    # Test prompt
    prompt = "Q: You are a medical assistant. What is a fever? A: "
    
    print("\nGenerating response to: " + prompt)
    
    output = llm(
        prompt, 
        max_tokens=64, 
        stop=["Q:", "\n"], 
        echo=True
    )
    
    print("\nResponse:")
    print(output['choices'][0]['text'])

except Exception as e:
    print(f"Failed to load or run model: {e}")

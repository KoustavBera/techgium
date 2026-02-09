"""
AI LLM Validation Script

Tests all AI models used in the Health Screening Pipeline:
1. Gemini 2.5 Flash (via LangChain)
2. GPT-OSS-120B via Groq (via HuggingFace InferenceClient)
3. II-Medical-8B via Featherless (via HuggingFace InferenceClient)

Run: python scripts/validate_ai_llms.py
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def test_gemini():
    """Test Gemini API via LangChain."""
    console.print("\n[bold cyan]Testing Gemini 2.5 Flash...[/bold cyan]")
    
    try:
        from app.core.llm.gemini_client import GeminiClient
        
        client = GeminiClient()
        
        if not client.is_available:
            return False, "Client not available (no API key or initialization failed)", True
        
        response = client.generate(
            prompt="What is a normal resting heart rate for adults? Answer in one sentence.",
            system_instruction="You are a helpful health assistant."
        )
        
        if response.is_mock:
            return False, "Mock response received (API key may be invalid)", True
        
        return True, response.text[:200], False
        
    except Exception as e:
        return False, str(e), False

def test_hf_model(model_id: str, model_name: str):
    """Test a HuggingFace model via InferenceClient."""
    console.print(f"\n[bold cyan]Testing {model_name}...[/bold cyan]")
    
    try:
        from huggingface_hub import InferenceClient
        
        api_key = os.environ.get("HF_TOKEN")
        if not api_key:
            return False, "HF_TOKEN not found in environment", True
        
        client = InferenceClient(api_key=api_key)
        
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": "What is a normal blood pressure reading? Answer in one sentence."
                }
            ],
            max_tokens=100
        )
        
        response_text = completion.choices[0].message.content
        return True, response_text[:200], False
        
    except Exception as e:
        return False, str(e), False

def main():
    console.print(Panel.fit(
        "[bold green]AI LLM Validation Script[/bold green]\n"
        "Testing all AI models used in Health Screening Pipeline",
        border_style="green"
    ))
    
    results = []
    
    # Test 1: Gemini
    success, message, is_config_issue = test_gemini()
    results.append(("Gemini 2.5 Flash", success, message, is_config_issue))
    
    # Test 2: GPT-OSS-120B via Groq
    success, message, is_config_issue = test_hf_model(
        "openai/gpt-oss-120b:groq",
        "GPT-OSS-120B (Groq)"
    )
    results.append(("GPT-OSS-120B (Groq)", success, message, is_config_issue))
    
    # Test 3: II-Medical-8B via Featherless
    success, message, is_config_issue = test_hf_model(
        "Intelligent-Internet/II-Medical-8B-1706:featherless-ai",
        "II-Medical-8B (Featherless)"
    )
    results.append(("II-Medical-8B (Featherless)", success, message, is_config_issue))
    
    # Display results table
    console.print("\n")
    table = Table(title="Validation Results", show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan", width=30)
    table.add_column("Status", justify="center", width=10)
    table.add_column("Response / Error", width=60)
    
    all_passed = True
    for model, success, message, is_config in results:
        status = "[green]✓ PASS[/green]" if success else "[red]✗ FAIL[/red]"
        if not success:
            all_passed = False
            if is_config:
                message = f"[yellow]CONFIG:[/yellow] {message}"
            else:
                message = f"[red]ERROR:[/red] {message[:100]}..."
        else:
            message = f"[dim]{message[:100]}...[/dim]"
        
        table.add_row(model, status, message)
    
    console.print(table)
    
    # Summary
    if all_passed:
        console.print(Panel.fit(
            "[bold green]✓ All AI models are working correctly![/bold green]",
            border_style="green"
        ))
    else:
        console.print(Panel.fit(
            "[bold yellow]⚠ Some models failed validation. Check your API keys in .env[/bold yellow]",
            border_style="yellow"
        ))
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

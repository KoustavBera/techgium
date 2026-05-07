
import sys
import os
import asyncio
from typing import Dict, Any

# Add the project root to sys.path so we can import app.*
project_root = r"c:\Users\Swetanjana Maity\Desktop\kblndt\techgium\fastapi2"
if project_root not in sys.path:
    sys.path.append(project_root)

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(project_root, ".env"))
except ImportError:
    pass

# Environment variables will be loaded from .env via the app config or manual export
token = os.environ.get("HF_TOKEN", "NOT_SET")
print(f"DEBUG: HF_TOKEN starts with: {token[:5]}... (Length: {len(token)})")

from app.core.agents.hf_client import HuggingFaceClient, HFConfig
from app.core.agents.medical_agents import MedGemmaAgent, AgentConsensus
from app.core.extraction.base import PhysiologicalSystem

async def test_agents():
    print("=== Agent Folder Health Check ===")
    
    # 1. Test HF Client
    print("\n[1/3] Testing HuggingFaceClient...")
    try:
        hf_client = HuggingFaceClient()
        stats = hf_client.get_stats()
        print(f"  - Client initialized: {'Yes' if hf_client else 'No'}")
        print(f"  - Real API available: {'Yes' if stats['is_available'] else 'No (Mock Mode Active)'}")
        
        # Simple generation test
        response = await hf_client.generate_async("Hello, identify as a medical assistant.")
        print(f"  - Response status: {'Success' if response.text else 'Failed'}")
        print(f"  - Is Mock: {response.is_mock}")
    except Exception as e:
        print(f"  - HF Client Error: {e}")

    # 2. Test Medical Agent
    print("\n[2/3] Testing MedGemmaAgent...")
    try:
        agent = MedGemmaAgent()
        dummy_data = {"heart_rate": {"value": 72, "unit": "bpm", "status": "normal"}}
        result = agent.validate_biomarkers(dummy_data, PhysiologicalSystem.CARDIOVASCULAR)
        print(f"  - Agent validation status: {result.status.value}")
        print(f"  - Flags raised: {len(result.flags)}")
        print(f"  - Explanation: {result.explanation[:60]}...")
    except Exception as e:
        print(f"  - Medical Agent Error: {e}")

    # 3. Test Consensus Orchestrator
    print("\n[3/3] Testing AgentConsensus...")
    try:
        consensus = AgentConsensus()
        # Mock result for consensus testing
        from app.core.agents.medical_agents import ValidationResult, ValidationStatus
        mock_results = {
            "MedGemma": ValidationResult(agent_name="MedGemma", model_used="medgemma", status=ValidationStatus.VALID, confidence=0.9),
            "OpenBioLLM": ValidationResult(agent_name="OpenBioLLM", model_used="openbiollm", status=ValidationStatus.VALID, confidence=0.85)
        }
        res = consensus.compute_consensus(mock_results)
        print(f"  - Consensus status: {res.overall_status.value}")
        print(f"  - Agent agreement: {res.agent_agreement:.0%}")
        print(f"  - Recommendation: {res.recommendation[:60]}...")
    except Exception as e:
        print(f"  - Consensus Error: {e}")

    print("\n=== Check Complete ===")

if __name__ == "__main__":
    asyncio.run(test_agents())


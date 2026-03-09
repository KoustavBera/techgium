#!/usr/bin/env python3
"""
Modal Deployment Quick Start Script
Automates setup and deployment process
"""
import subprocess
import sys
from pathlib import Path
from typing import Optional


class ModalSetup:
    def __init__(self):
        self.app_name = "health-screening-pipeline"
        self.script_dir = Path(__file__).parent
        
    def run_command(self, cmd: list, description: str = "", check: bool = True) -> bool:
        """Run command and handle errors"""
        print(f"\n📦 {description}")
        print(f"   $ {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {e.stderr}")
            return False
    
    def check_modal_installed(self) -> bool:
        """Check if Modal is installed"""
        return self.run_command(
            ["modal", "--version"],
            "Checking Modal installation...",
            check=False
        )
    
    def install_modal(self) -> bool:
        """Install Modal CLI"""
        return self.run_command(
            ["pip", "install", "modal"],
            "Installing Modal..."
        )
    
    def authenticate_modal(self) -> bool:
        """Authenticate with Modal"""
        response = input("\n✨ Authenticate with Modal? (y/n): ")
        if response.lower() == 'y':
            self.run_command(
                ["modal", "token", "new"],
                "Authenticating with Modal..."
            )
            return True
        return False
    
    def create_secrets(self) -> bool:
        """Guide user through secret creation"""
        print("\n" + "="*60)
        print("🔐 CREATE MODAL SECRETS")
        print("="*60)
        
        # Gemini Secret
        print("\n1️⃣  Google Gemini API Key")
        print("   Get your key from: https://aistudio.google.com/app/apikey")
        
        response = input("   Enter your GOOGLE_API_KEY (paste and press Enter): ").strip()
        if response:
            cmd = ["modal", "secret", "create", "gemini-secrets"]
            print(f"   $ {' '.join(cmd)}")
            try:
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                output, error = process.communicate(input=f"GOOGLE_API_KEY\n{response}\n")
                print("   ✓ Gemini secret created")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                return False
        
        # HuggingFace Secret
        print("\n2️⃣  HuggingFace API Token")
        print("   Get your token from: https://huggingface.co/settings/tokens")
        
        response = input("   Enter your HUGGINGFACE_TOKEN (paste and press Enter): ").strip()
        if response:
            cmd = ["modal", "secret", "create", "huggingface-secrets"]
            print(f"   $ {' '.join(cmd)}")
            try:
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                output, error = process.communicate(input=f"HUGGINGFACE_TOKEN\n{response}\n")
                print("   ✓ HuggingFace secret created")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                return False
        
        # List secrets
        print("\n📋 Verifying secrets...")
        self.run_command(
            ["modal", "secret", "list"],
            "Listing secrets...",
            check=False
        )
        
        return True
    
    def create_volumes(self) -> bool:
        """Create persistent volumes"""
        print("\n" + "="*60)
        print("💾 CREATE PERSISTENT VOLUMES")
        print("="*60)
        
        volumes = [
            "health-screening-models",
            "health-screening-reports",
            "health-screening-cache",
        ]
        
        for vol in volumes:
            self.run_command(
                ["modal", "volume", "create", vol],
                f"Creating volume: {vol}",
                check=False
            )
        
        print("\n✓ All volumes created")
        return True
    
    def upload_models(self) -> bool:
        """Upload models to volume"""
        model_file = self.script_dir / "pose_landmarker.task"
        
        if model_file.exists():
            print(f"\n📤 Uploading {model_file.name} to Modal volume...")
            return self.run_command(
                [
                    "modal", "volume", "put",
                    "health-screening-models",
                    str(model_file),
                    "/app/models/pose_landmarker.task"
                ],
                "Uploading model..."
            )
        else:
            print(f"\n⚠️  Model file not found: {model_file}")
            print("   You can upload it later with:")
            print(f"   modal volume put health-screening-models ./pose_landmarker.task /app/models/")
            return True
    
    def deploy_app(self) -> bool:
        """Deploy to Modal"""
        print("\n" + "="*60)
        print("🚀 DEPLOY TO MODAL")
        print("="*60)
        
        response = input("\nDeploy now? (y/n): ")
        if response.lower() == 'y':
            return self.run_command(
                ["modal", "deploy", "modal_app.py"],
                "Deploying to Modal...",
            )
        return False
    
    def verify_deployment(self) -> bool:
        """Verify deployment"""
        print("\n" + "="*60)
        print("✅ DEPLOYMENT VERIFICATION")
        print("="*60)
        
        self.run_command(
            ["modal", "app", "list"],
            "Listing deployed apps...",
            check=False
        )
        
        print(f"\n✓ App deployed! Access it at:")
        print(f"   https://your-username--{self.app_name}.modal.run")
        
        return True
    
    def print_next_steps(self):
        """Print helpful next steps"""
        print("\n" + "="*60)
        print("📚 NEXT STEPS")
        print("="*60)
        print("""
1. Test your deployment:
   curl https://your-username--health-screening-pipeline.modal.run/health

2. Monitor logs:
   modal logs health-screening-pipeline

3. Check costs:
   modal app cost health-screening-pipeline

4. For updates:
   modal deploy modal_app.py

5. Documentation:
   See MODAL_DEPLOYMENT.md for detailed guide

🔗 Resources:
   - Modal Docs: https://modal.com/docs
   - Discord: https://modal.com/discord
   - Support: https://modal.com/support
        """)
    
    def run_full_setup(self):
        """Run complete setup"""
        print("""
╔══════════════════════════════════════════════════════════╗
║  Modal Serverless Deployment - Health Screening Pipeline ║
║                   Quick Start Setup                       ║
╚══════════════════════════════════════════════════════════╝
        """)
        
        steps = [
            ("Modal CLI", self.check_modal_installed, self.install_modal),
            ("Secrets", self.create_secrets, None),
            ("Volumes", self.create_volumes, None),
            ("Models", self.upload_models, None),
            ("Deploy", self.deploy_app, None),
            ("Verify", self.verify_deployment, None),
        ]
        
        for step_name, check_fn, fallback_fn in steps:
            print(f"\n{'='*60}")
            print(f"STEP: {step_name}")
            print('='*60)
            
            if not check_fn():
                if fallback_fn and fallback_fn():
                    continue
                else:
                    response = input(f"\n⚠️  Continue anyway? (y/n): ")
                    if response.lower() != 'y':
                        print("Exiting.")
                        return False
        
        self.print_next_steps()
        return True


def main():
    try:
        setup = ModalSetup()
        
        if len(sys.argv) > 1 and sys.argv[1] == "--full":
            setup.run_full_setup()
        else:
            print("""
Modal Quick Start Commands
===========================

1. Full interactive setup:
   python modal_setup.py --full

2. Individual steps:
   python modal_setup.py check      # Check Modal installation
   python modal_setup.py auth       # Authenticate
   python modal_setup.py secrets    # Setup secrets
   python modal_setup.py volumes    # Create volumes
   python modal_setup.py deploy     # Deploy app
   python modal_setup.py verify     # Verify deployment

3. Get help:
   modal --help
   modal deploy --help
            """)
    
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

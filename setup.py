#!/usr/bin/env python3
"""
Setup script for TDS Virtual TA
This script helps prepare the environment and data for the Virtual TA
"""

import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

def check_data_structure():
    """Check if the required data structure exists"""
    required_paths = [
        "data/tools-in-data-science-public",
        "data/tds_discourse_posts.json"
    ]
    
    missing_paths = []
    for path in required_paths:
        if not Path(path).exists():
            missing_paths.append(path)
    
    if missing_paths:
        print("❌ Missing required data files/directories:")
        for path in missing_paths:
            print(f"   - {path}")
        print("\nPlease ensure you have:")
        print("1. Course materials in 'data/tools-in-data-science-public/'")
        print("2. Forum posts in 'data/tds_discourse_posts_1.json'")
        return False
    
    print("✅ All required data files/directories found")
    return True

def check_environment():
    """Check if environment variables are set"""
    missing_env = []
    # if not os.getenv('GEMINI_API_KEY'):
    #     missing_env.append('GEMINI_API_KEY')
    if not os.getenv('AIPIPE_TOKEN'):
        missing_env.append('AIPIPE_TOKEN')
    if missing_env:
        for var in missing_env:
            print(f"❌ {var} environment variable not set")
        print("Please set the missing API keys in the .env file or environment")
        return False
    
    print("✅ Environment variables configured")
    return True

def create_directories():
    """Create necessary directories"""
    dirs = ["embeddings", "logs"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}")

def main():
    """Main setup function"""
    print("🚀 Setting up TDS Virtual TA...")
    
    # Check data structure
    if not check_data_structure():
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        print("💡 Create a .env file with your API keys")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run locally: python app.py")
    print("3. Test with: curl -X POST http://localhost:5000/api/ -H 'Content-Type: application/json' -d '{\"question\": \"What is TDS?\"}'")

if __name__ == "__main__":
    main()
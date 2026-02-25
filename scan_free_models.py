import os
import requests
import json
import time
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

def print_header(title):
    print("\n" + "="*50)
    print(f" {title}")
    print("="*50)

def fetch_openrouter_models():
    print_header("OpenRouter (无需 Key)")
    try:
        response = requests.get("https://openrouter.ai/api/v1/models")
        response.raise_for_status()
        models = response.json()["data"]
        free_models = [m for m in models if ":free" in m["id"]]
        
        print(f"Found {len(free_models)} free models:")
        for model in free_models:
            print(f" - {model['name']} ({model['id']})")
            
    except Exception as e:
        print(f"Error fetching OpenRouter models: {e}")

def fetch_github_models():
    print_header("GitHub Models (无需 Key)")
    try:
        # GitHub Marketplace API usually requires scraping or specific API access
        # For this script, we'll try to fetch from the public endpoint used in the original script
        # Note: The original script iterates pages. We'll just fetch page 1 for quick check.
        url = "https://github.com/marketplace?type=models&page=1"
        response = requests.get(
            url,
            headers={
                "Accept": "application/json",
                "x-requested-with": "XMLHttpRequest", 
            }
        )
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            print(f"Found {len(results)} models on page 1 (Sample):")
            for model in results:
                name = model.get("friendly_name", "Unknown")
                print(f" - {name}")
        else:
            print(f"Could not access GitHub Models API (Status: {response.status_code})")
            
    except Exception as e:
        print(f"Error fetching GitHub models: {e}")

def check_configured_providers():
    print_header("Configured Providers (需要 .env Key)")
    
    providers = [
        {"name": "Mistral AI", "env": "MISTRAL_API_KEY"},
        {"name": "Groq", "env": "GROQ_API_KEY"},
        {"name": "Google Gemini", "env": "GCP_PROJECT_ID"}, # Usually needs more than just ID
        {"name": "Cohere", "env": "COHERE_API_KEY"},
        {"name": "Hyperbolic", "env": "HYPERBOLIC_API_KEY"},
        {"name": "Scaleway", "env": "SCALEWAY_API_KEY"},
        {"name": "Cloudflare", "env": "CLOUDFLARE_API_KEY"},
        {"name": "Lambda Labs", "env": "LAMBDA_API_KEY"},
    ]
    
    for p in providers:
        key = os.environ.get(p["env"])
        if key:
            print(f"[√] {p['name']}: Configured (Key found)")
            # In a full version, we could call their APIs here to check validity/quota
        else:
            print(f"[x] {p['name']}: Not Configured (Missing {p['env']})")

def main():
    print("正在扫描免费 AI 模型资源...")
    print("Scanning for free AI model resources...")
    
    # 1. OpenRouter (Public)
    fetch_openrouter_models()
    
    # 2. GitHub Models (Public)
    fetch_github_models()
    
    # 3. Check Local Configuration
    check_configured_providers()
    
    print("\n" + "="*50)
    print("扫描完成 | Scan Complete")
    input("按回车键退出... Press Enter to exit...")

if __name__ == "__main__":
    main()

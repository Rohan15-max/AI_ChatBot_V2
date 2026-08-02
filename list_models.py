# list_models.py
from google import genai
import os

# Get API key from environment or enter it here
API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")

client = genai.Client(api_key=API_KEY)

print("\n" + "="*80)
print("AVAILABLE GEMINI MODELS")
print("="*80 + "\n")

for model in client.models.list():
    # Only show models that support generateContent (chat/text generation)
    if "generateContent" in model.supported_actions:
        print(f"📌 {model.name}")
        print(f"   Display: {model.display_name}")
        print(f"   Input tokens: {model.input_token_limit:,}")
        print(f"   Output tokens: {model.output_token_limit:,}")
        print(f"   Supports: {', '.join(model.supported_actions)}")
        print("-" * 50)
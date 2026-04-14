import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), http_options={'api_version': 'v1'})

print("--- AVAILABLE MODELS FOR YOUR KEY ---")
for model in client.models.list():
    print(f"MODEL ID: {model.name}")
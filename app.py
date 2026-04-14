import os
from flask import Flask, request, jsonify, render_template
from google import genai
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# This setup forces the Stable v1 connection
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={'api_version': 'v1'}
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    try:
        # We are using the 2.0-flash model which replaced 1.5 in 2026
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", 
            contents=user_message
        )
        return jsonify({"reply": response.text})
    except Exception as e:
        # If this fails, it will tell us exactly why in the chat bubble
        return jsonify({"reply": f"Developer Note: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
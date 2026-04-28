"""
GeminiCode Backend — Gemini 3 Flash Code Assistant
Install: pip install google-genai flask flask-cors gunicorn
"""

import os
import json
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types

app = Flask(__name__)
CORS(app, origins='*')

MODEL = "gemini-3-flash-preview"


def get_client():
    """Read API key fresh every request — fixes stale key issues on Render."""
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in environment variables")
    return genai.Client(api_key=api_key)


def build_system_prompt(lang="python"):
    return f"""You are GeminiCode, an expert AI code assistant. You help developers write, debug, explain, review and improve code.
When analyzing code: be precise, point out line numbers, provide working examples in markdown code blocks, highlight bugs and security issues.
Current language: {lang}"""


@app.route('/api/debug', methods=['GET'])
def debug():
    """Visit this URL on your Render app to verify the key is loaded."""
    
    masked = (key[:8] + "..." + key[-4:]) if len(key) > 12 else "NOT SET or TOO SHORT"
    return jsonify({
        "key_is_set": bool(key),
        "key_preview": masked,
        "key_length": len(key),
        "model": MODEL,
        "env_vars_with_api": [k for k in os.environ if "API" in k.upper() or "GEMINI" in k.upper()]
    })


@app.route('/api/health', methods=['GET'])
def health():
    key = os.environ.get("GEMINI_API_KEY", "")
    return jsonify({'status': 'ok', 'key_set': bool(key), 'model': MODEL})


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    prompt = data.get('prompt', '').strip()
    code = data.get('code', '').strip()
    lang = data.get('lang', 'python')
    use_search = data.get('useSearch', False)

    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    user_message = f"Here is my {lang} code:\n\n```{lang}\n{code}\n```\n\n{prompt}" if code else prompt

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_message)])]
    tools = [types.Tool(googleSearch=types.GoogleSearch())] if use_search else []
    config = types.GenerateContentConfig(
        system_instruction=build_system_prompt(lang),
        thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        tools=tools if tools else None,
    )

    try:
        client = get_client()
        response = client.models.generate_content(model=MODEL, contents=contents, config=config)
        return jsonify({'reply': response.text or "No response.", 'model': MODEL})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/explain', methods=['POST'])
def explain():
    data = request.get_json()
    code = data.get('code', '').strip()
    lang = data.get('lang', 'python')
    if not code:
        return jsonify({'error': 'No code provided'}), 400

    prompt = f"Explain this {lang} code step by step:\n\n```{lang}\n{code}\n```"
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    config = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="HIGH"))

    try:
        client = get_client()
        response = client.models.generate_content(model=MODEL, contents=contents, config=config)
        return jsonify({'explanation': response.text, 'model': MODEL})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/review', methods=['POST'])
def review():
    data = request.get_json()
    code = data.get('code', '').strip()
    lang = data.get('lang', 'python')
    if not code:
        return jsonify({'error': 'No code provided'}), 400

    prompt = f"Review this {lang} code for bugs, security, performance and best practices. Provide improved version:\n\n```{lang}\n{code}\n```"
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    config = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="HIGH"))

    try:
        client = get_client()
        response = client.models.generate_content(model=MODEL, contents=contents, config=config)
        return jsonify({'review': response.text, 'model': MODEL})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    key = os.environ.get("GEMINI_API_KEY", "")
    print(f"API Key: {'SET ✓' if key else 'NOT SET ✗'}")
    app.run(host='0.0.0.0', port=5000, debug=True)

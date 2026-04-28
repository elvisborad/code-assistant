"""
GeminiCode Backend — Gemini 3 Flash Code Assistant
Run: python app.py
Install: pip install google-genai flask flask-cors
"""

import os
import json
from flask import Flask, request, Response, send_from_directory
from flask_cors import CORS
from google import genai
from google.genai import types

app = Flask(__name__, static_folder='.')
CORS(app)

# ── CONFIG ──
GEMINI_API_KEY = os.environ.get("Gemini API Key", "AIzaSyAK1B_wjrv6ADYFg4GkCFWmI3QNhGaDvoo")
MODEL = "gemini-3-flash-preview"

client = genai.Client(api_key=Gemini API Key)


def build_system_prompt(lang: str = "python") -> str:
    return f"""You are GeminiCode, an expert AI code assistant. You help developers write, debug, explain, review and improve code.

When analyzing code:
- Be precise and technical
- Point out specific line numbers when relevant
- Provide working, runnable code examples
- Use markdown code blocks with language tags like ```python or ```javascript
- Be concise but thorough
- Highlight potential bugs, performance issues, or security concerns
- Suggest best practices and modern patterns

Current language context: {lang}

Format your responses clearly. When providing code, always wrap it in proper markdown code blocks."""


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Non-streaming chat endpoint."""
    data = request.get_json()
    prompt = data.get('prompt', '').strip()
    code = data.get('code', '').strip()
    lang = data.get('lang', 'python')
    use_search = data.get('useSearch', False)
    mode = data.get('mode', 'chat')

    if not prompt:
        return {'error': 'No prompt provided'}, 400

    # Build user message
    user_message = prompt
    if code:
        user_message = f"Here is my {lang} code:\n\n```{lang}\n{code}\n```\n\n{prompt}"

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        )
    ]

    # Configure tools
    tools = []
    if use_search:
        tools.append(types.Tool(googleSearch=types.GoogleSearch()))

    config = types.GenerateContentConfig(
        system_instruction=build_system_prompt(lang),
        thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        tools=tools if tools else None,
    )

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=config,
        )
        reply = response.text or "No response generated."
        return {'reply': reply, 'model': MODEL}
    except Exception as e:
        return {'error': str(e)}, 500


@app.route('/api/stream', methods=['POST'])
def stream():
    """Streaming chat endpoint — yields chunks as SSE."""
    data = request.get_json()
    prompt = data.get('prompt', '').strip()
    code = data.get('code', '').strip()
    lang = data.get('lang', 'python')
    use_search = data.get('useSearch', False)

    if not prompt:
        return {'error': 'No prompt provided'}, 400

    user_message = prompt
    if code:
        user_message = f"Here is my {lang} code:\n\n```{lang}\n{code}\n```\n\n{prompt}"

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        )
    ]

    tools = []
    if use_search:
        tools.append(types.Tool(googleSearch=types.GoogleSearch()))

    config = types.GenerateContentConfig(
        system_instruction=build_system_prompt(lang),
        thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        tools=tools if tools else None,
    )

    def generate():
        try:
            for chunk in client.models.generate_content_stream(
                model=MODEL,
                contents=contents,
                config=config,
            ):
                if chunk.text:
                    # SSE format
                    yield f"data: {json.dumps({'text': chunk.text})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


@app.route('/api/explain', methods=['POST'])
def explain():
    """Dedicated explain endpoint — returns structured explanation."""
    data = request.get_json()
    code = data.get('code', '').strip()
    lang = data.get('lang', 'python')

    if not code:
        return {'error': 'No code provided'}, 400

    prompt = f"""Analyze this {lang} code and provide:

1. **Overview** — What does this code do in 1-2 sentences?
2. **Line-by-line breakdown** — Explain each section clearly
3. **Key concepts** — What patterns, algorithms, or techniques are used?
4. **Potential issues** — Any bugs, edge cases, or improvements?
5. **Complexity** — Time and space complexity if applicable

Code:
```{lang}
{code}
```"""

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
    )

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=config,
        )
        return {'explanation': response.text, 'model': MODEL}
    except Exception as e:
        return {'error': str(e)}, 500


@app.route('/api/review', methods=['POST'])
def review():
    """Code review endpoint."""
    data = request.get_json()
    code = data.get('code', '').strip()
    lang = data.get('lang', 'python')

    if not code:
        return {'error': 'No code provided'}, 400

    prompt = f"""Perform a comprehensive code review of this {lang} code.

Review for:
- **Bugs & Logic errors** — Are there any correctness issues?
- **Security** — SQL injection, XSS, unsafe operations, etc.
- **Performance** — Inefficient algorithms, unnecessary operations
- **Readability** — Naming conventions, clarity, complexity
- **Best practices** — Language-specific idioms and patterns
- **Testing** — What test cases would be needed?

Provide an improved version of the code at the end.

```{lang}
{code}
```"""

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
    )

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=config,
        )
        return {'review': response.text, 'model': MODEL}
    except Exception as e:
        return {'error': str(e)}, 500


@app.route('/api/health', methods=['GET'])
def health():
    return {'status': 'ok', 'model': MODEL, 'thinking': 'HIGH'}


if __name__ == '__main__':
    print("═" * 50)
    print("  GeminiCode Backend")
    print(f"  Model: {MODEL}")
    print("  Thinking: HIGH")
    print("  Web Search: Available")
    print("═" * 50)

    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("\n  ⚠ WARNING: Set your GEMINI_API_KEY environment variable")
        print("  export GEMINI_API_KEY=your_key_here\n")

    app.run(host='0.0.0.0', port=5000, debug=True)

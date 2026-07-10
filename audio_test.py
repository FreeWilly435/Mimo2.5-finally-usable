#!/usr/bin/env python3
"""Functional audio-input test for MiMo-V2.5 omni on :8000 (stdlib only)."""
import base64, json, sys, urllib.request

WAV = "/usr/share/sounds/alsa/Front_Center.wav"
URL = "http://localhost:8000/v1/chat/completions"

with open(WAV, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

payload = {
    "model": "MiMo-V2.5",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "You are given an audio clip. Transcribe or describe exactly what is spoken or heard. Be brief."},
            {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
        ],
    }],
    "max_tokens": 512,
    "temperature": 0.0,
}

req = urllib.request.Request(
    URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
)
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    print("=== AUDIO RESPONSE ===")
    if reasoning:
        print("[reasoning]:", reasoning[:400])
    print("[content]:", content.strip())
    print("=== usage:", d.get("usage"))
    print("AUDIO_TEST_RESULT: OK")
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:600])
    print("AUDIO_TEST_RESULT: FAIL")
except Exception as e:
    print("ERR", repr(e))
    print("AUDIO_TEST_RESULT: FAIL")

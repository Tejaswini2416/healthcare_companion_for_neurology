"""
Audio Text-to-Speech (TTS) Accessibility Player Component
Provides an accessible, browser-native Web Speech API audio player
specifically designed for neuro-oncology patients dealing with
visual field deficits (hemianopia), diplopia, or fatigue.
Supports English, Spanish, Hindi, Mandarin, and French with speed control.
"""

import streamlit as st
import streamlit.components.v1 as components
import json


def render_audio_tts_player(
    text_to_speak: str,
    language_code: str = "en-US",
    label: str = "Listen to Audio Summary",
    height: int = 110
):
    """
    Renders an accessible Web Speech API audio controller in Streamlit.
    """
    # Sanitize text for JavaScript string literal
    escaped_text = json.dumps(text_to_speak)

    tts_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }}
            body {{
                background: transparent;
                padding: 6px 0;
            }}
            .tts-card {{
                background: #f8fafc;
                border: 1.5px solid #cbd5e1;
                border-radius: 10px;
                padding: 10px 14px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }}
            .tts-left {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .tts-title {{
                font-size: 13px;
                font-weight: 700;
                color: #0f172a;
            }}
            .tts-subtitle {{
                font-size: 11px;
                color: #64748b;
            }}
            .btn-group {{
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .btn-tts {{
                background: #0f766e;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 14px;
                font-size: 12px;
                font-weight: 600;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 6px;
                transition: background 0.2s;
            }}
            .btn-tts:hover {{
                background: #0d9488;
            }}
            .btn-ctrl {{
                background: #e2e8f0;
                color: #334155;
                border: none;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 12px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
            }}
            .btn-ctrl:hover {{
                background: #cbd5e1;
            }}
            .speed-select {{
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 5px 8px;
                font-size: 11px;
                color: #334155;
                background: white;
                cursor: pointer;
            }}
            .status-indicator {{
                font-size: 11px;
                font-weight: 600;
                color: #0f766e;
                margin-left: 6px;
                display: none;
            }}
        </style>
    </head>
    <body>
        <div class="tts-card">
            <div class="tts-left">
                <span style="font-size: 22px;">🎧</span>
                <div>
                    <div class="tts-title">{label}</div>
                    <div class="tts-subtitle">Audio speech synthesis for visual accessibility</div>
                </div>
            </div>

            <div class="btn-group">
                <select class="speed-select" id="speech-speed" title="Playback Speed">
                    <option value="0.85">0.85x Gentle</option>
                    <option value="1.0" selected>1.0x Normal</option>
                    <option value="1.15">1.15x Brisk</option>
                </select>

                <button class="btn-tts" id="btn-play">
                    <span>▶️ Listen</span>
                </button>
                <button class="btn-ctrl" id="btn-pause" title="Pause / Resume">⏸️ Pause</button>
                <button class="btn-ctrl" id="btn-stop" title="Stop Audio">⏹️ Stop</button>
                <span class="status-indicator" id="tts-status">🔊 Playing...</span>
            </div>
        </div>

        <script>
            const textToSpeak = {escaped_text};
            const langCode = "{language_code}";
            let synth = window.speechSynthesis;
            let utterance = null;
            let isPaused = false;

            const btnPlay = document.getElementById('btn-play');
            const btnPause = document.getElementById('btn-pause');
            const btnStop = document.getElementById('btn-stop');
            const speedSelect = document.getElementById('speech-speed');
            const statusIndicator = document.getElementById('tts-status');

            function stopSpeech() {{
                if (synth) {{
                    synth.cancel();
                    statusIndicator.style.display = 'none';
                    isPaused = false;
                    btnPause.textContent = '⏸️ Pause';
                }}
            }}

            btnPlay.addEventListener('click', () => {{
                if (!synth) return;
                stopSpeech();

                utterance = new SpeechSynthesisUtterance(textToSpeak);
                utterance.lang = langCode;
                utterance.rate = parseFloat(speedSelect.value);

                // Voice resolution
                const voices = synth.getVoices();
                const matchedVoice = voices.find(v => v.lang.startsWith(langCode.substring(0, 2)));
                if (matchedVoice) {{
                    utterance.voice = matchedVoice;
                }}

                utterance.onstart = () => {{
                    statusIndicator.style.display = 'inline';
                    statusIndicator.textContent = '🔊 Playing...';
                }};

                utterance.onend = () => {{
                    statusIndicator.style.display = 'none';
                    isPaused = false;
                }};

                utterance.onerror = () => {{
                    statusIndicator.style.display = 'none';
                }};

                synth.speak(utterance);
            }});

            btnPause.addEventListener('click', () => {{
                if (!synth || !synth.speaking) return;
                if (!isPaused) {{
                    synth.pause();
                    isPaused = true;
                    btnPause.textContent = '▶️ Resume';
                    statusIndicator.textContent = '⏸️ Paused';
                }} else {{
                    synth.resume();
                    isPaused = false;
                    btnPause.textContent = '⏸️ Pause';
                    statusIndicator.textContent = '🔊 Playing...';
                }}
            }});

            btnStop.addEventListener('click', stopSpeech);
        </script>
    </body>
    </html>
    """

    components.html(tts_html, height=height)

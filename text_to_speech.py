import argparse
from pathlib import Path
import pyttsx3

def build_engine(rate: int = 150, volume: float = 1.0, voice_hint: str | None = None) -> pyttsx3.Engine:
    engine = pyttsx3.init()
    engine.setProperty("rate", rate) # Tốc độ 150 phù hợp với tiếng Việt hơn
    engine.setProperty("volume", max(0.0, min(1.0, volume)))

    if voice_hint:
        hint = voice_hint.lower()
        for voice in engine.getProperty("voices"):
            haystack = f"{voice.id} {voice.name}".lower()
            if hint in haystack:
                engine.setProperty("voice", voice.id)
                break

    return engine

def speak_text(text: str, rate: int = 150, volume: float = 1.0, voice_hint: str | None = "An") -> None:
    """Hàm này sẽ được gọi từ ai_core.py để đọc văn bản"""
    engine = build_engine(rate=rate, volume=volume, voice_hint=voice_hint)
    engine.say(text)
    engine.runAndWait()

def save_to_wav(text: str, output_file: Path, rate: int = 150, volume: float = 1.0, voice_hint: str | None = "An") -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    engine = build_engine(rate=rate, volume=volume, voice_hint=voice_hint)
    engine.save_to_file(text, str(output_file))
    engine.runAndWait()

# Giữ lại phần này nếu bạn muốn chạy riêng text_to_speech.py qua Terminal để test
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-to-Speech tool")
    parser.add_argument("--text", type=str, required=True, help="Văn bản cần đọc")
    args = parser.parse_args()
    
    speak_text(args.text)
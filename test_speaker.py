import pyttsx3

def list_available_voices():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    
    print(f"Tìm thấy {len(voices)} giọng đọc trên máy tính của bạn:\n")
    
    for index, voice in enumerate(voices):
        print(f"Giọng {index + 1}:")
        print(f" - Tên (Name): {voice.name}")
        print(f" - ID: {voice.id}")
        print("-" * 40)

if __name__ == "__main__":
    list_available_voices()
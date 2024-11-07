import queue
import threading
import json
import pyaudio
from vosk import Model, KaldiRecognizer
from rapidfuzz import fuzz, process

# Load the Vosk model
model = Model("./model/vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(model, 16000)

# Load reference sentences from script.txt
with open("script.txt", "r") as file:
    reference_sentences = [line.strip() for line in file.readlines()]

# Audio queue
audio_queue = queue.Queue()

# Audio capture thread
def audio_stream():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=4000,
    )
    stream.start_stream()

    while True:
        data = stream.read(4000, exception_on_overflow=False)
        audio_queue.put(data)

threading.Thread(target=audio_stream, daemon=True).start()

# Function to perform live transcription and fuzzy matching
def live_fuzzy_match():
    print("Listening for audio and performing fuzzy matching...")
    while True:
        data = audio_queue.get()
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            transcription = result.get("text", "")
            if transcription:
                # Find the closest match from reference sentences
                match, score = process.extractOne(
                    transcription, reference_sentences, scorer=fuzz.ratio
                )
                print(f"Transcription: {transcription}")
                print(f"Best Match: {match} (Score: {score})\n")
        else:
            partial_result = json.loads(recognizer.PartialResult())
            transcription = partial_result.get("partial", "")
            if transcription:
                print(f"Partial Transcription: {transcription}", end="\r")

if __name__ == "__main__":
    live_fuzzy_match()

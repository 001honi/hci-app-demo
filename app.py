from flask import Flask, render_template, Response, jsonify
import queue
import threading
import json
import pyaudio
from vosk import Model, KaldiRecognizer
from rapidfuzz import fuzz, process
from gtts import gTTS
import os
import time

import pygame
pygame.mixer.init()

from nltk.corpus import stopwords

# Load English stopwords
STOP_WORDS = set(stopwords.words('english'))

# Flask app initialization
app = Flask(__name__)

# Load Vosk model
model = Model("./model/vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(model, 16000)

# Load reference sentences from script.txt
with open("script.txt", "r") as file:
    reference_sentences = [line.strip().lower() for line in file.readlines()]

# Audio queue
audio_queue = queue.Queue()

# Shared state variables
current_sentence = ""
# mispronunciations = {}

# Ensure the 'sound' directory exists
os.makedirs("sound", exist_ok=True)

# Generate feedback audio
def generate_feedback_audio(word):
    audio_file = f"sound/{word}.mp3"
    if not os.path.exists(audio_file):
        tts = gTTS(text=word, lang='en')
        tts.save(audio_file)


def play_audio_file(audio_file):
    """
    Plays the given audio file using Pygame.
    """
    try:
        if os.path.exists(audio_file):
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)  # Keeps the thread alive while audio is playing
        else:
            print(f"Audio file not found: {audio_file}")
    except Exception as e:
        print(f"Error playing audio file {audio_file}: {e}")

def play_feedback(word):
    """
    Plays the audio feedback for a given word in a separate thread.
    """
    audio_file = f"sound/{word}.mp3"  # Ensure the file extension matches your files
    playback_thread = threading.Thread(target=play_audio_file, args=(audio_file,))
    playback_thread.start()


# # Play feedback audio with timeout
# def play_feedback(word):
#     audio_file = f"sound/{word}.mp3"
#     if os.path.exists(audio_file):
#         playback_thread = threading.Thread(target=playsound, args=(audio_file,))
#         playback_thread.start()
#         playback_thread.join(timeout=5)  # Timeout to prevent stuck threads

# def play_feedback(word):
#     audio_file = f"sound/{word}.mp3"
#     if os.path.exists(audio_file):
#         audio = AudioSegment.from_file(audio_file)
#         play(audio)

@app.route("/play_audio/<word>", methods=["POST"])
def play_audio_server_side(word):
    audio_file = f"sound/{word}.mp3"
    if os.path.exists(audio_file):
        threading.Thread(target=play_feedback, args=(audio_file,)).start()  # Play audio asynchronously
        return "Audio played", 200
    return "Audio file not found", 404


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
    while True:
        data = stream.read(4000, exception_on_overflow=False)
        audio_queue.put(data)

threading.Thread(target=audio_stream, daemon=True).start()

# Real-time processing thread
def generate_transcription_and_feedback():
    # global current_sentence, mispronunciations
    global current_sentence
    last_sent_data = None

    while True:
        data = audio_queue.get()
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            transcription = result.get("text", "").split()

            # Fuzzy match to find the closest reference sentence
            match, score, _ = process.extractOne(" ".join(transcription), reference_sentences, scorer=fuzz.ratio)
            if score >= 60:  # Threshold for fuzzy matching
                current_sentence = match
            else:
                current_sentence = ""
                # Send live transcription updates
                yield f"data: {json.dumps({'transcription': ' '.join(transcription)})}\n\n"
                return

            # Initialize colored_sentence with all words set to default
            current_words = current_sentence.split() if current_sentence else []
            colored_sentence = [{'word': word, 'status': 'default', 'ref-i': i, 'match-score': None, 'match-word': None, 'match-i': None} for i, word in enumerate(current_words)]

            # Remove stop words from both current_words and transcription
            filtered_current_words = [word for word in current_words if word.lower() not in STOP_WORDS]
            filtered_transcription = [word for word in transcription if word.lower() not in STOP_WORDS]

            # Update the status of matched words in colored_sentence
            for word in filtered_transcription:
                if word in filtered_current_words:
                    # Find the index of the matched word in current_words
                    index = current_words.index(word)
                    colored_sentence[index]['status'] = 'matched'
                    # Remove the word from filtered_current_words to prevent duplicate matches
                    filtered_current_words.remove(word)
                else:
                    match, score, _ = process.extractOne(" ".join(word), filtered_current_words, scorer=fuzz.ratio)
                    if score >= 10:
                        index = current_words.index(match)
                        colored_sentence[index]['status'] = 'mispronounced'
                        colored_sentence[index]['match-word'] = word

            mispronunciations = {}

            for word_data in colored_sentence:
                if word_data['status'] == 'mispronounced':
                    mispronunciations.setdefault(word_data['word'], []).append(word_data['match-word'])
                    generate_feedback_audio(word_data['word'])
                    play_feedback(word_data['word'])

            # Prepare data to send
            current_data = {
                'colored_sentence': colored_sentence,
                'mispronunciations': mispronunciations,
            }

            # Send live transcription updates
            yield f"data: {json.dumps({'transcription': ' '.join(transcription)})}\n\n"

            # Send updated sentence and mispronunciation data only if changed
            if current_data != last_sent_data:
                last_sent_data = current_data
                yield f"data: {json.dumps(current_data)}\n\n"
        else:
            # Handle partial results (live transcription)
            partial = json.loads(recognizer.PartialResult()).get("partial", "")
            yield f"data: {json.dumps({'transcription': partial})}\n\n"



# Flask routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcription")
def transcription():
    return Response(generate_transcription_and_feedback(), mimetype="text/event-stream")

@app.route("/script_sentences")
def script_sentences():
    return jsonify(reference_sentences)

if __name__ == "__main__":
    app.run(debug=True)

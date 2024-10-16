import os
import subprocess
from deepspeech import Model
import numpy as np
import wave
from transformers import pipeline

# Extract audio from video
def extract_audio(video_path, audio_path):
    command = f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 44100 -vn {audio_path}"
    subprocess.call(command, shell=True)

# Transcribe audio using DeepSpeech
def transcribe_audio(audio_path, model_path):
    model = Model(model_path)
    
    with wave.open(audio_path, 'rb') as w:
        rate = w.getframerate()
        frames = w.getnframes()
        buffer = w.readframes(frames)

    data16 = np.frombuffer(buffer, dtype=np.int16)
    text = model.stt(data16)
    return text

# Improve transcription using a language model
def improve_transcription(text):
    summarizer = pipeline("summarization")
    improved_text = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return improved_text[0]['summary_text']

# Main process
video_path = "static/sample.mp4"
audio_path = "static/sample.wav"
deepspeech_model_path = "path/to/deepspeech/model.pbmm"

# Extract audio
extract_audio(video_path, audio_path)

# Transcribe audio
transcription = transcribe_audio(audio_path, deepspeech_model_path)

# Improve transcription
improved_transcription = improve_transcription(transcription)

print(improved_transcription)
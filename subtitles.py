import whisper
import subprocess
from pathlib import Path
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize


# Set custom NLTK data path
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download the necessary NLTK data to the custom path
nltk.download("punkt", download_dir=nltk_data_path, quiet=True)
nltk.download("punkt_tab", download_dir=nltk_data_path, quiet=True)


# Extract audio from video
def extract_audio(video_path, audio_path):
    command = f"ffmpeg -i {video_path} -ab 160k -ac 1 -ar 16000 -vn {audio_path}"
    subprocess.call(command, shell=True)

# Transcribe audio using Whisper
def transcribe_audio(audio_path):
    try:
        # Load the model (using 'base' for faster processing, can use 'small' or 'medium' for better accuracy)
        model = whisper.load_model("base")
        
        # Transcribe the audio
        result = model.transcribe(audio_path)
        
        return result["text"]
    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        return f"Transcription failed: {str(e)}"

# Improve transcription using a language model
def improve_transcription(text):
    """Improve transcription using a language model"""
    try:
        summarizer = pipeline("summarization")
        
        # Split text into chunks if it's too long
        max_chunk_length = 500
        chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        
        improved_chunks = []
        for chunk in chunks:
            if len(chunk.strip()) > 10:  # Only process non-empty chunks
                improved_chunk = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
                improved_chunks.append(improved_chunk[0]['summary_text'])
        
        return " ".join(improved_chunks)
    except Exception as e:
        print(f"Error in improving transcription: {str(e)}")
        return text

# Main process
video_path = "static/sample.mp4"
audio_path = "static/sample.wav"

# Extract audio
extract_audio(video_path, audio_path)

# Transcribe audio
transcription = transcribe_audio(audio_path)

# Improve transcription (optional)
improved_transcription = improve_transcription(transcription)

print("Subtitles saved to:", subtitle_output)
print("Improved transcription:", improved_transcription)

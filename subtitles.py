import os
import subprocess
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
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


# New function to transcribe audio using Wav2Vec2
def transcribe_audio(audio_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    audio, rate = librosa.load(audio_path, sr=16000)

    input_values = processor(
        audio, sampling_rate=rate, return_tensors="pt"
    ).input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    # Split transcription into chunks of approximately 10 seconds each
    total_duration = len(audio) / rate
    chunk_duration = 10  # seconds
    num_chunks = int(total_duration / chunk_duration) + 1

    sentence_timestamps = []
    for i in range(num_chunks):
        start_time = i * chunk_duration
        end_time = min((i + 1) * chunk_duration, total_duration)

        # Calculate the corresponding text chunk
        start_char = int(start_time / total_duration * len(transcription))
        end_char = int(end_time / total_duration * len(transcription))
        chunk = transcription[start_char:end_char]

        # Try to split the chunk into sentences
        sentences = sent_tokenize(chunk)
        if not sentences:
            sentences = [chunk]  # If no sentences detected, use the whole chunk

        # Distribute time evenly among sentences in the chunk
        time_per_sentence = (end_time - start_time) / len(sentences)
        for j, sentence in enumerate(sentences):
            sentence_start = start_time + j * time_per_sentence
            sentence_end = sentence_start + time_per_sentence
            sentence_timestamps.append(
                {
                    "sentence": sentence.strip(),
                    "start_offset": sentence_start,
                    "end_offset": sentence_end,
                }
            )

    print(f"Number of sentences detected: {len(sentence_timestamps)}")
    print(f"First few sentences: {sentence_timestamps[:3]}")

    return transcription, sentence_timestamps


# Improve transcription using a language model
def improve_transcription(text):
    summarizer = pipeline(
        "summarization", device=0 if torch.cuda.is_available() else -1
    )
    improved_text = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return improved_text[0]["summary_text"]


def format_and_save_subtitles(transcription, sentence_timestamps, output_file):
    print(f"Formatting {len(sentence_timestamps)} sentences")
    with open(output_file, "w", encoding="utf-8") as f:
        for i, sentence_info in enumerate(sentence_timestamps):
            sentence = sentence_info["sentence"]
            start_time = sentence_info["start_offset"]
            end_time = sentence_info["end_offset"]

            # Format timestamps as HH:MM:SS,mmm
            start_formatted = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{start_time%60:06.3f}".replace(
                ".", ","
            )
            end_formatted = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{end_time%60:06.3f}".replace(
                ".", ","
            )

            f.write(f"{i+1}\n")
            f.write(f"{start_formatted} --> {end_formatted}\n")
            f.write(f"{sentence}\n\n")

            if i < 3:  # Print first 3 entries for debugging
                print(
                    f"Entry {i+1}: {start_formatted} --> {end_formatted} : {sentence}"
                )


# Main process
video_path = "static/sample.mp4"
audio_path = "static/sample.wav"
subtitle_output = "static/subtitles.srt"

# Extract audio
extract_audio(video_path, audio_path)

# Transcribe audio using Wav2Vec2 (GPU if available)
transcription, word_timestamps = transcribe_audio(audio_path)

# Format and save subtitles
format_and_save_subtitles(transcription, word_timestamps, subtitle_output)

# Improve transcription (optional)
improved_transcription = improve_transcription(transcription)

print("Subtitles saved to:", subtitle_output)
print("Improved transcription:", improved_transcription)

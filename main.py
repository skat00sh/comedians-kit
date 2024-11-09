from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
from subtitles import extract_audio, transcribe_audio, improve_transcription

app = FastAPI()

# Get the absolute path to the static directory
static_dir = Path(__file__).parent / "static"

# Mount the static directory
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Process video and get transcription
    video_path = "static/sample.mp4"
    audio_path = "static/sample.wav"

    # Extract audio if it doesn't exist
    if not Path(audio_path).exists():
        extract_audio(video_path, audio_path)

    # Get transcription
    try:
        transcription = transcribe_audio(audio_path)
        print("Raw transcription:", transcription)  # Debug print
        improved_transcription = improve_transcription(transcription)
        print("Improved transcription:", improved_transcription)  # Debug print
    except Exception as e:
        improved_transcription = f"Transcription not available: {str(e)}"
        print("Transcription error:", str(e))  # Debug print

    # Define bookmarks with proper structure
    bookmarks = [
        {"time": 10, "label": "Big laugh here", "color": "#FF0000"},
        {"time": 25, "label": "Smirks no laughs", "color": "#FFFF00"},
        {"time": 40, "label": "Okayish laughs", "color": "#FFA500"},
    ]

    print("Bookmarks being passed:", bookmarks)  # Debug print

    return templates.TemplateResponse(
        "video_player.html", 
        {
            "request": request, 
            "bookmarks": bookmarks,
            "transcript": improved_transcription,
            "raw_transcript": transcription
        }
    )

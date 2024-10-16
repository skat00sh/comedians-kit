from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/")
async def read_root(request: Request):
    bookmarks = [
        {"time": 10, "label": "Big laugh here", "color": "#FF0000"},  # Red
        {"time": 25, "label": "Smirks no laughs", "color": "#FFFF00"},  # Yellow
        {"time": 40, "label": "Okayish laughs", "color": "#FFA500"},  # Orange
    ]
    return templates.TemplateResponse(
        "video_player.html", {"request": request, "bookmarks": bookmarks}
    )

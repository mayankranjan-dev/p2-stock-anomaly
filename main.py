from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from database import init_db
from routers import analysis

app = FastAPI(title="Stock Anomaly Detection API")

templates = Jinja2Templates(directory="templates")

# create the sqlite tables on startup, cheap enough to just run every time
init_db()

app.include_router(analysis.router)


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

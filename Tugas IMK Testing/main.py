from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from proses import proses

app = FastAPI(title="Klasifikasi Penyakit Kanker")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def welcome():
    return {"message": "Model Klasifikasi Penyakit Kanker"}

@app.post("/prediksi")
async def predict_image(file: UploadFile = File(...)):
    conf, label = proses(file)
    if conf is None:
        return {"error": label}
    return {"label": label, "confidence": f"{conf * 100:.2f}%"}

@app.get("/ui")
async def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
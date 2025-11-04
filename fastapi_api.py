from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from PIL import Image
from use_trained_model import classify_image
from pydantic import BaseModel
import pyphen

app = FastAPI(title="Image Classification API", description="Upload an image and get its classification.", version="1.0")

# CORS middleware (optional, add if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20 MB


def allowed_file(filename: str) -> bool:
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.exception_handler(413)
async def request_entity_too_large_handler(request, exc):
    return JSONResponse(
        status_code=413,
        content={"error": "Arquivo excede o limite de 20 MB"}
    )


@app.post("/classificacao", response_model=Dict[str, str], status_code=200)
async def classify_upload_image(file: UploadFile = File(...)):
    """
    Receives an image file and returns its classification.

    - **file**: image file to classify (max size 20MB)
    - Returns JSON with category and classification details.
    """
    # Check extension
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Tipo de arquivo não permitido")
    
    # Check file size
    contents = await file.read()
    if len(contents) > MAX_CONTENT_LENGTH:
        raise HTTPException(status_code=413, detail="Arquivo excede o limite de 20 MB")
    
    # Verify if it is an image
    try:
        img = Image.open(file.file)
        img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="O arquivo enviado não é uma imagem válida")
    
    # Reset file pointer to beginning after reading
    file.file.seek(0)
    
    # Classify image (assumed classify_image accepts UploadFile or file-like object)
    try:
        classification_result, prediction = classify_image(file.file)
        return {
            "category": prediction,
            "details": classification_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

dic = pyphen.Pyphen(lang='pt_BR')

class Palavra(BaseModel):
    palavra: str

@app.post("/silabas")
def separar_silabas(item: Palavra):
    if not item.palavra:
        raise HTTPException(status_code=400, detail="Campo 'palavra' é obrigatório")

    silabizada = dic.inserted(item.palavra)
    return {"silaba": silabizada}

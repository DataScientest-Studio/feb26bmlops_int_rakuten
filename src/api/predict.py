from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from PIL import Image
import io

from .schemas import PredictResponse
from src.models.classifier import classifier_service

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictResponse)
async def predict_single(
    file: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=27),
):
    """
    Run inference on a single uploaded image.
    Returns top-k class predictions with confidence scores.
    """
    if not classifier_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        result = classifier_service.predict_single(img, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return PredictResponse(**result)


@router.post("/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Run inference on multiple uploaded images.
    Saves them to a temp location, runs batch inference, cleans up.
    """
    if not classifier_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    import tempfile, os

    tmp_paths = []
    try:
        # Save uploads to temp files
        for upload in files:
            contents = await upload.read()
            suffix = os.path.splitext(upload.filename)[-1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(contents)
                tmp_paths.append(tmp.name)

        results = classifier_service.predict_batch(tmp_paths)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference error: {e}")
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except Exception:
                pass

    # Attach filenames to results
    for i, res in enumerate(results):
        res["filename"] = files[i].filename

    return JSONResponse(content={"predictions": results})

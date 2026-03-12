from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from hand_detector import analyze_rps_from_bytes
from pose_detector import analyze_dbdbd_from_bytes

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/detect/rps")
async def detect_rps(file: UploadFile = File(...)):
    try:
        data = await file.read()
        result = analyze_rps_from_bytes(data)
        print("RPS 판정:", result)
        return result
    except Exception as e:
        print("RPS 에러:", e)
        return {
            "success": False,
            "hand_detected": False,
            "label": "unknown",
            "message": str(e)
        }

@app.post("/detect/dbdbd")
async def detect_dbdbd(file: UploadFile = File(...)):
    try:
        data = await file.read()
        result = analyze_dbdbd_from_bytes(data)
        print("DBDBD 판정:", result)
        return result
    except Exception as e:
        print("DBDBD 에러:", e)
        return {
            "success": False,
            "pose_detected": False,
            "label": "unknown",
            "message": str(e)
        }
import os
import sys
import subprocess

# Configure LD_LIBRARY_PATH dynamically to include nvidia packages paths
try:
    import nvidia.cublas
    import nvidia.cudnn
    cublas_path = os.path.join(os.path.dirname(nvidia.cublas.__file__), "lib")
    cudnn_path = os.path.join(os.path.dirname(nvidia.cudnn.__file__), "lib")
    
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    new_ld = f"{cublas_path}:{cudnn_path}"
    if current_ld:
        new_ld = f"{new_ld}:{current_ld}"
    
    os.environ["LD_LIBRARY_PATH"] = new_ld
    print(f"Dynamically set LD_LIBRARY_PATH to: {new_ld}")
except Exception as e:
    print(f"Warning: Could not configure LD_LIBRARY_PATH dynamically: {e}")

import uuid
import shutil
import time
from pathlib import Path
from contextlib import asynccontextmanager
import json
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import torch
from faster_whisper import WhisperModel, BatchedInferencePipeline
from pyannote.audio import Pipeline

# Setup upload and output directories inside the workspace or /tmp
# Using workspace for persistence if needed, but temp is fine for upload cache
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Device detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3-turbo")

print(f"--- Starting Whisper Service ---")
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_NAME}")

asr = None
diarization_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr, diarization_pipeline
    # Clean up any leftover files in the uploads directory on startup
    try:
        if UPLOAD_DIR.exists():
            for file_path in UPLOAD_DIR.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
            print("Leftover upload files cleaned up on startup.")
    except Exception as e:
        print(f"Warning: Failed to clean up leftover uploads: {e}")
        
    try:
        print("Loading faster-whisper model...")
        compute_type = "float16" if DEVICE == "cuda" else "float32"
        
        base_asr = WhisperModel(
            MODEL_NAME,
            device=DEVICE,
            compute_type=compute_type
        )
        asr = BatchedInferencePipeline(model=base_asr)
        print("Model loaded successfully with BatchedInferencePipeline!")
    except Exception as e:
        print(f"CRITICAL: Failed to load Whisper model: {e}")
        
    try:
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            print("Loading pyannote speaker diarization pipeline...")
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token
            )
            if diarization_pipeline:
                diarization_pipeline.to(torch.device(DEVICE))
                print("Diarization pipeline loaded successfully!")
        else:
            print("Warning: HF_TOKEN env variable not set. Speaker diarization will be disabled.")
    except Exception as e:
        print(f"Warning: Failed to load Diarization model: {e}")
        
    yield
    if asr:
        del asr
    if diarization_pipeline:
        del diarization_pipeline

app = FastAPI(title="Whisper Transcription Service", lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

def preprocess_audio(input_path: Path) -> Path:
    """Converts any video or audio file to 16kHz mono WAV for diarization & transcription."""
    output_path = input_path.parent / f"{input_path.stem}_processed.wav"
    command = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_path)
    ]
    # Run ffmpeg silently
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        raise ValueError("Не удалось обработать аудиофайл с помощью ffmpeg.")
    return output_path


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    diarize: bool = Form(True),
    num_speakers: Optional[int] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None)
):
    if not asr:
        raise HTTPException(status_code=500, detail="Модель Whisper еще не загружена или произошла ошибка инициализации.")
    
    file_ext = Path(file.filename).suffix.lower()
    unique_id = uuid.uuid4().hex
    temp_file = UPLOAD_DIR / f"{unique_id}{file_ext}"
    
    # Save the uploaded file synchronously BEFORE returning StreamingResponse.
    # Otherwise, FastAPI will close the file upload socket/object once the endpoint function returns.
    try:
        with temp_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении файла: {str(e)}")
        
    async def event_generator():
        processing_start = time.time()
        processed_path = None
        
        try:
            # 1. Preprocess file to 16kHz mono WAV (crucial for pyannote and Whisper)
            yield json.dumps({"status": "processing", "stage": "preprocess", "message": "Подготовка аудио (конвертация в 16кГц)..."}) + "\n"
            try:
                processed_path = preprocess_audio(temp_file)
            except Exception as e:
                print(f"Warning: Audio preprocessing failed: {e}. Falling back to original file.")
                processed_path = temp_file
                
            # Run speaker diarization if pipeline is loaded and requested
            speaker_segments = []
            if diarize and diarization_pipeline:
                yield json.dumps({"status": "processing", "stage": "diarize", "message": "Определение спикеров (диаризация)..."}) + "\n"
                try:
                    diarize_kwargs = {}
                    if num_speakers is not None:
                        diarize_kwargs["num_speakers"] = num_speakers
                    if min_speakers is not None:
                        diarize_kwargs["min_speakers"] = min_speakers
                    if max_speakers is not None:
                        diarize_kwargs["max_speakers"] = max_speakers

                    diarization = diarization_pipeline(str(processed_path), **diarize_kwargs)
                    # Handle pyannote.audio v3.x vs v4.x output structure
                    annotation = diarization.speaker_diarization if hasattr(diarization, "speaker_diarization") else diarization
                    
                    for turn, _, speaker in annotation.itertracks(yield_label=True):
                        speaker_segments.append({
                            "start": turn.start,
                            "end": turn.end,
                            "speaker": speaker
                        })
                    print(f"Diarization finished. Found {len(speaker_segments)} segments.")
                except Exception as e:
                    print(f"Warning: Diarization failed: {e}")
                    yield json.dumps({"status": "processing", "stage": "diarize_failed", "message": f"Ошибка диаризации ({e}), продолжаем без нее..."}) + "\n"
                    
            # 3. Transcribe
            yield json.dumps({"status": "processing", "stage": "transcribe_init", "message": "Инициализация распознавания..."}) + "\n"
            try:
                batch_size = int(os.getenv("WHISPER_BATCH_SIZE", 8))
            except ValueError:
                batch_size = 8
            
            segments, info = asr.transcribe(
                str(processed_path),
                language="ru",
                beam_size=1,
                batch_size=batch_size
            )
            
            text_segments = []
            for segment in segments:
                # Align speaker with segment
                if speaker_segments:
                    best_speaker = "Неизвестный Спикер"
                    max_overlap = 0.0
                    for spk_seg in speaker_segments:
                        overlap_start = max(segment.start, spk_seg["start"])
                        overlap_end = min(segment.end, spk_seg["end"])
                        overlap = overlap_end - overlap_start
                        if overlap > max_overlap:
                            max_overlap = overlap
                            best_speaker = spk_seg["speaker"]
                            
                    if best_speaker.startswith("SPEAKER_"):
                        try:
                            spk_id = int(best_speaker.split("_")[1]) + 1
                            best_speaker = f"Спикер {spk_id}"
                        except:
                            pass
                            
                    mins_s = int(segment.start // 60)
                    secs_s = int(segment.start % 60)
                    timestamp_str = f"[{mins_s:02d}:{secs_s:02d}]"
                    line_text = f"{timestamp_str} {best_speaker}: {segment.text.strip()}"
                else:
                    mins_s = int(segment.start // 60)
                    secs_s = int(segment.start % 60)
                    timestamp_str = f"[{mins_s:02d}:{secs_s:02d}]"
                    line_text = f"{timestamp_str} {segment.text.strip()}"
                    
                text_segments.append(line_text)
                # Yield current segment to show real-time progress
                yield json.dumps({
                    "status": "processing",
                    "stage": "transcribe",
                    "message": f"Распознано: \"{segment.text.strip()}\""
                }) + "\n"
                
            text = "\n\n".join(text_segments)
            
            # Clean up files
            if temp_file.exists():
                temp_file.unlink()
            if processed_path and processed_path.exists() and processed_path != temp_file:
                processed_path.unlink()
                
            duration = time.time() - processing_start
            print(f"Finished transcription in {duration:.2f} seconds.")
            
            yield json.dumps({
                "status": "completed",
                "stage": "done",
                "text": text,
                "duration_seconds": round(duration, 2),
                "filename": Path(file.filename).stem
            }) + "\n"
            
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            # Clean up files in case of error
            if temp_file.exists():
                temp_file.unlink()
            if processed_path and processed_path.exists() and processed_path != temp_file:
                processed_path.unlink()
            yield json.dumps({
                "status": "error",
                "stage": "error",
                "message": f"Ошибка транскрибации: {str(e)}"
            }) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

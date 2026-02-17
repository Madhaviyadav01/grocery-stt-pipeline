from fastapi import FastAPI, UploadFile, File
import shutil
import os

from pipeline.phase_A_Audio_preprocessing.run_phase_A import run_phase_A
from pipeline.phase_B_transcription.run_phase_B import run_phase_B
from pipeline.phase_C_structured_boundary_extraction.run_phase_C import run_phase_C
from pipeline.phase_D_fuzzy_canonical_mapping.run_phase_D import run_phase_D
import pandas as pd

app = FastAPI(title="Grocery AI STT API")

@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):

    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Sequential Execution
        # Phase A: Preprocessing -> Returns path to cleaned audio
        cleaned_audio_path = run_phase_A(temp_path)
        
        # Phase B: Transcription -> Returns path to transcript text
        transcript_path = run_phase_B(cleaned_audio_path)
        
        # Phase C: Structure Extraction -> Returns path to structured JSON
        structured_json_path = run_phase_C(transcript_path)
        
        # Phase D: Mapping -> Returns path to final CSV
        final_csv_path = run_phase_D(structured_json_path)

        # Read result and return as JSON
        if os.path.exists(final_csv_path):
            df = pd.read_csv(final_csv_path)
            result = df.to_dict(orient="records")
        else:
            result = {"error": "Pipeline completed but no output file found."}
            
        return {"status": "success", "mapped_items": result}

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        # Cleanup uploaded file
        if os.path.exists(temp_path):
            os.remove(temp_path)

import os
import shutil
import subprocess
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import torch
import librosa
import textgrid
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# --- Configuration ---
ACOUSTIC_MODEL_PATH = "C:/Users/Vini/Desktop/Voxia/pronunciation-tester/model/portuguese_mfa.zip"
DICTIONARY_PATH = "C:/Users/Vini/Desktop/Voxia/pronunciation-tester/model/portuguese_mfa.dict"

# --- DNN Model Configuration ---
SCORING_MODEL_NAME = "caiocrocha/wav2vec2-large-xlsr-53-phoneme-portuguese"

processor = None
model = None
device = None

# --- FastAPI Application ---
app = FastAPI(
    title="Pronunciation Scoring Service",
    description="A service to perform forced alignment (MFA) and Goodness of Pronunciation (GOP) scoring.",
)

@app.on_event("startup")
async def startup_event():
    """
    Loads the DNN model and processor into memory when the server starts.
    """
    global processor, model, device
    print("Loading DNN model for scoring...")
    
    # Check if a GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the processor and the model
    processor = Wav2Vec2Processor.from_pretrained(SCORING_MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(SCORING_MODEL_NAME)
    
    # Move the model to the GPU (if available) for much faster inference
    model.to(device) # type: ignore
    print("DNN model loaded successfully!")

def cleanup_dir(path: str):
    """Safely removes a directory and all its contents."""
    print(f"Cleaning up temporary directory: {path}")
    shutil.rmtree(path, ignore_errors=True)

@app.post("/score/", summary="Align audio and calculate GOP scores")
async def align_and_score_pronunciation(
    background_tasks: BackgroundTasks,
    text: str = Form(..., description="The sentence spoken in the audio."),
    audio_file: UploadFile = File(..., description="The user's audio recording (WAV/etc).")
):
    temp_dir = tempfile.mkdtemp()
    background_tasks.add_task(cleanup_dir, temp_dir)

    try:
        # --- Stage 1: Prepare files and Run MFA Alignment ---
        corpus_dir = os.path.join(temp_dir, "mfa_input")
        os.makedirs(corpus_dir)
        base_name = "user_utterance"
        
        audio_path = os.path.join(corpus_dir, f"{base_name}.wav")
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
            
        text_path = os.path.join(corpus_dir, f"{base_name}.lab")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)

        output_dir = os.path.join(temp_dir, "mfa_output")
        mfa_command = ["mfa", "align", corpus_dir, DICTIONARY_PATH, ACOUSTIC_MODEL_PATH, output_dir, "--clean", "--overwrite"]
        
        print(f"Running MFA command: {' '.join(mfa_command)}")
        try:
            subprocess.run(mfa_command, check=True, capture_output=True, text=True, encoding="utf-8")
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"MFA alignment failed. Error: {e.stderr}")

        result_grid_path = os.path.join(output_dir, f"{base_name}.TextGrid")
        if not os.path.exists(result_grid_path):
            raise HTTPException(status_code=404, detail="MFA finished, but the output TextGrid was not found.")

        # --- Stage 2: Parse TextGrid and Score with DNN ---
        print("MFA alignment successful. Starting GOP scoring...")

        audio_input, sample_rate = librosa.load(audio_path, sr=16000)

        tg = textgrid.TextGrid.fromFile(result_grid_path)
        phone_tier = tg.getFirst("phones")

        scores = []
        for interval in phone_tier: # type: ignore
            phoneme = interval.mark
            if not phoneme or phoneme.lower() in ["sil", "spn", ""]:
                continue

            start_time = interval.minTime
            end_time = interval.maxTime

            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            audio_slice = audio_input[start_sample:end_sample]

            if len(audio_slice) == 0:
                continue

            inputs = processor(audio_slice, sampling_rate=16000, return_tensors="pt", padding=True) # type: ignore
            
            input_values = inputs.input_values.to(device)
            
            with torch.no_grad():
                logits = model(input_values).logits # type: ignore

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            target_phoneme_id = processor.tokenizer.convert_tokens_to_ids(phoneme) # type: ignore

            # Calculate the average log-probability for the TARGET phoneme across the audio slice
            # This value is our GOP score. Closer to 0 is better.
            avg_log_prob = log_probs[0, :, target_phoneme_id].mean().item()

            scores.append({
                "phoneme": phoneme,
                "start": round(start_time, 4),
                "end": round(end_time, 4),
                "score": round(avg_log_prob, 4)
            })

        print("Scoring complete.")
        return JSONResponse(content={"scores": scores})

    except HTTPException as he:
        raise he

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
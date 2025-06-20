import os
import json
import time
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum

import aiofiles
import httpx
import PyPDF2
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
from pydantic_settings import BaseSettings
from pydantic import BaseModel
import io

# Configure structured logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_dir / "app.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# Configuration class
class Settings(BaseSettings):
    upload_dir: str = "uploads"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: List[str] = [".pdf", ".txt", ".csv"]
    ollama_model: str = "llama3:8b"  # Using Llama 3 as default model
    ollama_url: str = "http://localhost:11434"  # Default Ollama URL

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()

# Pydantic models for request/response
class QueryRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    response: str

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to AutoTrainerX API"}

# Health check endpoint for Ollama
@app.get("/health/ollama")
async def check_ollama_health():
    """Check if Ollama is running and accessible"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.ollama_url}/api/tags")
            if response.status_code == 200:
                return {"status": "healthy", "ollama_url": settings.ollama_url}
            else:
                return {"status": "unhealthy", "error": f"Status code: {response.status_code}"}
    except Exception as e:
        logger.error(f"Ollama health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

# Ensure upload directory exists
Path(settings.upload_dir).mkdir(exist_ok=True)

async def extract_text_from_pdf(pdf_path: str) -> str:
    """ Extracts text from a PDF asynchronously using streaming."""
    text = ""
    try:
        async with aiofiles.open(pdf_path, "rb") as file:
            pdf_content = await file.read()
            pdf_file = io.BytesIO(pdf_content)
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        logger.info(f"Extracted text from PDF: {pdf_path}")
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise HTTPException(status_code=500, detail="Error extracting text from PDF")
    return text

async def load_text_file(text_path: str) -> str:
    """ Loads text from a file asynchronously. """
    try:
        async with aiofiles.open(text_path, "r", encoding="utf-8") as file:
            text = await file.read()
        logger.info(f"Loaded text from file: {text_path}")
    except Exception as e:
        logger.error(f"Error loading text file: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading text file")
    return text

async def load_csv_data(csv_path: str) -> List[Dict[str, Any]]:
    """ Loads CSV data asynchronously without blocking. """
    try:
        async with aiofiles.open(csv_path, "r", encoding="utf-8") as file:
            content = await file.read()
        import io
        df = pd.read_csv(io.StringIO(content))
        data = df.to_dict(orient="records")
        logger.info(f"Loaded CSV data from file: {csv_path}")
    except Exception as e:
        logger.error(f"Error loading CSV data: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading CSV data")
    return data

async def fetch_with_retries(api_call, retries=3, delay=2):
    """ Wrapper for API calls with exponential backoff. """
    for attempt in range(retries):
        try:
            return await api_call()
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(delay * (2 ** attempt))
            else:
                raise HTTPException(status_code=500, detail=f"API error: {str(e)}")

async def analyze_content(text: str) -> Tuple[str, float, Optional[str]]:
    """ Uses Ollama to analyze content type asynchronously. """
    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            response = await client.post(
                f"{settings.ollama_url}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": f"Classify the following text into valid_conversation, technical_documentation, nonsense, or irrelevant. Respond in JSON format with category, confidence (0-1), and explanation.\n\nText: {text[:1000]}",
                    "stream": False  # Use non-stream for classification
                }
            )
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Ollama API returned status {response.status_code}: {response.text}")

            result = response.json()
            logger.debug(f"Ollama analyze_content response: {result}")
            # Try to parse the response as JSON
            try:
                parsed_classification = json.loads(result.get("response", "{}"))
                category = parsed_classification.get("category", "unknown")
                confidence = float(parsed_classification.get("confidence", 0.0))
                explanation = parsed_classification.get("explanation")
            except Exception:
                category, confidence, explanation = "unknown", 0.0, None

            if category == "unknown":
                logger.warning(f"Could not extract classification from Ollama response: {result}")
                raise HTTPException(status_code=500, detail="Ollama API did not return a valid classification.")

            return category, confidence, explanation

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Ollama API error: {str(e)}")

async def validate_file(file: UploadFile):
    """ Validates file size and format asynchronously. """
    await file.seek(0, os.SEEK_END)
    size = await file.tell()
    await file.seek(0)
    if size > settings.max_file_size:
        raise HTTPException(status_code=400, detail="File too large")
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file format")

async def process_file(file: UploadFile) -> List[Dict[str, Any]]:
    """ Processes a file asynchronously, returning fine-tuning data. """
    await validate_file(file)
    file_path = Path(settings.upload_dir) / file.filename

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(await file.read())

    if file_path.suffix == ".pdf":
        extracted_text = await extract_text_from_pdf(str(file_path))
    elif file_path.suffix == ".txt":
        extracted_text = await load_text_file(str(file_path))
    elif file_path.suffix == ".csv":
        extracted_text = json.dumps(await load_csv_data(str(file_path)))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    category, confidence, explanation = await analyze_content(extracted_text)
    if category in ["nonsense", "irrelevant"]:
        raise HTTPException(status_code=400, detail=f"Content rejected: {explanation}")

    return [{"messages": [{"role": "user", "content": extracted_text}]}]

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    """ Handles multiple file uploads asynchronously. """
    results = []
    tasks = [process_file(file) for file in files]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    for file, result in zip(files, responses):
        if isinstance(result, Exception):
            logger.error(f"File {file.filename} failed: {result}")
            continue
        results.extend(result)

    async with aiofiles.open("finetune_data.jsonl", "w") as f:
        for entry in results:
            await f.write(json.dumps(entry) + "\n")

    return {"message": "Files processed successfully", "data_count": len(results)}

@app.get("/files/")
async def list_files():
    """ Lists all processed files in the upload directory. """
    try:
        files = []
        upload_dir = Path(settings.upload_dir)
        if upload_dir.exists():
            for file_path in upload_dir.glob("*.*"):
                if file_path.suffix.lower() in settings.allowed_extensions:
                    files.append({
                        "filename": file_path.name,
                        "size": file_path.stat().st_size,
                        "last_modified": file_path.stat().st_mtime
                    })
        return files
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail="Error listing files")

@app.post("/query/", response_model=QueryResponse)
async def query_model(request: QueryRequest):
    """ Queries the Ollama model with a prompt. """
    try:
        user_prompt = request.prompt.strip()
        
        # Validate input
        if not user_prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        # Limit prompt length to prevent hanging
        if len(user_prompt) > 4000:
            user_prompt = user_prompt[:4000]
            logger.warning("Prompt truncated to 4000 characters")
        
        # Check if the prompt is a general question about 'Constitution' and not already specific to India
        if "constitution" in user_prompt.lower() and "india" not in user_prompt.lower():
            user_prompt += " of India"
            logger.info(f"Modified prompt to: {user_prompt}")

        logger.info(f"Querying Ollama with prompt: {user_prompt[:100]}...")
        
        # Create a timeout wrapper
        async def make_ollama_request():
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
                logger.debug(f"Making request to: {settings.ollama_url}/api/generate")
                
                # First verify the model exists
                try:
                    model_check = await client.post(
                        f"{settings.ollama_url}/api/show",
                        json={"name": settings.ollama_model},
                        timeout=10.0
                    )
                    if model_check.status_code != 200:
                        raise HTTPException(status_code=404, detail=f"Model '{settings.ollama_model}' not found or not loaded")
                except httpx.TimeoutException:
                    logger.warning("Model check timed out, proceeding anyway")
                
                # Make the generation request
                response = await client.post(
                    f"{settings.ollama_url}/api/generate",
                    json={
                        "model": settings.ollama_model,
                        "prompt": user_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "num_predict": 512  # Limit response length
                        }
                    }
                )
                
                logger.debug(f"Ollama response status: {response.status_code}")
                
                if response.status_code != 200:
                    error_text = response.text
                    logger.error(f"Ollama API error - Status: {response.status_code}, Response: {error_text}")
                    
                    if response.status_code == 404:
                        raise HTTPException(status_code=404, detail=f"Model '{settings.ollama_model}' not found")
                    elif response.status_code == 400:
                        raise HTTPException(status_code=400, detail=f"Bad request: {error_text}")
                    else:
                        raise HTTPException(status_code=500, detail=f"Ollama error {response.status_code}: {error_text}")

                return response

        # Use asyncio.wait_for to add an overall timeout
        try:
            response = await asyncio.wait_for(make_ollama_request(), timeout=90.0)
        except asyncio.TimeoutError:
            logger.error("Ollama request timed out after 90 seconds")
            raise HTTPException(status_code=504, detail="Request timed out. The model may be too slow or stuck.")

        # Parse response
        try:
            result = response.json()
            logger.debug(f"Ollama response keys: {result.keys()}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Raw response: {response.text[:500]}")
            raise HTTPException(status_code=500, detail="Invalid JSON response from Ollama")

        # Extract response content
        if 'response' not in result:
            logger.error(f"Missing 'response' field. Available fields: {list(result.keys())}")
            raise HTTPException(status_code=500, detail="Ollama response missing 'response' field")

        response_content = result['response'].strip()
        
        if not response_content:
            logger.warning("Ollama returned empty response")
            # Check if there's an error in the response
            if 'error' in result:
                raise HTTPException(status_code=500, detail=f"Ollama error: {result['error']}")
            else:
                raise HTTPException(status_code=500, detail="Ollama returned empty response")

        logger.info(f"Successfully got response from Ollama (length: {len(response_content)})")
        return QueryResponse(response=response_content)

    except HTTPException:
        raise
    except asyncio.TimeoutError:
        logger.error("Overall request timeout")
        raise HTTPException(status_code=504, detail="Request timed out")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
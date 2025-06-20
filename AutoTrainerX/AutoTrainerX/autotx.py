from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi_jwt_auth import AuthJWT
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession
from google.cloud import storage
import os
import asyncio
import aiofiles

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)

GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")

def upload_to_gcp_bucket(file_path: str, bucket_name: str, destination_blob_name: str) -> str:
    """Uploads a file to the GCP bucket and returns the public URL."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    blob.make_public()
    return blob.public_url

async def process_file(file: UploadFile) -> dict:
    """Process and upload a file to GCP bucket."""
    file_location = f"temp_{file.filename}"
    async with aiofiles.open(file_location, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)
    gcp_url = upload_to_gcp_bucket(file_location, GCP_BUCKET_NAME, file.filename)
    os.remove(file_location)
    return {"filename": file.filename, "gcp_url": gcp_url}

@app.post("/upload/")
async def upload_files(files: list[UploadFile] = File(...), db: AsyncSession = Depends(get_db)):
    """Handles file uploads with JWT authentication and cloud storage."""
    tasks = [process_file(file) for file in files]
    file_results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in file_results:
        if isinstance(result, dict):
            db.add(UploadedFile(filename=result["filename"], gcp_url=result["gcp_url"]))

    await db.commit()
    return {"message": "Files uploaded and stored in GCP bucket."}

@app.post("/query/")
@limiter.limit("5/minute")
async def query_model(prompt: str, auth: AuthJWT = Depends()):
    """Query AI model with JWT authentication."""
    auth.jwt_required()
    model = select_model(prompt)

    async with httpx.AsyncClient() as client:
        response = await fetch_with_retries(lambda: client.post(
            "https://api.openai.com/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": prompt}]}
        ))

    return {"response": response.json()["choices"][0]["message"]["content"]}
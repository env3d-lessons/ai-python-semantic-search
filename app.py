from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os

app = FastAPI(title="Image Semantic Search")

# Serve the images folder at /images
app.mount("/images", StaticFiles(directory="images"), name="images")

# Serve index.html at root
@app.get("/", response_class=FileResponse)
async def root():
    index_path = "index.html"
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="index.html not found")

@app.get("/search")
async def search(query: str):
    # Placeholder for search functionality
    return JSONResponse(None)

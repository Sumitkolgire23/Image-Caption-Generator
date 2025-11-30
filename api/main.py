from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from src.inference import generate_caption

app = FastAPI()

@app.post('/caption')
async def caption_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    with open('tmp_upload.jpg', 'wb') as f:
        f.write(contents)
    caption = generate_caption('tmp_upload.jpg')
    return JSONResponse({'caption': caption})

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)

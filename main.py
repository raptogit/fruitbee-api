from fastapi import FastAPI, File 
from fastapi.middleware.cors import CORSMiddleware
from model import predict


app = FastAPI() 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload_files")
async def UploadImage(file: bytes = File(...)):
    try :
        with open('image.jpg','wb') as image:
            image.write(file)
            image.close()
        res = predict(file)
        return {"prediction": res}
    except Exception as e:
        return e

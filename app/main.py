from fastapi import FastAPI, HTTPException, status

app = FastAPI()

@app.get("/")
async def hello_world():
    return {"response": "Hello World"}

@app.get("/upload_images")
async def upload_images():
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED
    )


@app.get("/process_raw_images")
async def process_raw_images():
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED
    )


@app.get("/track_growth")
async def track_growth():
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED
    )

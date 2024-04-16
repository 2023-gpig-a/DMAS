from datetime import datetime
from fastapi import FastAPI, HTTPException, status, File, UploadFile
from PIL import Image
import io
import torch

from fastapi.staticfiles import StaticFiles

from util import file_handler
from util.data import RawEntry, ProcessedEntry, load_database_config, connect, insert_raw_entry, insert_processed_entry
from util.exceptions import PlantsUndetectedError, GPSUndefinedError
from util.plant_detector import detect
from models.human_detection.human_detector import Classifier as HumanDetector
from app.messages import StatusEnum, MessageResponse

# START OPTIONS
# TODO this is only temporarily disabled for testing, enable in the demo
CHECK_HUMANS = False  # Verify if humans are in the image

# Set up app
app = FastAPI()
app.mount("/ui", StaticFiles(directory="app/static", html=True), name="static")

# Set up PostgreSQL
config = load_database_config()
conn = connect(config)

# Get device for ML
device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
if torch.cuda.is_available():
    device = "cuda"

# Load Models
human_detector = HumanDetector(pretrained=True)
human_detector = human_detector.to(device)


@app.get("/")
async def hello_world():
    return {"response": "Hello World"}


@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):

    # This try catch clause is shamelessly stolen from:
    # https://stackoverflow.com/questions/73810377/how-to-save-an-uploaded-image-to-fastapi-using-python-imaging-library-pil
    try:
        im = Image.open(file.file)
        if im.mode in ("RGBA", "P"):
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, 'JPEG', quality=50)
        buf.seek(0)  # to rewind the cursor to the start of the buffer
    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        file.file.close()
        buf.close()

    # Check For Humans
    if CHECK_HUMANS:
        img_normalised = human_detector.tensor_transform(im)
        img_normalised = img_normalised.unsqueeze_(0)
        img_normalised = img_normalised.to(device)
        output = human_detector(img_normalised)
        output = torch.argmax(output.detach().cpu())
        if output == 1:
            # If found, exit
            return MessageResponse(status=StatusEnum.human_detected, message=["Human Detected"])

    # Get GPS Data
    try:
        gps_info = file_handler.gps_details(im)
    except GPSUndefinedError:
        return MessageResponse(status=StatusEnum.gps_undefined, message=["No GPS Data Attached To Image"])

    # Create Raw Entry
    loc = file_handler.save(im, file.filename)

    raw_entry = RawEntry(
        latitude=list(gps_info["GPSLatitude"]),
        longitude=list(gps_info["GPSLongitude"]),
        image_uri=loc,
        date=datetime.now()
    )

    # Processed Entry
    try:
        plant_data = detect(loc)
        plant_class = plant_data[0]['species']['commonNames'][0]
    except PlantsUndetectedError as _:
        return MessageResponse(status=StatusEnum.plant_not_detected, message=["No plants detected in image"])

    # TODO store the results that pass a threshold instead of the top one
    processed_entry = ProcessedEntry(
        image_uri=loc,
        plant_id=plant_class
    )

    # Insert Entries
    insert_raw_entry(conn, raw_entry)
    insert_processed_entry(conn, processed_entry)

    return MessageResponse(
        status=StatusEnum.success,
        message=[
            f"Human not detected,",
            f"Image saved successfully at {loc}",
            f"Plant most likely: {plant_class}"
        ])


@app.get("/track_growth")
async def track_growth():
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED
    )

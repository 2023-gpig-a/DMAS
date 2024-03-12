from fastapi import FastAPI, HTTPException, status, File, UploadFile
from pydantic import BaseModel
from enum import Enum
from PIL import Image
import io
import torch

from fastapi.staticfiles import StaticFiles
from torch.utils.data import DataLoader

from Util import FileHandler
from Util.Data import RawEntry, ProcessedEntry, PlantIDMapEntry
from Models.SpeciesIdentifiers.IdentifierKnotweed import IdentifierKnotweed
from Models.HumanDetection.HumanDetector import Classifier as HumanDetector


app = FastAPI()
app.mount("/ui", StaticFiles(directory="app/static", html=True), name="static")

# Get device
device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
if torch.cuda.is_available():
    device = "cuda"

# Load Models
knotweed_identifier = IdentifierKnotweed()

human_detector = HumanDetector()
human_detector.load_state_dict(torch.load("Models/HumanDetection/weights/human_classification_weights.pkl"))
human_detector.eval()
human_detector = human_detector.to(device)


class StatusEnum(str, Enum):
    success = "Success"
    human_detected = "Human Detected"


class MessageResponse(BaseModel):
    status: StatusEnum
    message: str


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
        # to get the entire bytes of the buffer use:
        contents = buf.getvalue()
        # or, to read from `buf` (which is a file-like object), call this first:
        buf.seek(0)  # to rewind the cursor to the start of the buffer
    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        file.file.close()
        buf.close()

    # Check for humans
    img_normalised = human_detector.tensor_transform(im)
    img_normalised = img_normalised.unsqueeze_(0)
    img_normalised = img_normalised.to(device)
    output = human_detector(img_normalised)
    output = torch.argmax(output.detach().cpu())
    if output == 1:
        # If found, exit
        return MessageResponse(status=StatusEnum.human_detected, message="Human Detected")

    # Upload image to storage, this returns uri
    loc = FileHandler.save(im, file.filename)

    return MessageResponse(status=StatusEnum.success, message=f"Human not detected, image saved successfully at {loc}")

    # Create raw entry and upload to postgresql
    # Process the raw entry and upload the processed entry to postgresql
    # If we have created a new plant id then run it against our plant id mapper
    #   If we have found a mapping, create PlantIDMapEntry and upload to postgresql


async def process_raw_images(raw_entry: RawEntry):
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED
    )

    # https://www.mdpi.com/2073-4395/10/11/1721

    # Find all plants within the image
    # Cluster and identify these
    # Return processed entry


@app.get("/track_growth")
async def track_growth():
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED
    )

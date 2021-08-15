from PIL import Image
import base64 # enables uploaded image -> PIL image
from io import BytesIO


def b64_to_pil(content):
    string = content.split(';base64,')[-1]
    decoded = base64.b64decode(string)
    buffer = BytesIO(decoded)
    im = Image.open(buffer)
    return im
from PIL import Image
import base64 # enables uploaded image -> PIL image
from io import BytesIO


def uploadedToPil(content):
    """
    Converts an uploaded file from dash into a pillow image type
    """
    string = content.split(';base64,')[-1]
    decoded = base64.b64decode(string)
    buffer = BytesIO(decoded)
    im = Image.open(buffer)
    return im

def pilToHtml(image, imgType):
    """
    Converts a pillow type image to a 64 bit encoded string to display
    Requires the type of the image in string format ("png", "jpeg", etc)
    """
    content = BytesIO()
    image.save(content, format=imgType)
    return base64.b64encode(content.getvalue()).decode("utf-8")
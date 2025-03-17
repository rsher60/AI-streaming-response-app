import base64
import os
from PIL import Image
import pathlib
from langchain_openai import ChatOpenAI
import io
from langchain_core.messages import HumanMessage, SystemMessage



#Helper function to encode the images
async def encode_image(image_bytes):
    # Open the image using PIL
    image = Image.open(io.BytesIO(image_bytes))

    # Re-encode the image as JPEG (or other format)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")  # Or PNG, GIF etc.  JPEG is a good default.

    # Base64 encode the re-encoded image
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return img_str

#Helper function to use GPT 4o LLM
def summarize_image(img_base64 , prompt):
    """ This function will leverage GPT 4 to summarize the base64 encoded images
    """
    chat = ChatOpenAI(model="gpt-4o",max_tokens = 1024)
    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type":"text","text":prompt},
                    {
                        "type":"image_url",
                        "image_url": {"url":f"data:image/jpeg;base64,{img_base64}"}
                    },

                ]
            )
        ]
    )

    return msg.content



#Main function to call GPT and summarize the uploaded image
def get_image_summary(file_name: str) -> str:
    """Generates a concise summary of an image.
    Args:
        file_name: The path to the image file.
    Returns:
        The image summary as a string.
    """
    prompt = """You are an assistant tasked with summarizing images for retrieval.\
        These summaries will be embedded and used to retreive the raw images.\
        Give a consice summary of the image that is well optimised for retrieval and storage in a vector"""

    try:
        # Check if the file is a valid image using pathlib
        image_path = pathlib.Path(file_name)
        if image_path.is_file() and image_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            base64_image = encode_image(file_name)
            return summarize_image(base64_image, prompt)
        else:
            raise ValueError("Unsupported file format")
    except (ValueError, OSError) as e:
        return f"Error: {e}"
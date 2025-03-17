from openai import OpenAI
from fastapi import FastAPI, Form, Request, WebSocket ,File, UploadFile
from typing import Annotated
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
import io
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from utils import *
import asyncio
import base64
load_dotenv()
from PIL import Image
openai = OpenAI(
    api_key = os.getenv('OPENAI_API_SECRET_KEY')
)

app = FastAPI()

templates = Jinja2Templates(directory="templates")

chat_responses = []

@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "chat_responses": chat_responses})


chat_log = [{'role': 'system',
             'content': 'You summarize images.'
             }]






@app.websocket("/ws")
async def chat(websocket: WebSocket):

    await websocket.accept()

    while True:
        #user_input = await websocket.receive_text()
        user_input = await websocket.receive()
        #print(user_input)

        image_bytes = user_input['bytes']
        user_input['message'] = 'Thank You for uploading your image. let me summarise it for you!'

        img_str = await encode_image(image_bytes)


        chat_log.append({'role': 'user', 'content': user_input})
        chat_responses.append(user_input)

        try:
            """
            response = openai.chat.completions.create(
                model='gpt-4',
                messages=chat_log,
                temperature=0.6,
                stream=True
            )
            """
            prompt = """You are an assistant tasked with summarizing images for retrieval.\
                    These summaries will be embedded and used to retreive the raw images.\
                    Give a consice summary of the image that is well optimised for retrieval and storage in a vector"""

            response = openai.chat.completions.create(
                model="gpt-4o",  # Using GPT-4o
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                            },
                        ],
                    }
                ],
                max_tokens=1024,  # Setting a token limit
                stream=True
            )

            ai_response = ''

            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    ai_response += chunk.choices[0].delta.content
                    await asyncio.sleep(0.05)
                    await websocket.send_text(chunk.choices[0].delta.content)
            chat_responses.append(ai_response)

        except Exception as e:
            await websocket.send_text(f'Error: {str(e)}')
            break


@app.post("/", response_class=HTMLResponse)
async def chat(request: Request, user_input: Annotated[str, Form()]):

    chat_log.append({'role': 'user', 'content': user_input})
    chat_responses.append(user_input)

    response = openai.chat.completions.create(
        model='gpt-4',
        messages=chat_log,
        temperature=0.6
    )

    bot_response = response.choices[0].message.content
    chat_log.append({'role': 'assistant', 'content': bot_response})
    chat_responses.append(bot_response)

    return templates.TemplateResponse("home.html", {"request": request, "chat_responses": chat_responses})


















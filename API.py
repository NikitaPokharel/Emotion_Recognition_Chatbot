from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from Scripts.test import *
import uuid
from fastapi.responses import HTMLResponse
import subprocess
# import moviepy.editor as moviepy
app=FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=[""],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return{'Chatbot'}


def buildtoken():
    # Load tokenizer
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('Scripts\\tokenizer.tf')

    # Define start and end token to indicate the start and end of a sentence
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

    # Vocabulary size plus start and end token
    VOCAB_SIZE = tokenizer.vocab_size + 2
    return tokenizer, START_TOKEN,END_TOKEN,VOCAB_SIZE

tokenizer, START_TOKEN,END_TOKEN,VOCAB_SIZE=buildtoken()

@app.post('/audio')
def upload_audio(audio:UploadFile=File(...)):
    file_location = f"static/audio/{uuid.uuid1()}{audio.filename}" #audio.filename le audio object ko filename leko ho
    with open (file_location, "wb+") as file_object:
        file_object.write(audio.file.read())
    dest_path=f'static/audio/{uuid.uuid1()}test.wav'

    print(dest_path)
    command = f'ffmpeg -i {file_location} {dest_path}'
    subprocess.call(command,shell=True)

    reply,MyText=generater(tokenizer, START_TOKEN,END_TOKEN,VOCAB_SIZE,dest_path)

    print("reply:",reply)
    print("mytext:",MyText)
    return {"texts":MyText,"reply": reply}
    # return {"mytext":"hi","reply":"hello"}

@app.post('/text')
def getText():
    str="Hello World"
    return HTMLResponse(content=str)
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import time
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load text data and vocabulary
path_to_file = tf.keras.utils.get_file(
    'shakespeare.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt',
)
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))

# Initialize character and ID mappings
example_text = ['stefan', 'nafets']
chars = tf.strings.unicode_split(example_text, input_encoding='UTF-8')

ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab),
    mask_token=None
)
ids = ids_from_chars(chars)

chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(),
    invert=True,
    mask_token=None
)

# Define function to convert IDs back to text
def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

# Load the model
model = tf.saved_model.load('one_step')

@app.get("/")
def form(request: Request):
    return templates.TemplateResponse('form.html',{'request': request})

@app.post("/pred")
def generate_text(request: Request, next_char: str = Form(...)):
    next_char=tf.constant([next_char])
    start = time.time()
    states = None
    result = [next_char]

    for _ in range(1000):
        next_char, states = model.generate_one_step(
            next_char, states=states
        )
        result.append(next_char)

    result = tf.strings.join(result)
    end = time.time()
    generated_text = result[0].numpy().decode('utf-8')
    return templates.TemplateResponse("index.html", {"request": request, "generated_text": generated_text, "runtime": end-start})

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8221, reload=True)

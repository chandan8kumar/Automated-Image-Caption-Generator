import numpy as np
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Make sure the image path and extension is correct")
    image = image.resize((299,299))
    image = np.array(image)
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

tokenizer = load(open("./tokenizer.p","rb"))
model = load_model("./Models/model_9.h5")
xception_model = Xception(include_top=False, pooling="avg")

def caption_callback(description):
    caption.configure(text=description)

def callback():
    dire=filedialog.askopenfilename()
    photo = extract_features(dire, xception_model)
    description = generate_desc(model, tokenizer, photo, 32)
    description = description.replace('start ', '')
    description = description.replace(' end', '.')
    description=description.capitalize()
    img2 = ImageTk.PhotoImage(Image.open(dire))
    panel.configure(image=img2)
    panel.image = img2
    caption_callback(description)

root = Tk()
root.title("Automated Image Caption Generator")
root.iconbitmap("./Icon.ico")
# root.geometry("800x700")
Label(root,text="Automated Image Caption Generator",font='Calibri 20 bold underline',fg='Black').grid(row=0, column=1, columnspan=2)
path="./default_image.jpg"
img = ImageTk.PhotoImage(Image.open(path))
panel = Label(root, image=img)
caption=Label(root, text="",font='Calibri 12')
panel.grid(row=2, column=1, columnspan=2)
caption.grid(row=3, column=1, columnspan=2)
button1=Button(root, text="Choose Image", command=callback)
button1.grid(row=1, column=0, columnspan=3)
root.bind("<Return>", callback)
root.mainloop()

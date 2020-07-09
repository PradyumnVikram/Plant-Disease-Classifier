import tensorflow as tf
import cv2
import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk

root = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(root, 'siplify.csv'))
CATEGORIES = []
for c in df['PREDICTED']:
    CATEGORIES.append(c)

m = tf.keras.models.load_model(os.path.join(
    root, 'Seq_Acc-9185_Loss-2670_ValAcc-7738_ValLoss-7729__cat_33.model'))
IMG_SIZE = 75


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)

        self.master = master
        self.init_window()

    def init_window(self):
        global button

        self.master.title('PDC')

        self.pack(fill=BOTH, expand=1)

        button = Button(self, text='Upload Image', command=self.upload_img)

        button.place(x=80, y=145)

        T2 = Label(root, text='Plant Disease Classifier', fg='red', font='Helvetica 18 bold')
        T2.place(x=150, y=0)

        P = Label(root, text='Diagnosis:', fg='blue', font='bold')
        P.place(x=20, y=80)

        tp = Label(root, text='Plant Type: ', fg='blue', font='bold')
        tp.place(x=20, y=50)

        cp = Label(root, text='© Pradyumn Vikram')
        cp.place(x=400, y=300)

    def upload_img(self):
        filepath = filedialog.askopenfilename()
        if len(filepath) > 0:
            load = Image.open(filepath)
            load = load.resize((100, 100), Image.ANTIALIAS)
            render = ImageTk.PhotoImage(load)
            img = Label(self, image=render)
            img.image = render
            img.place(x=300, y=40)
            img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            new_array = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
            new_array = new_array/255
            cv2.imshow
            prediction = m.predict_classes([new_array])
            x = CATEGORIES[int(prediction[0])]
            for i, j, k in zip(df['PREDICTED'], df['NAME'], df['TYPE']):
                i = str(i)
                if str(x) == i:

                    T = Label(root, text=k)

                    T.place(x=125, y=55)

                    pt = Label(root, text=j)

                    pt.place(x=115, y=85)
                    flag = True
                    val = True


root = Tk()
root.geometry("550x300")
app = Window(root)
root.mainloop()

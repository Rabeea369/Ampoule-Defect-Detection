import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk
import numpy as np
import joblib
import cv2
from os import listdir, makedirs
from os.path import isfile, join, dirname, realpath, isdir, split
from skimage.feature import hog
from skimage.transform import resize
import sklearn

bottle_image = 0
img_show = 0
folder_path = ''
filename = ''
Med = joblib.load("new_med_cl_def.sav")
Low = joblib.load("new_low_cl_def.sav")
High = joblib.load("new_high_cl_def.sav")
lsi_cls = joblib.load("Line_vs_imp.sav")
size = 128
classifier = 0


def browse():
    global folder_path
    global filename
    path = ' '
    path = tkinter.filedialog.askopenfilename(initialdir="./",
                                              title="Select an Image",
                                              filetypes=(("BMP files",
                                                          "*.bmp*"),
                                                         ("all files",
                                                          "*.*")))

    # label_file_explorer.configure(text="File Opened: "+filename)
    if not path == ' ':
        folder_path, filename = split(path)
        print(folder_path)
        print(filename)
        open_img()


def scr_imp(img):
    lab = 0
    abcd = cv2.Canny(img, 10, 120)
    ker1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    abcd = cv2.morphologyEx(np.float32(abcd), cv2.MORPH_CLOSE, ker1)
    abcd = abcd.astype(np.uint8)
    contours, hierarchy = cv2.findContours(abcd, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(abcd, [approx], 0, (255), 3)
    contours, hierarchy = cv2.findContours(abcd, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    scratch = 0
    imprint = 0
    for cnt in contours:
        r = 1
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        w, h = rect[1]
        box = np.int0(box)
        cv2.drawContours(abcd, [box], 0, (255), 2)
        if w < h:
            if h < 1.5 * w:
                r = 0
        else:
            if w < 1.5 * h:
                r = 0
        if r == 1:
            scratch = scratch + 1
            lab = 1
        else:
            imprint = imprint + 1
            if not lab == 1:
                lab = 2
    # abcd = scipy.ndimage.morphology.binary_fill_holes(abcd)
    return lab


def open_img():
    global bottle_image, img_show
    global folder_path
    global filename
    img = Image.open(join(folder_path, filename))
    bottle_image = np.array(img)
    img = img.resize((400, 400), Image.ANTIALIAS)
    img_show = ImageTk.PhotoImage(img)
    img_label.configure(image=img_show)


def HOG(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = resize(image, (128, 64))
    des, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return des


def classify(image):
    global classifier
    global lsi_cls
    img_des = HOG(image)
    img_des = img_des.reshape(1, -1)
    lab = classifier.predict(img_des)
    typ = 0
    if lab == 1:
        typ = lsi_cls.predict(img_des)
        if typ == 1:
            typ = scr_imp(image)
    return lab, typ


def detect():
    result = r'You Messed Up Again :/'
    global size, bottle_image, img_show
    global classifier
    global Med, Low, High
    line = scratch = imprint = 0
    sensitivity = var_sens.get()
    if sensitivity == "Medium":
        classifier = Med
    elif sensitivity == "Low":
        classifier = Low
    else:
        classifier = High
    labels = np.empty((0, 1))
    if bottle_image.shape[0] > 128:
        image = bottle_image[:, 250:800, :]
        r = image.shape[0]
        c = image.shape[1]
        image = cv2.resize(image, dsize=(int(c), int(r * 0.5)))
        print(image.shape)
        cont_r = True
        samples = np.empty((size, size, 0))
        dummy = np.zeros((image.shape[0], image.shape[1]))
        i = 0
        while cont_r == True:
            j = 0
            cont_c = True
            start_r = int(i * (size) * (3 / 4))
            if (start_r > image.shape[0] - 128):
                cont_r = False
                start_r = image.shape[0] - 128
            end_r = start_r + 128
            while cont_c == True:
                start_c = int(j * (size / 2))
                if (start_c > image.shape[1] - 128):
                    cont_c = False
                    start_c = image.shape[1] - 128
                end_c = start_c + 128
                img = image[start_r:end_r, start_c:end_c, :]
                img = img.astype(np.uint8)
                lab, typ = classify(img)
                if lab == 1:
                    if typ == 0:
                        line += 1
                    elif typ == 1:
                        scratch += 1
                    elif typ == 2:
                        imprint += 1
                    start_p = (start_c, start_r)
                    end_p = (end_c, end_r)
                    dummy = cv2.rectangle(dummy, start_p, end_p, (255), 2)
                labels = np.vstack((labels, lab))
                samples = np.dstack((samples, img))
                j += 1
            i += 1
    else:
        dummy = bottle_image
        lab = classify(dummy)
        labels = np.vstack((labels, lab))
    dummy = dummy.astype(np.uint8)
    image = image.astype(np.uint8)
    contours, hierarchy = cv2.findContours(dummy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    dummy = cv2.drawContours(image, contours, -1, (255, 0, 0), 5)
    if np.any(labels):
        result = 'Defected'
        if line >= scratch and line >= imprint:
            def_result = 'line'
        elif scratch >= line and scratch >= imprint:
            def_result = 'scratch'
        else:
            def_result = 'imprint'
    else:
        result = 'Clean'
        def_result = 'no defect'

    detect_label.config(text=result)
    defect_label.config(text=('defect type: ' + def_result))
    cv2.imwrite("./xample.bmp", dummy)
    dummy = Image.fromarray(dummy)
    dummy = dummy.resize((400, 400), Image.ANTIALIAS)
    img_show = ImageTk.PhotoImage(dummy)
    img_label.configure(image=img_show)
    return


def next():
    global filename, folder_path
    print(folder_path)
    print(filename)
    count = 0
    for files in listdir(folder_path):
        if files == filename:
            filename = listdir(folder_path)[count + 1]
            print(filename)
            break
        count += 1
    open_img()


Height = 700
Width = 800
root = tk.Tk()
root.title("HealthHub Ampoule Bottle Defect Detection")
canvas = tk.Canvas(root, height=Height, width=Width)
frame = tk.Frame(root, bg='#DFD8F3')
frame.place(relx=0.1, rely=0.1, relwidth=0.8, relheight=0.8)

img_label = tk.Label(frame, text="File Explorer using Tkinter",
                     width=400, height=400,
                     fg="blue")
# filename = tk.StringVar()
# folder_path=tk.StringVar()
browse_btn = tk.Button(frame, text='Browse', bg='#C9B6FF', fg='#3E3754', command=lambda: browse())
img_label.place(relx=0.05, rely=0.1, relwidth=0.4, relheight=0.8)
browse_btn.place(relx=0.625, rely=0.1, relwidth=0.2, relheight=0.1)
detect_btn = tk.Button(frame, text='Detect', bg='#C9B6FF', fg='#3E3754', command=lambda: detect())
detect_btn.place(relx=0.625, rely=0.5, relwidth=0.2, relheight=0.1)
detect_label = tk.Label(frame, text="Result",fg="blue")
detect_label.place(relx=0.5, rely=0.65, relwidth=0.2, relheight=0.1)
var_sens = tk.StringVar(frame)
var_sens.set("Medium")
menu_sens = tk.OptionMenu(frame, var_sens, "Low", "Medium", "High")
menu_sens.place(relx=0.625, rely=0.3, relwidth=0.2, relheight=0.1)
next_btn = tk.Button(frame, text='Next Image', bg='#C9B6FF', fg='#3E3754', command=lambda: next())
next_btn.place(relx=0.625, rely=0.8, relwidth=0.2, relheight=0.1)
defect_label = tk.Label(frame, text="Defect Type",fg="blue")
defect_label.place(relx=0.75, rely=0.65, relwidth=0.2, relheight=0.1)

root.mainloop()
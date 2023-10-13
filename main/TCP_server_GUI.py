

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 21:34:28 2021

@author: Rydstorm
"""
import tkinter as tk
import tkinter.filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import joblib
import cv2
import os
from os import listdir, makedirs
from os.path import isfile, join, dirname, realpath, isdir, split, exists
from skimage.feature import hog
from skimage.transform import resize
import socket
import sklearn
from datetime import datetime
from threading import Thread
import csv


class App:
    def __init__(self, root, s):
        self.s = s
        self.root = root
        self.root.state('zoomed')
        self.root.title("HealthHub Ampoule Bottle Defect Detection")
        self.flag = bottle_image = img_show = classifier = 0
        self.path_to_save_img = ''
        self.path_to_save_csv = ''
        self.Med = joblib.load("new_med_cl_def.sav")
        self.Low = joblib.load("new_low_cl_def.sav")
        self.High = joblib.load("new_high_cl_def.sav")
        self.lsi_cls = joblib.load("Line_vs_imp.sav")
        self.size = 128
        self.Height = 700
        self.Width = 800

        self.conn = 'no'
        self.addr = ''

        self.canvas = tk.Canvas(self.root, height=self.Height, width=self.Width)

        self.frame = tk.Frame(self.root, bg='white')
        self.frame.place(relx=0.1, rely=0.1, relwidth=0.8, relheight=0.8)

        self.img_label = tk.Label(self.frame, text="File Explorer using Tkinter", width=400, height=400, fg="gray")
        self.img_label.place(relx=0.1, rely=0.1, relwidth=0.4, relheight=0.8)

        self.connect_btn = tk.Button(self.frame, text="Connect", bg='black', fg='white', command=lambda: self.connect())
        self.connect_btn.place(relx=0.55, rely=0.6, relwidth=0.2, relheight=0.1)

        self.disconnect_btn = tk.Button(self.frame, text="Disconnect", bg='black', fg='white',
                                        command=lambda: self.disconnect())
        self.disconnect_btn.place(relx=0.75, rely=0.6, relwidth=0.2, relheight=0.1)

        self.start_btn = tk.Button(self.frame, text="Start", bg='black', fg='white', command=lambda: self.start())
        self.start_btn.place(relx=0.55, rely=0.75, relwidth=0.2, relheight=0.1)

        self.stop_btn = tk.Button(self.frame, text="Stop", bg='black', fg='white', command=lambda: self.stop())
        self.stop_btn.place(relx=0.75, rely=0.75, relwidth=0.2, relheight=0.1)

        self.client_lab = tk.Label(self.frame, text="No client connected", fg="gray")
        self.client_lab.place(relx=0.6, rely=0.1, relwidth=0.3, relheight=0.05)

        self.server_label = tk.Label(self.frame, text=("Operating on " + str(TCP_IP) + " , Port: " + str(TCP_PORT)),
                                     fg="gray")
        self.server_label.place(relx=0.6, rely=0.25, relwidth=0.3, relheight=0.05)

        self.detect_label = tk.Label(self.frame, text="Result", fg="gray")
        self.detect_label.place(relx=0.55, rely=0.9, relwidth=0.2, relheight=0.05)

        self.var_sens = tk.StringVar(self.frame)
        self.var_sens.set("Medium")
        self.menu_sens = tk.OptionMenu(self.frame, self.var_sens, "Low", "Medium", "High")
        self.menu_sens.place(relx=0.65, rely=0.4, relwidth=0.2, relheight=0.1)

        self.defect_label = tk.Label(self.frame, text="Defect Type", fg="gray")
        self.defect_label.place(relx=0.75, rely=0.9, relwidth=0.2, relheight=0.05)

    def recvall(self, count):
        print(count)
        buf = b''
        while count:
            newbuf = self.conn.recv(count)
            # print(newbuf)
            if not newbuf: return None

            buf += newbuf
            count -= len(newbuf)
        return buf

    def HOG(self, image):
        image = resize(image, (128, 64))
        des, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        return des

    def scr_imp(self, img):
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

    def classify(self, image):

        img_des = self.HOG(image)
        img_des = img_des.reshape(1, -1)
        lab = self.classifier.predict(img_des)
        typ = 0
        if lab == 1:
            typ = self.lsi_cls.predict(img_des)
            if typ == 1:
                typ = self.scr_imp(image)
        return lab, typ

    def detect(self):
        line = scratch = imprint = 0
        sensitivity = self.var_sens.get()
        if sensitivity == "Medium":
            self.classifier = self.Med
        elif sensitivity == "Low":
            self.classifier = self.Low
        else:
            self.classifier = self.High
        labels = np.empty((0, 1))
        image = self.bottle_image[:, 250:800]
        r = image.shape[0]
        c = image.shape[1]
        image = cv2.resize(image, dsize=(int(c), int(r * 0.5)))
        print(image.shape)
        cont_r = True
        samples = np.empty((self.size, self.size, 0))
        dummy = np.zeros((image.shape[0], image.shape[1]))
        i = 0
        while cont_r == True:
            j = 0
            cont_c = True
            start_r = int(i * (self.size) * (3 / 4))
            if (start_r > image.shape[0] - 128):
                cont_r = False
                start_r = image.shape[0] - 128
            end_r = start_r + 128
            while cont_c == True:
                start_c = int(j * (self.size / 2))
                if (start_c > image.shape[1] - 128):
                    cont_c = False
                    start_c = image.shape[1] - 128
                end_c = start_c + 128
                img = image[start_r:end_r, start_c:end_c]
                img = img.astype(np.uint8)
                lab, typ = self.classify(img)
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
        dummy = dummy.astype(np.uint8)
        image = image.astype(np.uint8)
        contours, hierarchy = cv2.findContours(dummy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        dummy = cv2.drawContours(image, contours, -1, (0), 5)
        if np.any(labels):
            result = 'Defected'
            answer = 1
            if line >= scratch and line >= imprint:
                def_result = 'line'
            elif scratch >= line and scratch >= imprint:
                def_result = 'scratch'
            else:
                def_result = 'imprint'
        else:
            result = 'Clean'
            answer = 0
            def_result = 'no defect'

        self.detect_label.config(text=result)
        self.defect_label.config(text=('defect type: ' + def_result))
        dummy = Image.fromarray(dummy)
        dummy = dummy.resize((400, 400), Image.ANTIALIAS)
        self.img_show = ImageTk.PhotoImage(dummy)
        self.img_label.configure(image=self.img_show)
        return answer, def_result, dummy

    def connect(self):
        if self.conn == 'no':
            while True:
                self.conn, self.addr = self.s.accept()
                if self.conn:
                    self.client_lab.configure(text=("Client connected: " + str(self.addr)))
                    break
        else:
            messagebox.showinfo("Error",
                                "Already connected to a client! Start detection by pressing 'Start' or Disconnect from current client first.")

        return

    def disconnect(self):
        if (not self.conn == 'no') and (self.start_btn['state'] == tk.NORMAL):
            self.conn.close()
            self.client_lab.configure(text="Client disconnected")
            self.conn = "no"
            self.root.update()
        elif self.start_btn['state'] == tk.DISABLED:
            messagebox.showinfo("Error",
                                "Detection in process! Press 'Stop' to stop the detection first before disconnecting.")
        return

    def save_img(self, image, name, defect):
        now = datetime.now()
        now = now.strftime("%Y-%m-%d_%H-%M-%S")
        name = str(now)
        if defect == "no defect":
            r = 'Clean'
        else:
            r = 'Defected'
        path_img = join(self.path_to_save_img, (name + ".jpg"))
        image.save(path_img)

        with open(self.path_to_save_csv, 'a') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow([name, r, defect])

        return

    def main_func(self):
        if self.flag == 1:
            with self.conn:
                img_cnt = 1;
                while True:
                    # Recieved size of image
                    if self.flag == 0:
                        return
                    length = self.recvall(8)
                    self.root.update()
                    stringData = self.recvall(int(length))
                    self.root.update()
                    data = np.frombuffer(stringData, dtype='uint8')
                    self.bottle_image = cv2.imdecode(data, 0)
                    res, typ, marked_img = self.detect()
                    self.root.update()
                    Result_Message = "Result:" + str(res)  # 0: OK, 1: NG
                    self.conn.sendall(Result_Message.encode())
                    self.save_img(marked_img, str(img_cnt), typ)
                    img_cnt += 1
                    self.root.update()
                self.start_btn['state'] = tk.NORMAL
            self.conn.close()
        return

    def start(self):
        if self.conn == "no":
            messagebox.showinfo("Error", "Connect to a client first to start detection! Use the ""Connect"" button")
        else:
            date = datetime.now()
            date = date.strftime("%Y-%m-%d")
            self.path_to_save_img = join((os.getcwd()), r'Results')
            self.path_to_save_img = join((self.path_to_save_img), date)
            self.path_to_save_csv = join((self.path_to_save_img), (date + '.csv'))
            if not exists(self.path_to_save_img):
                makedirs(self.path_to_save_img)
            if not isfile(self.path_to_save_csv):
                with open(self.path_to_save_csv, 'w') as csvfile:
                    filewriter = csv.writer(csvfile, delimiter=',',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    filewriter.writerow(['Image name', 'Status', 'Defect Type'])
            self.start_btn['state'] = tk.DISABLED
            self.root.update()
            self.flag = 1
            t = Thread(target=self.main_func())
            t.start()
            self.root.update()
        return

    def stop(self):
        self.flag = 0
        self.start_btn['state'] = tk.NORMAL
        self.root.update()
        return


TCP_IP = socket.gethostbyname(socket.gethostname())
print(TCP_IP)
TCP_PORT = 8000
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((TCP_IP, TCP_PORT))
sock.listen(True)
master = tk.Tk()
GUI = App(master, sock)
master.mainloop()
sock.close()







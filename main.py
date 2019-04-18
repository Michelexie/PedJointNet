# -*- coding: utf-8 -*-
# PedJointNet行人頭肩偵測系統-main.py
from tkinter import *
from PIL import Image,ImageTk
import tkinter.filedialog
import subprocess
root = Tk()
#设置标题
root.title('PedJointNet行人頭肩偵測系統')
root.geometry('1280x1280')
e = tkinter.StringVar()
L1 = Label(root)
#load image
def load_image():
    selectFileName = tkinter.filedialog.askopenfilename(title='选择文件')
    img_open = Image.open(selectFileName)
    img = ImageTk.PhotoImage(img_open)
    L1.config(image=img)
    L1.image = img  # keep a reference
    Label(image=img).grid(row=2, column=0,columnspan=5,sticky=W)

#detection
def detection():
    status = subprocess.call(["./A.py", selectFileName])
    Label(image=status).grid(row=2, column=1, columnspan=5, sticky=S)

lbA = Label(root,text = 'PedJointNet行人頭肩偵測系統',bg='#22C9C9', fg='white',
                                   font=('微软雅黑', 36), width='34')
lbA.grid(row = 0,column = 0,columnspan = 10)
# 创建两个按钮
b1 = Button(root, text='輸入影像', bg='SlateGray', fg='white', font=('微软雅黑', 13, "bold"), width=34, height=4,command=load_image)
b1.grid(row=1, column=0, sticky=W, padx=80, pady=80)
b2 = Button(root, text='一鍵偵測',  bg='SkyBlue', fg='white',font=('微软雅黑', 13, "bold"), width=34, height=4,command=detection)
b2.grid(row=1, column=1, sticky=W, padx=80, pady=80)
# 进入消息循环
root.mainloop()
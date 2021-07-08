# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 00:51:50 2019


"""
from tkinter import *  
from PIL import ImageTk,Image  
def pras():
    root = Tk()  
    canvas = Canvas(root, width = 500, height = 500)  
    canvas.pack()  
    img = ImageTk.PhotoImage(Image.open("ge1.png"))  
    canvas.create_image(10, 10, anchor=NW, image=img) 
    root.mainloop()
    
    
pras()













 
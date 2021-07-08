# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 23:52:09 2019

@author: Dell
"""

from tkinter import *  
from PIL import ImageTk,Image  
def pras():
    root = Tk()  
    canvas = Canvas(root, width = 500, height = 500)  
    canvas.pack()  
    img = ImageTk.PhotoImage(Image.open("pras.png"))  
    canvas.create_image(10, 10, anchor=NW, image=img) 
    root.mainloop()
    
    
pras()
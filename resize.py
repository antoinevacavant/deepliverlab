from PIL import Image
import os, sys

path = "/home/utkarsh/Downloads/3Dircadb1.1/PATIENT_DICOM_PNG/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((100,100), Image.ANTIALIAS)
            imResize.save(f +' resized.jpg', 'JPEG', quality=90)

resize()
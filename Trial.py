import cv2
import easyocr

image_path = r"C:\Users\sdzyr\Pictures\Screenshots\Screenshot 2025-08-12 201816.png"
 
img = cv2.imread(image_path)

reader= easyocr.Reader(['en'], gpu=False)

text = reader.readtext(img)

for t in text:
    print(t)
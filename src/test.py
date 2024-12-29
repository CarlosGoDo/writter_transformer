import cv2
import pytesseract
import os



# Funcion encargada de extraer el texto de la imagen
def ocr_core(img):
    text = pytesseract.image_to_string(img)
    return text

# Funcion para obtener las imagenes en una escala de grises
def grey_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Funcion para eliminar el ruido de la imagen 
def remove_noise(img):
    return cv2.medianBlur(img, 5)

# Thresholding 
def thresholding(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#print(str(os.getcwd())+'\src\images\images.jpg')

#img_path = os.path.join(os.path.dirname(__file__), " src\images\images.jpg")
img_path = r"C:\Users\carlo\writter_transformer\src\imagenes\images.PNG"

img = cv2.imread(img_path)


img = grey_scale(img)
img = thresholding(img)
img = remove_noise(img)

print(ocr_core(img))


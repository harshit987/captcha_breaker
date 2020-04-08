import cv2
import numpy as np

img = cv2.imread("./train/QTB.png")
# edge_img = cv2.imread('edges.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray_img, 119, 255, 0)
# cv2.imshow('thresh', thresh)
# edged = cv2.Canny(gray_img, 30, 200)
# Grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Gray_Scale', gray)
# Find Canny edges
# edged = cv2.Canny(gray, 30, 200)
# cv2.imshow('Canny Edged', edged)
# kernel_erosion = np.ones((2,1), np.uint8)
# # kernel_dilation = np.ones((3,2), np.uint8)

kernel_erosion=np.ones((3,3),np.uint8)
kernel_dilation = np.ones((3, 1), np.uint8)
img_dilation = cv2.dilate(thresh, kernel_dilation, iterations=1)
img_dilation=cv2.erode(img_dilation,kernel_erosion,iterations=1)
# img_dilation = cv2.dilate(thresh, kernel_dilation, iterations=1)
# img_erosion = cv2.erode(thresh, kernel_erosion, iterations=1)
# cv2.imshow('Erosion before find contours', img_erosion)
# cv2.imshow('Dilation before find contours', img_dilation)
# # img_dilation1 = cv2.dilate(edged, kernel_dilation, iterations=1)
# # img_erosion1 = cv2.erode(edged, kernel_erosion, iterations=1)
# cv2.imshow('Erosion1 before find contours', img_erosion)
cv2.imshow('Dilation1 before find contours', img_dilation)
contours, hierarchy = cv2.findContours(img_dilation,
                                       cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# cv2.imshow('Canny Edges After Contouring', edged)
l = []
for cnt in contours:
  x, y, w, h = cv2.boundingRect(cnt)
  if ((w<120 and h>60 and w > 20 and h < 120)):
    l.append((x, y, w, h))
fl = []
for item1 in l:
  flag = 0
  for item2 in l:
    if ((item1[0] > item2[0]) and ((item1[0] + item1[2]) < (item2[0] + item2[2])) and (item1[1] > item2[1]) and ((item1[1] + item1[3]) < (item2[1] + item2[3]))):
      flag = 1
      break
  if flag == 0:
    fl.append(item1)
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
i=0
fl.sort()
for cnt in fl:
  (x, y, w, h) = cnt
  i=i+1
  new_img = img_dilation[y-2:y+h+2, x-2: x+w+2]
  # resized_image = cv2.resize(new_img,(int(100),int(100)))
  # cv2.imwrite(str(i) + ".jpg", resized_image)
  cv2.rectangle(img_dilation, (x-2, y-2), (x+w+2, y+h+2), (0,0, 0), 3)
  


print("Number of Contours found = " + str(len(contours))) 
  
# Draw all contours 
# -1 signifies drawing all contours 
# cv2.drawContours(img, contours, -1, (0, 255, 0), 3) 
# cv2.imshow('Dilation after find contours', img_dilation)
# cv2.imshow('Erosion after find contours', img_erosion)
cv2.imshow('Contours', img_dilation) 
# kernel_erosion = np.ones((2,1), np.uint8)
# kernel_dilation = np.ones((3,1), np.uint8)

# # img_erosion = cv2.erode(edge_img, kernel_erosion, iterations=1)
# # img_dilation = cv2.dilate(edge_img, kernel_dilation, iterations=1)
# img_erosion = cv2.erode(edge_img, kernel_erosion, iterations=1)
# img_dilation = cv2.dilate(img_erosion, kernel_dilation, iterations=1)
# # img_erosionO = cv2.erode(img, kernel, iterations=1)
# # img_dilationO = cv2.dilate(img, kernel, iterations=1)


# # cv2.imshow('Input', edge_img)
# cv2.imshow('Erosion', img_erosion)
# # cv2.imshow('Input', img)
# # cv2.imshow('Erosion', img_erosionO)
# cv2.imshow('Dilation', img_dilation)

# cv2.imshow('Edges', edge_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('edges.png', edge_img)
# pytesseract.pytesseract.tesseract_cmd = '/usr/local/Cellar/tesseract/4.0.0_1/bin/tesseract'
# print(pytesseract.image_to_string('./text1.png'))


# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# _, thresh = cv2.threshold(
#     gray_img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


# img_contours = cv2.findContours(
#     thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

# img_contours = sorted(img_contours, key=cv2.contourArea)

# for i in img_contours:
#   if cv2.contourArea(i) > 100:
#     break

# mask = np.zeros(img.shape[:2], np.uint8)

# cv2.drawContours(mask, [i],-1, 255, -1)

# new_img = cv2.bitwise_and(img, img, mask=mask)

# cv2.imshow("Original Image", img)

# cv2.imshow("Image with background removed", new_img)

# cv2.waitKey(0)
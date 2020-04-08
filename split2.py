import cv2
import numpy as np

img = cv2.imread("./train/XPTP.png")
# edge_img = cv2.imread('edges.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray_img, 127, 255, 0)
# cv2.imshow('thresh', thresh)
# edged = cv2.Canny(gray_img, 30, 200)
# Grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Gray_Scale', gray)
# Find Canny edges
# edged = cv2.Canny(gray, 30, 200)
# cv2.imshow('Canny Edged', edged)
# kernel_erosion = np.ones((2,1), np.uint8)
# kernel_dilation = np.ones((2,1), np.uint8)


# img_dilation = cv2.dilate(gray, kernel_dilation, iterations=1)
# img_erosion = cv2.erode(gray, kernel_erosion, iterations=1)
# cv2.imshow('Erosion before find contours', img_erosion)
# cv2.imshow('Dilation before find contours', img_dilation)
# img_dilation1 = cv2.dilate(edged, kernel_dilation, iterations=1)
# img_erosion1 = cv2.erode(edged, kernel_erosion, iterations=1)
# cv2.imshow('Erosion1 before find contours', img_erosion1)
# cv2.imshow('Dilation1 before find contours', img_dilation1)
contours, hierarchy = cv2.findContours(thresh,
                                       cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# cv2.imshow('Canny Edges After Contouring', edged)

# Convert BGR to HSV
# hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

# Lchannel = hsl[:,:,1]
# #change 250 to lower numbers to include more values as "white"
# mask = cv2.inRange(Lchannel, 0, 127)

# res = cv2.bitwise_and(img,img, mask= mask)
# res_not = cv2.bitwise_not(res)
# cv2.imshow('res', res_not)
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# # define range of blue color in HSV
# lower_blue = np.array([110,50,50])
# upper_blue = np.array([130,255,255])
# # Threshold the HSV image to get only blue colors
# mask = cv2.inRange(hsv, lower_blue, upper_blue)
# cv2.imshow('masked', mask)
# # Find Canny edges
# edged = cv2.Canny(mask, 30, 200)
# cv2.imshow('Canny Edged', edged)
kernel_erosion = np.ones((2,1), np.uint8)
kernel_dilation = np.ones((5,5), np.uint8)


# img_dilation = cv2.dilate(mask, kernel_dilation, iterations=3)
# img_erosion = cv2.erode(mask, kernel_erosion, iterations=1)
# cv2.imshow('Erosion before find contours', img_erosion)
# cv2.imshow('Dilation before find contours', img_dilation)
l = []
for cnt in contours:
  x, y, w, h = cv2.boundingRect(cnt)
  if ((w<120 and h>50)):
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
  new_img = thresh[y-10:y+h+10, x-10: x+w+10]
  resized_image = cv2.resize(new_img,(int(100),int(100)))
  cv2.imwrite(str(i) + ".jpg", resized_image)
  cv2.rectangle(img, (x-10, y-10), (x+w+10, y+h+10), (0,0, 0), 3)

for cnt in contours:
# cnt = contours[0]
  x, y, w, h = cv2.boundingRect(cnt)
  rect = cv2.minAreaRect(cnt)
  box = cv2.boxPoints(rect)
  box = np.int0(box)
  cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

print("Number of Contours found = " + str(len(contours))) 
  
# Draw all contours 
# -1 signifies drawing all contours 
# cv2.drawContours(img, contours, -1, (0, 255, 0), 3) 
# cv2.imshow('Dilation after find contours', img_dilation)
# cv2.imshow('Erosion after find contours', img_erosion)
# cv2.imshow('Contours', img) 
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
cv2.imshow('Input', img)
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
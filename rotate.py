import cv2
import numpy as np
from scipy import ndimage

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(
      image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
  return result


def rotateAndSave(image, angle, name):
  rot_img = rotateImage(image, angle)
  # rot_img = image
  gray_img = cv2.cvtColor(rot_img, cv2.COLOR_BGR2GRAY)
  ret, thresh = cv2.threshold(gray_img, 127, 255, 0)
  contours, hierarchy = cv2.findContours(thresh,
                                          cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  l = []
  for cnt in contours:
      x, y, w, h = cv2.boundingRect(cnt)
      l.append((x, y, w, h))
      # x, y, w, h = cv2.boundingRect(cnt)
      # rect = cv2.minAreaRect(cnt)
      # box = cv2.boxPoints(rect)
      # box = np.int0(box)
      # cv2.drawContours(rot_img, [box], 0, (0, 0, 255), 2)
  # cv2.imshow('after rotation',rot_img)
  
  x, y, w, h = l[1]
  # cv2.rectangle(rot_img, (x-5, y-5), (x+w+5, y+h+5), (255, 255, 255), 3)
  new_img = rot_img[y-5:y+h+5, x-5: x+w+5]
  resized_image = cv2.resize(new_img,(int(100),int(100)))
  cv2.imwrite('RotatedData/' + name + ".jpg", resized_image)
  # fl = []
  # for item1 in l:
  #     flag = 0
  #     for item2 in l:
  #         if ((item1[0] > item2[0]) and ((item1[0] + item1[2]) < (item2[0] + item2[2])) and (item1[1] > item2[1]) and ((item1[1] + item1[3]) < (item2[1] + item2[3]))):
  #             flag = 1
  #             break
  #     if flag == 0:
  #         fl.append(item1)
  #         x, y, w, h = item1
  #         cv2.rectangle(rot_img, (x-5, y-5), (x+w+5, y+h+5), (0, 255, 0), 3)
  #         cv2.imwrite('RotatedData/' + name + ".jpg", rot_img[y-5:y+h+5, x-5: x+w+5])


# img = cv2.imread('reference/B.png')
angles = [-30, -20, -10, 0, 10, 20, 30]
for code in range(ord('A'), ord('Z') + 1):
  img = cv2.imread('reference/' + chr(code) + '.png')
  i=0
  for ang in angles:
    i=i+1
    rotateAndSave(img, ang, chr(code) + str(i))


#rotation angle in degree
# rotated = ndimage.rotate(img, 45)
# rotateAndSave(img, 40, 'a new')
        # cv2.rectangle(rot_img, (x-5, y-5), (x+w+10, y+h+10), (255,0, 0), 3)
        # print(fl)
# cv2.imshow('before rotation', img)
# cv2.imshow('after rotation', rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

from PIL import Image
from PIL import ImageFilter
import numpy as np
import cv2
from keras.models import load_model
loaded_model=load_model("model.h5")


def loop(filenames):
  errs=0
  numCharsList=[]
  codes = []
  for file in filenames:
      
      im = Image.open(file)
      white = im.filter(ImageFilter.BLUR).filter(ImageFilter.MaxFilter(15))
      grey = im.convert('L')
      width,height = im.size
      grey.putdata([min(255, max(255 + x[0] - y[0], 255 + x[1] - y[1], 255 + x[2] - y[2])) for (x, y) in zip(im.getdata(), white.getdata())])
      img=cv2.cvtColor(np.array(grey),cv2.COLOR_RGB2BGR)
      gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

      ret, thresh = cv2.threshold(gray_img, 200, 255, 0)
      img_dilation=thresh
      contours, hierarchy = cv2.findContours(img_dilation,
                                          cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      l = []
      for cnt in contours:
          x, y, w, h = cv2.boundingRect(cnt)
          if ((w < 120 and h > 60 and w > 20 and h < 120)):
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
              (x, y, w, h) = item1
      fl.sort()
      i = 0
      code = ''
      try:
          for cnt in fl:
              (x, y, w, h) = cnt
              new_img = img_dilation[y-2:y+h+2, x-2: x+w+2]
              resized_image = cv2.resize(new_img, (int(100), int(100)))
              gray = resized_image

              gray = cv2.resize(255-gray, (100, 100))
              flatten = gray.flatten() / 255.0

              pred = loaded_model.predict(flatten.reshape(1, 100, 100, 1))
              code=code+chr(pred.argmax()+65)
              i = i+1
      except:
          errs+=1
          i=4
          code = "AAAA"
      numCharsList.append(i)
      codes.append(code)
  numChars = np.array(numCharsList) 
  return (numChars, codes)
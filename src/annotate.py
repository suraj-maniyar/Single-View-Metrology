import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt

img = cv2.imread('image.jpg')
r,c,temp = img.shape
img = cv2.resize(img,(c/4,r/4))


P1 = [465,400,1]
P2 = [615,280,1]
P3 = [435,260,1]
P4 = [604,140,1]
P5 = [186,126,1]
P6 = [355,55,1]
P7 = [235,255,1]


X1_e1 = P1
X1_e2 = P2

X2_e1 = P3
X2_e2 = P4

X3_e1 = P5
X3_e2 = P6

Y1_e1 = P1
Y1_e2 = P7

Y2_e1 = P3
Y2_e2 = P5

Y3_e1 = P4
Y3_e2 = P6

Z1_e1 = P1
Z1_e2 = P3

Z2_e1 = P2
Z2_e2 = P4

Z3_e1 = P7
Z3_e2 = P5



###  X  ###
cv2.line(img,(X1_e1[0],X1_e1[1]),(X1_e2[0],X1_e2[1]),(0,0,255),2)        #blue
cv2.line(img,(X2_e1[0],X2_e1[1]),(X2_e2[0],X2_e2[1]),(0,0,255),2)
cv2.line(img,(X3_e1[0],X3_e1[1]),(X3_e2[0],X3_e2[1]),(0,0,255),2)

###  Y  ###
cv2.line(img,(Y1_e1[0],Y1_e1[1]),(Y1_e2[0],Y1_e2[1]),(0,255,0),2)     #green
cv2.line(img,(Y2_e1[0],Y2_e1[1]),(Y2_e2[0],Y2_e2[1]),(0,255,0),2)
cv2.line(img,(Y3_e1[0],Y3_e1[1]),(Y3_e2[0],Y3_e2[1]),(0,255,0),2)


###  Z  ###
cv2.line(img,(Z1_e1[0],Z1_e1[1]),(Z1_e2[0],Z1_e2[1]),(250,0,0),2)      #red
cv2.line(img,(Z2_e1[0],Z2_e1[1]),(Z2_e2[0],Z2_e2[1]),(250,0,0),2)
cv2.line(img,(Z3_e1[0],Z3_e1[1]),(Z3_e2[0],Z3_e2[1]),(255,0,0),2)



wo = P1
ref_x = P2
ref_y = P7
ref_z = P3

data = np.zeros((7,3))
data[0,:] = P1
data[1,:] = P2
data[2,:] = P3
data[3,:] = P4
data[4,:] = P5
data[5,:] = P6
data[6,:] = P7

data = np.array(data)

df = pd.DataFrame({"X" : data[:,0], "Y" : data[:,1], "Z" : data[:,2]})
df.to_csv("coordinates.csv", index=False)

plt.imshow(img)




import numpy as np
import cv2
import pandas as pd


img = cv2.imread('image.jpg')
r,c,temp = img.shape
img = cv2.resize(img,(c/4,r/4))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

df = pd.read_csv('coordinates.csv');
data = np.array(df)

P1 = data[0]
P2 = data[1]
P3 = data[2]
P4 = data[3]
P5 = data[4]
P6 = data[5]
P7 = data[6]


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

wo = P1
ref_x = P2
ref_y = P7
ref_z = P3

ref_x = np.array([ref_x])
ref_y = np.array([ref_y])
ref_z = np.array([ref_z])


ax1,bx1,cx1 = np.cross(X1_e1,X1_e2)
ax2,bx2,cx2 = np.cross(X2_e1,X2_e2)
Vx = np.cross([ax1,bx1,cx1],[ax2,bx2,cx2])
Vx = Vx/Vx[2]

ay1,by1,cy1 = np.cross(Y1_e1,Y1_e2)
ay2,by2,cy2 = np.cross(Y2_e1,Y2_e2)
Vy = np.cross([ay1,by1,cy1],[ay2,by2,cy2])
Vy = Vy/Vy[2]

az1,bz1,cz1 = np.cross(Z1_e1,Z1_e2)
az2,bz2,cz2 = np.cross(Z2_e1,Z2_e2)
Vz = np.cross([az1,bz1,cz1],[az2,bz2,cz2])
Vz = Vz/Vz[2]



length_x = np.sqrt(np.sum(np.square(ref_x - wo)))   
length_y = np.sqrt(np.sum(np.square(ref_y - wo)))   
length_z = np.sqrt(np.sum(np.square(ref_z - wo)))   


ref_x = np.array(ref_x)
ref_y = np.array(ref_y)
ref_z = np.array(ref_z)
wo = np.array(wo)
Vx = np.array(Vx)
Vy = np.array(Vy)
Vz = np.array(Vz)


ax,resid,rank,s = np.linalg.lstsq( (Vx-ref_x).T , (ref_x - wo).T )
ax = ax[0][0]/length_x

ay,resid,rank,s = np.linalg.lstsq( (Vy-ref_y).T , (ref_y - wo).T )
ay = ay[0][0]/length_y

az,resid,rank,s = np.linalg.lstsq( (Vz-ref_z).T , (ref_z - wo).T )
az = az[0][0]/length_y

px = ax*Vx
py = ay*Vy
pz = az*Vz

P = np.empty([3,4])
P[:,0] = px
P[:,1] = py
P[:,2] = pz
P[:,3] = wo

Hxy = np.zeros((3,3))
Hyz = np.zeros((3,3))
Hzx = np.zeros((3,3))

Hxy[:,0] = px
Hxy[:,1] = py
Hxy[:,2] = wo

Hyz[:,0] = py
Hyz[:,1] = pz
Hyz[:,2] = wo

Hzx[:,0] = px
Hzx[:,1] = pz
Hzx[:,2] = wo


Hxy[0,2] = Hxy[0,2] 
Hxy[1,2] = Hxy[1,2]

Hyz[0,2] = Hyz[0,2] + 100
Hyz[1,2] = Hyz[1,2] + 100 

Hzx[0,2] = Hzx[0,2] - 50
Hzx[1,2] = Hzx[1,2] + 50



r,c,temp = img.shape
Txy = cv2.warpPerspective(img,Hxy,(r,c),flags=cv2.WARP_INVERSE_MAP)
Tyz = cv2.warpPerspective(img,Hyz,(r,c),flags=cv2.WARP_INVERSE_MAP)
Tzx = cv2.warpPerspective(img,Hzx,(r,c),flags=cv2.WARP_INVERSE_MAP)

cv2.imshow("Txy",Txy)
cv2.imshow("Tyz",Tyz)
cv2.imshow("Tzx",Tzx)

cv2.waitKey(0)
cv2.imwrite("XY.jpg",Txy)
cv2.imwrite("YZ.jpg",Tyz)
cv2.imwrite("ZX.jpg",Tzx)
print "Saved"

cv2.destroyAllWindows()

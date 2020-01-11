import cv2
import numpy as np
dic_size = 10
# dim = (dic_size,dic_size)
# for i in range(1,7):
# 	img = cv2.imread('new'+str(i)+'.png', cv2.IMREAD_GRAYSCALE)
# 	print img
# 	cv2.imwrite('new'+str(i)+'.png',resized)

def normalise(arr):
	mx = max(arr)
	mn = min(arr)/mx
	print (1/(1-mn)) , mn/(1-mn)
	for i in range(len(arr)):
		arr[i] /= mx
		arr[i] = arr[i]*(1/(1-mn)) - mn/(1-mn)

def findclosest(num):
    curr = dic_shades[0]
    ret = 0
    for ind,val in enumerate(dic_shades):
        if abs (num - val) < abs (num - curr):
            curr = val
            ret = ind
    return ret

im_gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
dic = [0]*6
dim = (dic_size,dic_size)
for i in range(1,7):
	img = cv2.imread("new"+str(i)+'.png', cv2.IMREAD_GRAYSCALE)
	dic[i-1] = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
dic_shades = [x.mean() for x in dic]
img_shades,img_dim = [],[]
h,w = im_gray.shape
for i in range(h/dic_size):
	for j in range(w/dic_size):
		mn = im_gray[(i*dic_size):(i+1)*dic_size,(j*dic_size):(j+1)*dic_size].mean()
		img_shades.append(mn)
		img_dim.append((i,j))
normalise(dic_shades)
normalise(img_shades)

new_img = np.ones((h ,w ))
for x,y in enumerate(img_dim):
	closest_dic = findclosest(img_shades[x])
	i,j = y[0],y[1]
	new_img[(i*dic_size):(i+1)*dic_size,(j*dic_size) :(j+1)*dic_size ] = dic[closest_dic]

# new_img = np.ones((h + h/dic_size,w + w/dic_size))
# for x,y in enumerate(img_dim):
# 	closest_dic = findclosest(img_shades[x])
# 	i,j = y[0],y[1]
# 	new_img[(i*dic_size + i):(i+1)*dic_size + i,(j*dic_size) + j :(j+1)*dic_size + j] = dic[closest_dic]

(thresh, new_img) = cv2.threshold(new_img, 127, 255, cv2.THRESH_BINARY)
# cv2.imshow("image",im_gray)
cv2.imshow("image",new_img)
cv2.waitKey(0)

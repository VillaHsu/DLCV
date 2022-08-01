import sys
import numpy as np
import cv2
import math

def Magnitude(pixel, mask):
	sizeY = 1
	sizeX = 3
	r = 0 
	for y in range(sizeY):
		for x in range(sizeX):
			r +=  pixel[y][x] * mask[y][x]

	r = r*0.5
	return r

def conv(img):
	img_new = np.full(img.shape,255, np.int)
	mask = [[-1,0,1]]
	for y in range(1,img.shape[0]-1):
		for x in range(1,img.shape[1]-1):
			neighbors = []
			neighbors.append([img[y][x-1],img[y][x],img[y][x+1]])
			G = Magnitude(neighbors, mask)
			img_new[y][x]= G
	
	return img_new

def GM(img1, img2):
	img_new = np.zeros(img1.shape, np.int)
	for y in range(1,img1.shape[0]-1):
		for x in range(1,img1.shape[1]-1):
			img_new[y][x] = (img1[y][x]**2 + img2[y][x]**2)**0.5

	return img_new

def main():
	assert len(sys.argv) == 2

	img = cv2.imread('lena.png', 0)
	img1 = cv2.imread('conv_gaussian_Lena.png', 0)
	img2 = cv2.imread('conv2_gaussian_Lena.png', 0)
	img3 = cv2.imread('Gaussian_Lena.png', 0)

	if sys.argv[1] == 'gaussian':
		dst = cv2.GaussianBlur(img,(3,3),sigmaX= 1/(2*math.log(2)),borderType=cv2.BORDER_DEFAULT)
		cv2.imwrite('Gaussian_Lena.png', dst)
		cv2.imshow('image',dst)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	if sys.argv[1] == 'conv':
		answer = conv(img3)
		cv2.imwrite('conv_gaussian_Lena.png', answer)

	if sys.argv[1] == 'GM':
		answer = GM(img1,img2)
		cv2.imwrite('result2.png', answer)
if __name__ == '__main__':
	main()
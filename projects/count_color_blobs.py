#this program identifies the colors of an image and counts the number of colors
#improvements need to be made and are identified below

import numpy as np
from skimage import io, morphology, measure
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import cv2

img = io.imread('https://i.stack.imgur.com/du0XZ.png')            #couldn't open bin file image so used this dummy image instead
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                       #convert image to grayscale

#print(img.shape)
#print(np.unique(img))

rows, cols = img.shape
X = img.reshape(rows*cols, 1)

kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
labels = kmeans.labels_.reshape(rows, cols)

for i in np.unique(labels):
    blobs = np.int_(morphology.binary_opening(labels == i))
    color = np.around(kmeans.cluster_centers_[i])
    count = len(np.unique(measure.label(blobs))) - 1
    print('Color: {}  >>  Count: {}'.format(color, count))         #prints color values and count of the colors




#things to work on: inputs, reshaping image, bin file conversions, parsing arguments, functions

#input prompt
# input width and height from user
#w= int(input())
#h= int(input())
#wh= [h,w]


#construct the argument parser and pasre the arguments for shape
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--input", required=True,
	#help="/home/project/sample.bin")
#ap.add_argument("-s", "--shape", required=True,
	#help="specify <height>,<width>")
#rgs = vars(ap.parse_args())


#input bin file
#with open('sample.bin', mode='rb') as f:
    #imagebin = np.fromfile(f,dtype=np.uint8,count=w*h).reshape(h,w)


#Make into PIL Image and save
#PILimage = Image.fromarray(imagebin)
#PILimage.save('result.png')
         

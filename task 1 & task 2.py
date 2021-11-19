import numpy as np
import time
import os
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
import math
import pysift
from sklearn.cluster import KMeans
from numpy.random import randint

start = time.time()

pathTraining = "COMP338_Assignment1_Dataset/COMP338_Assignment1_Dataset/Training"
pathTesting = "COMP338_Assignment1_Dataset/COMP338_Assignment1_Dataset/Test"

#function to load images
def loadImagesFiles(path):
    imageFiles = []
    #using os.walk go roots, directories and files
    for root, directories, files in os.walk(path, topdown=False):
        #pull file ending with .jpg
        for file in files:
            if file.endswith(".jpg"):
                imageFiles.append(str(os.path.join(root,file)))
    return imageFiles

#THIS WORKS BUT IT WILL TAKE 18 HOURS TO RUN WITHOUT CV2.SIFT
def tasks1To3():
    #array of image discriptors
    descriptorArray = []

    #load image files
    imgs = loadImagesFiles(path=pathTesting)
    descriptors = []
    #variable for length of list ##optimisation
    lengthOfFiles = len(imgs)
    print(lengthOfFiles)

    def task1():
        #from each image in the path, extract keypoints and descriptors
        for i in range(lengthOfFiles):
            img = cv2.imread(imgs[i], cv2.IMREAD_GRAYSCALE)
            #calculate keypoints and descriptors using pysift
            ##UNCOMMENT NEXT 2 LINES FOR TESTING AND COMMENT OUT LINE 44
            PYsift = cv2.SIFT_create()
            print(img)
            keypoints, descriptor = PYsift.detectAndCompute(img, None)
            ##keypoints, descriptor = pysift.computeKeypointsAndDescriptors(img)
            for j in descriptor:
                descriptors.append(j)
                #print(j)
            descriptorArray.append(descriptors)
        return descriptorArray
    task1()
    end = time.time()
    totalTime = end - start
    print("Time taken: ", totalTime, " seconds")
    print(descriptorArray)
    print(len(descriptorArray))

    def task2(k, descriptors):
        #define kmeans algorithm, calculate clusters, and determine centres
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(descriptors)
        centreWords = kmeans.cluster_centers_
        #returns an array that contains the central points of each cluster
        return centreWords

    centreWords = task2(500, descriptorArray)

#THIS WORKS BUT IT WILL TAKE 18 HOURS TO RUN

#end = time.time()
#totalTime = end - start
#print("Time taken: ", totalTime, " seconds")
##kmeans = KMeans(n_clusters = 800)
##featureExtraction()

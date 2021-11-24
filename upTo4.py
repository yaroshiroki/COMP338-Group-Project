import numpy as np
import time
import os
import cv2
from matplotlib import pyplot as plt
import pysift
from sklearn.cluster import KMeans
from scipy.spatial import distance

start = time.time()

pathTraining = "COMP338_Assignment1_Dataset/COMP338_Assignment1_Dataset/Training"
pathTesting = "COMP338_Assignment1_Dataset/COMP338_Assignment1_Dataset/Test"

#function to load images
#returns an array of images to be used for training and testing data
def loadImagesFiles(path):
    imageFiles = {}
    #extract the file name as well for labelling
    for filename in os.listdir(path):
        category = []
        newLayerPath = path + "\\" + filename
        if filename == ".DS_Store":
            continue
        for item in os.listdir(newLayerPath):
            img = cv2.imread(newLayerPath + "/" + item, 0)
            if img is not None:
                category.append(img)
            imageFiles[filename] = category
    return imageFiles

#THIS WORKS BUT IT WILL TAKE 18 HOURS TO RUN WITHOUT CV2.SIFT
def tasks1To3():
    #array of image discriptors
    descriptorArray = []

    #load image files
    testImageFiles = loadImagesFiles(path=pathTesting)
    trainImageFiles = loadImagesFiles(path=pathTraining)
    vectors = {}
    descriptors = []

    #calculates the feature vectors and an array of descriptors of sift features
    #used for calculating the sift features
    #uses the file path
    def task1(imageFiles):
        for key, files in imageFiles.items():
            i = 0
            features = []
            for img in files:
                PYsift = cv2.SIFT_create()
                keypoints, descriptor = PYsift.detectAndCompute(img, None)
                # The following 3 lines should be included only when testing
                if i >= 2:
                    break
                i+=1
            #calculate keypoints and descriptors using pysift
##############UNCOMMENT NEXT LINE FOR HAND IN AND COMMENT OUT LINES 51 AND 52
            ##keypoints, descriptor = pysift.computeKeypointsAndDescriptors(img)
                features.append(descriptor)
                for j in descriptor:
                    descriptors.append(j)
                descriptorArray.append(descriptors)
            vectors[key] = features
        return [descriptorArray, vectors]
    #THIS WORKS BUT IT WILL TAKE 18 HOURS TO RUN

    #extracting features for training and testing images
    descriptorArray, vectors = task1(trainImageFiles)
    descriptorArrayTesting, testVectors = task1(testImageFiles)

    #reshaping the descriptorArray from a 3d array to a 2d numpy array
    #used for applying kmeans.fit
    numpyDescriptorArrayy = np.array(descriptorArray)
    nsamples, ndescriptors, nfeatures = numpyDescriptorArrayy.shape
    d2DescriptorArray = numpyDescriptorArrayy.reshape((ndescriptors*nsamples, nfeatures))

    #calculating the centre points of the visual words by applying the kmeans algorithm
    #uses the number of codewords and a 2d descriptor numpy array
    def task2(k, descriptors):
        #define kmeans algorithm, calculate clusters, and determine centres
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(descriptors)
        centreWords = kmeans.cluster_centers_
        #returns an array that contains the central points of each cluster
        return centreWords

    #extracting sift descriptors using a dictionary of 500 codewords
    centreWords = task2(500, d2DescriptorArray)

    #calculating the closest central points to each sift descriptors to find an index
    #uses a descriptor and array of central clusters
    def indexFinder(descriptor, cluster):
        count = 0
        index = 0
        #keep calculating this for every item in the cluster array
        for i in range(len(cluster)):
            if(i == 0):
                #using scipy method to calculate the euclidean distance
               count = distance.euclidean(descriptor, cluster[i])
            else:
                euclideanDistance = distance.euclidean(descriptor, cluster[i])
                if(euclideanDistance < count):
                    index = i
                    count = euclideanDistance
        return index

    #calculating a dictionary of features which we will use for classification
    #uses a dictionary of feature vectors along with the centres of visual words
    def task3(descriptorsDictionary, centreWords):
        featureDictionary = {}
        for key, val in descriptorsDictionary.items():
            categoryArray = []
            for img in val:
                histogram = np.zeros(len(centreWords))
                for feature in img:
                    index = indexFinder(feature, centreWords)
                    histogram[index] += 1
                categoryArray.append(histogram)
            featureDictionary[key] = categoryArray
        return featureDictionary

    #creates histograms for the training and testing daa
    trainingDictionary = task3(vectors, centreWords)
    testingDictionary = task3(testVectors, centreWords)
    #visualise image patches that are assigned to codewords
    plt.hist(centreWords)
    plt.show()
    #return the training and testing histograms
    return (trainingDictionary, testingDictionary)

#calling tasks1To3() to get the training and testing histograms
trainingDictionary, testingDictionary = tasks1To3()

def task4(trainingImages, testingImages):
    numberOfTests = 0
    correctPredictions = 0
    classification = {}
    
    for testKey, testValue in testingImages.items():
        classification[testKey] = [0, 0] # [correct, all]
        for test in testValue:
            makePrediction = 0
            minimum = 0
            key = "a" #predicted
            for trainKey, trainValue in trainingImages.items():
                for train in trainValue:
                    if(makePrediction == 0):
                        minimum = distance.euclidean(test, train)
                        key = trainKey
                        makePrediction += 1
                    else:
                        dist = distance.euclidean(test, train)
                        if(dist < minimum):
                            minimum = dist
                            key = trainKey
            if(testKey == key):
                correctPredictions += 1
                classification[testKey][0] += 1
            numberOfTests += 1
            classification[testKey][1] += 1
            print("numberOfTests",numberOfTests)
            print("correctPredictions",correctPredictions)
            print("classification",classification)
    return [numberOfTests, correctPredictions, classification]

results = task4(trainingDictionary, testingDictionary)
print(results)

    
#task4(trainingDictionary, testingDictionary)

#####

#end = time.time()
#totalTime = end - start
#print("Time taken: ", totalTime, " seconds")


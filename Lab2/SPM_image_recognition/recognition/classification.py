__author__ = 'GongLi'

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import *
import utils
import math

# histogram intersection kernel
def histogramIntersection(M, N):
    m = M.shape[0]
    n = N.shape[0]

    result = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            temp = np.sum(np.minimum(M[i], N[j]))
            result[i][j] = temp

    return result

# classify using SVM
def SVM_Classify(trainDataPath, trainLabelPath, testDataPath, testLabelPath, kernelType):
    trainData = np.array(utils.loadDataFromFile(trainDataPath))
    trainLabels = utils.loadDataFromFile(trainLabelPath)

    testData = np.array(utils.loadDataFromFile(testDataPath))
    testLabels = utils.loadDataFromFile(testLabelPath)


    if kernelType == "HI":

        gramMatrix = histogramIntersection(trainData, trainData)
        clf = SVC(kernel='precomputed')
        clf.fit(gramMatrix, trainLabels)

        predictMatrix = histogramIntersection(testData, trainData)
        SVMResults = clf.predict(predictMatrix)
        correct = sum(1.0 * (SVMResults == testLabels))
        accuracy = correct / len(testLabels)
        print "SVM (Histogram Intersection): " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"

    else:
        clf = SVC(kernel = kernelType)
        clf.fit(trainData, trainLabels)
        SVMResults = clf.predict(testData)

        correct = sum(1.0 * (SVMResults == testLabels))
        accuracy = correct / len(testLabels)
        print "SVM (" +kernelType+"): " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"

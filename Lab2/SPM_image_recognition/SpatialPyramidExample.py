__author__ = 'GongLi'

from recognition import utils
from recognition import classification

def buildHistogram(path, level):

    # Read in vocabulary & data
    voc = utils.loadDataFromFile("Data/voc.pkl")
    trainData = utils.readImages("images/"+path)

    # Transform each feature into histogram
    featureHistogram = []
    labels = []

    index = 0
    for oneImage in trainData:

        featureHistogram.append(voc.buildHistogramForEachImageAtDifferentLevels(oneImage, level))
        labels.append(oneImage.label)

        index += 1

    utils.writeDataToFile("Data/"+path+"HistogramLevel" +str(level)+ ".pkl", featureHistogram)
    utils.writeDataToFile("Data/"+path+"labels.pkl", labels)



def main():

    # 1) build histograms
    level = 2
    buildHistogram("testing", level)
    buildHistogram("training", level)

    # 2) classify
    print " "
    classification.SVM_Classify("Data/trainingHistogramLevel"+str(level)+".pkl", "Data/traininglabels.pkl", "Data/testingHistogramLevel"+str(level)+".pkl", "Data/testinglabels.pkl", "linear")

if __name__ == "__main__":

    main()





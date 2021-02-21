from typing import DefaultDict
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import math
import statistics

def generateTrainData(experiments, trainDataLength):
    trainDataCombinations = np.array(list(combinations(experiments.T, trainDataLength)))

    return trainDataCombinations

def generateTestData(experiments, trainData):
    transposedExperiments = experiments.T
    testData = np.copy(transposedExperiments)

    for state in trainData:
        for j in range(0, len(testData)):
            if (state == testData[j]).all():
                testData = np.delete(testData, j, 0)
                break
    
    return testData

def decimal_to_binary(x, n):
    binary = bin(x).replace("0b", "")
    zeros = n - len(binary)

    return "0"*zeros + binary

def populatePredictorStates(predictorStates, predictors):
    classificator = np.zeros([predictorStates, predictors + 1], dtype=np.int) # -> 1 for classificator
    for i in range(0, predictorStates):
        state = decimal_to_binary(i, predictors)
        classificator[i, 0:predictors] = [s for s in state]

    return classificator

def trainClassificatorWithData(optimalClassificator, trainData):
    predictorStates, _ = optimalClassificator.shape
    
    countZeros = np.zeros([predictorStates])
    countOnes = np.zeros([predictorStates])
    for state in trainData:
        str = np.array2string(state[:-1], separator='')
        decimalState = binary_to_decimal(str[1:-1])
        if state[-1] == 0:
            countZeros[decimalState] = countZeros[decimalState] + 1 
        else:
            countOnes[decimalState] = countOnes[decimalState] + 1

    for i in range(0, predictorStates):
        if countZeros[i] == countOnes[i]:
            optimalClassificator[i, -1] = -1 # -1 marking that classificators with 0 and 1 should be observed separately to find the optimal one 
        elif countZeros[i] > countOnes[i]:
            optimalClassificator[i, -1] = 0
        else:
            optimalClassificator[i, -1] = 1

    return optimalClassificator

def binary_to_decimal(x):
    return int(x, 2)

def getOptimalClassificator(classificator, testData):
    countZeros = DefaultDict(int)
    countOnes = DefaultDict(int)

    for state in testData:
        strState = np.array2string(state[:-1], separator='')
        decimalState = binary_to_decimal(strState[1:-1])

        # this should be fixed in the case when we have two options with -1 and we have to make combinations when we combine them together
        if classificator[decimalState, -1] == -1:
            if state[-1] == 0:
                # testClassificator[decimalState, 0] = testClassificator[decimalState, 0] + 1
                countZeros[strState[1:-1]] += 1
            elif state[-1] == 1:
                countOnes[strState[1:-1]] += 1
            
    if countZeros or countOnes:
        for state in classificator:
            strState = np.array2string(state[:-1], separator='')

            if countZeros[strState[1:-1]] >= countOnes[strState[1:-1]] and countZeros[strState[1:-1]] != 0:
                state[-1] = 0
            elif countZeros[strState[1:-1]] < countOnes[strState[1:-1]]:
                state[-1] = 1

    return classificator

def calculateOptClassificatorError(classificator, testData):
    errors = 0
    for state in testData:
        str = np.array2string(state[:-1], separator='')
        decimalState = binary_to_decimal(str[1:-1])

        if classificator[decimalState, -1] != state[-1]:
            errors = errors + 1

    numberOfTestData, _ = testData.shape
    return round(errors/numberOfTestData, 3)

def trainConstantClassificator(trainData, testData):
    countConstants = np.zeros(2)
    for state in trainData:
        countConstants[state[-1]] = countConstants[state[-1]] + 1

    constantClassificator = 0
    if countConstants[1] > countConstants[0]:
        constantClassificator = 1
    elif countConstants[0] == countConstants[1]:
        constantClassificator = chooseOptConstantClassificator(testData)    

    return constantClassificator

def chooseOptConstantClassificator(testData):
    if calcConstantClassificatorError(0, testData) < calcConstantClassificatorError(1, testData):
        constantClassificator = 0
    else:
        constantClassificator = 1

    return constantClassificator

def calcConstantClassificatorError(constantClassificator, testData):
    errors = 0
    for state in testData:
        if state[-1] != constantClassificator:
            errors = errors + 1

    numberOfTestData, _ = testData.shape
    return round(errors/numberOfTestData, 3)

def calculateCoD(optConstantClassificatorError, optClassificatorError):
    if optConstantClassificatorError != 0:
        cod = (optConstantClassificatorError - optClassificatorError)/optConstantClassificatorError
    else:
        cod = 1

    return cod

def main():
    genes = 3
    predictors = 2
    predictorStates = int(math.pow(2, predictors))

    experiments = np.array([[1, 1, 0, 1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 0, 1, 0, 1],
                            [0, 1, 0, 1, 0, 1, 0, 0]])
    meanCoD = np.zeros(8)
    sumOfCombinations = 0
    for i in range(1, 8):
        trainDataCombinations = generateTrainData(experiments, i)
        sumOfCombinations += len(trainDataCombinations)

        allCoD = np.zeros(len(trainDataCombinations))
        for j in range(0, len(trainDataCombinations)):

            trainData = trainDataCombinations[j]
            testData = generateTestData(experiments, trainDataCombinations[j])

            # generate all possible states of predictors
            commonClassificator = populatePredictorStates(predictorStates, predictors)
            
            # count different outputs for the possible states and get the most common one for the classificator
            commonClassificator = trainClassificatorWithData(commonClassificator, trainData)

            # build all possible classificators depending on the results of the common classificator
            optimalClassificator = getOptimalClassificator(commonClassificator, testData)

            # after the classificators are build, test them on the test data
            optClassificatorError = calculateOptClassificatorError(optimalClassificator, testData)

            #get optimal constant classificator
            optConstantClassificator = trainConstantClassificator(trainData, testData)

            #calculate optimal constant classificator error
            optConstantClassificatorError = calcConstantClassificatorError(optConstantClassificator, testData)
            cod = calculateCoD(optConstantClassificatorError, optClassificatorError)
            allCoD[j] = round(cod, 3)
        
        meanCoD[i] = statistics.mean(allCoD)

    print(sumOfCombinations)
    plt.plot(meanCoD)
    plt.show()

if __name__ == "__main__":
    main()

import numpy as np
import math
import itertools

def generateTrainTestData(experiments, trainDataLength):
    testDataLength = len(experiments.T) - trainDataLength

    trainDataCombinations = np.array(list(itertools.combinations(experiments.T, trainDataLength)))
    

    testDataCombinations = np.array(list(itertools.combinations(experiments.T, testDataLength)))

    return (trainDataCombinations, testDataCombinations)

def decimal_to_binary(x, n):
    binary = bin(x).replace("0b", "")
    zeros = n - len(binary)

    return "0"*zeros + binary

def populatePredictorStates(predictors, predictorStates):
    optimalClassificator = np.zeros([predictorStates, predictors + 1]) # -> 1 for classificator
    for i in range(0, predictorStates):
        state = decimal_to_binary(i, predictors)
        optimalClassificator[i, 0:predictors] = [s for s in state]

    return optimalClassificator

def trainClassificatorWithData(optimalClassificator, trainData):
    predictorStates, _ = optimalClassificator.shape
    
    countZeros = np.zeros([predictorStates])
    countOnes = np.zeros([predictorStates])
    for state in trainData.T:
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

def buildAllClassificators(optimalClassificator, testData):
    allClassificators = []
    allClassificators.append(optimalClassificator)
    for state in testData.T:
        str = np.array2string(state[:-1], separator='')
        decimalState = binary_to_decimal(str[1:-1])

        # this should be fixed in the case when we have two options with -1 and we have to make combinations when we combine them together
        if optimalClassificator[decimalState, -1] == -1:
            optimalClassificator[decimalState, -1] = 0
            allClassificators.append(np.copy(optimalClassificator))

            optimalClassificator[decimalState, -1] = 1
            allClassificators.append(np.copy(optimalClassificator))

    return allClassificators

def calculateOptClassificatorError(classificator, testData):
    predictorStates, _ = classificator.shape
    errors = 0
    for state in testData.T:
        str = np.array2string(state[:-1], separator='')
        decimalState = binary_to_decimal(str[1:-1])

        if classificator[decimalState, -1] != state[-1]:
            errors = errors + 1

    return errors/predictorStates

def trainConstantClassificator(trainData, testData):
    constantClassificator = 0
    countConstants = np.zeros(2)
    for state in trainData.T:
        countConstants[state[-1]] = countConstants[state[-1]] + 1

    if countConstants[0] != countConstants[1]:
        constantClassificator = countConstants.index(max(countConstants))
    else:
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
    for state in testData.T:
        if state[-1] != constantClassificator:
            errors = errors + 1

    return errors/4

def calculateCoD(optConstantClassificatorError, optClassificatorError):
    return (optConstantClassificatorError - optClassificatorError)/optConstantClassificatorError

def main():
    genes = 3
    predictors = 2
    predictorStates = int(math.pow(2, predictors))
    # numberOfExperiments = 8

    experiments = np.array([[1, 1, 0, 1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 0, 1, 0, 1],
                            [0, 1, 0, 1, 0, 1, 0, 0]])
    
    trainDataCombinations, testDataCombinations = generateTrainTestData(experiments, 1)
    # print(testDataCombinations)
    # print(trainDataCombinations)

    trainData = experiments[:, 0:4]
    testData = experiments[:, 4:8]

    # generate all possible states of predictors
    optimalClassificator = populatePredictorStates(predictors, predictorStates)
    
    # count different outputs for the possible states and get the most common one for the classificator
    optimalClassificator = trainClassificatorWithData(optimalClassificator, trainData)
            
    # build all possible classificators depending on the results of the common classificator
    allClassificators = buildAllClassificators(optimalClassificator, testData)

    # after the classificators are build, test them on the test data
    optClassificatorError = np.zeros(len(allClassificators))
    if len(allClassificators) == 1:
        optClassificatorError = calculateOptClassificatorError(allClassificators[0], testData)
    else:
        for i in range(0, len(allClassificators)):
            optClassificatorError[i] = calculateOptClassificatorError(allClassificators[i], testData) 

    #get optimal constant classificator
    optConstantClassificator = trainConstantClassificator(trainData, testData)
    
    #calculate optimal constant classificator error
    optConstantClassificatorError = calcConstantClassificatorError(optConstantClassificator, testData)

    cod = calculateCoD(optConstantClassificatorError, optClassificatorError)
    print(cod)

if __name__ == "__main__":
    main()

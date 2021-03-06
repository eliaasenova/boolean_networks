import numpy as np
import math
import pandas as pd
from decimal import *

from calculate_cod import *

def generate_transition_matrix(numberOfGenes, booleanArgs, booleanFunctions, perturbation):
    numberOfStates = int(math.pow(2, numberOfGenes))
    T = [ [ 0 for i in range(numberOfStates) ] for j in range(numberOfStates) ]
    for i in range(0, numberOfStates):
        for j in range(0, numberOfStates):
            if i != j:
                differentGenes = get_number_of_different_genes(i, j, numberOfGenes)
                T[i][j] = calculate_transition_probability(perturbation, numberOfGenes, differentGenes)
            else:
                T[i][j] = 0

        nextState = get_next_state(i, booleanArgs, booleanFunctions)
        decimalNextState = binary_to_decimal(nextState)
        T[i][decimalNextState] += 1 - sum(T[i])

        print(sum(T[i]))
        
    return T

def get_next_state(i, booleanArgs, booleanFunctions):
    _, numberOfGenes = booleanFunctions.shape
    currentState = decimal_to_binary(i, numberOfGenes)
    nextState = ''
    for i in range(0, numberOfGenes):
        booleanArgsStates = ''.join([currentState[j] for j in booleanArgs[i] if j >= 0])
        booleanArgsDecimal = binary_to_decimal(booleanArgsStates)
        nextState += str(booleanFunctions[booleanArgsDecimal, i])
    
    return nextState

def calculate_transition_probability(perturbation, numberOfGenes, differentGenes):
    transitionProbability = (perturbation**differentGenes)*(1-perturbation)**(numberOfGenes - differentGenes)
    return transitionProbability

def get_number_of_different_genes(fromState, toState, numberOfGenes):
    fromState = decimal_to_binary(fromState, numberOfGenes)
    toState = decimal_to_binary(toState, numberOfGenes)
    differentGenesCounter = 0
    for i in range(0, len(fromState)):
        if fromState[i] != toState[i]:
            differentGenesCounter += 1

    return differentGenesCounter

def printTransitionMatrix(transitionMatrix):
    for i in range(0, len(transitionMatrix)):
        for j in range(0, len(transitionMatrix)):
            print('{0:1f}'.format(transitionMatrix[i][j]), end=" ")

        print("\n")

def main():
    numberOfGenes = 3
    perturbation = Decimal('0.1')
    f_vars = [[0, 1, 2], [0, 1, 2], [1, 2, -1]]
    funcs = np.array([[0, 0, 0],
                      [1, 1, 0],
                      [1, 1, 0],
                      [1, 1, 1],
                      [1, 0, -1],
                      [1, 0, -1],
                      [1, 0, -1],
                      [1, 0, -1]])
    transitionMatrix1 = generate_transition_matrix(numberOfGenes, f_vars, funcs, perturbation)
    pd.DataFrame(transitionMatrix1).to_csv("resources/transition_matrix1.csv", index = False, header=False)

    numberOfGenes2 = 10
    perturbation2 = Decimal('0.01')
    f_vars2 = np.array([[3, 4, -1],[7, 8, 9],[2, 3, 9], [0, 1, 2],[0, 2, -1],[0, 1, 2], [3, 4, 0],[3,4,5],[6,7,-1],[6,7,-1]])
    funcs2 = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [1, 0, 0, 1, 1, 1, 0, 0, 0, 1],
                      [1, 0, 0, 1, 0, 1, 1, 1, 0, 1],
                      [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                      [-1, 1, 0, 1, -1, 1, 1, 1, -1, -1],
                      [-1, 0, 0, 1, -1, 1, 0, 1, -1, -1],
                      [-1, 0, 0, 1, -1, 1, 1, 1, -1, -1],
                      [-1, 1, 1, 1, -1, 1, 1, 1, -1, -1]])
    transitionMatrix2 = generate_transition_matrix(numberOfGenes2, f_vars2, funcs2, perturbation2)
    pd.DataFrame(transitionMatrix2).to_csv("resources/transition_matrix2.csv", index = False, header=False)

if __name__ == "__main__":
    main()
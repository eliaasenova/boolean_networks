import numpy as np
import math

def generate_transition_matrix(numberOfGenes, booleanFunctions, perturbation):
    states = int(math.pow(2, numberOfGenes))
    T = np.zeros([states, states])
    for i in range(0, states):
        nextState = booleanFunctions[i]
        for j in range(0, states):
            if i != j:
                differentGenes = number_of_different_genes(i, j, numberOfGenes)
                T[i][j] = calculate_transition_probability(perturbation, numberOfGenes, differentGenes)
            else:
                T[i][j] = 0

        nextStateStr = np.array2string(nextState, separator='')
        decimalNextState = binary_to_decimal(nextStateStr[1:-1])
        T[i][decimalNextState] += 1 - sum(T[i])

        print(sum(T[i]))
        
    return T

def calculate_transition_probability(perturbation, numberOfGenes, differentGenes):
    return math.pow(perturbation, differentGenes)*math.pow(1-perturbation, numberOfGenes - differentGenes)

def number_of_different_genes(fromState, toState, numberOfGenes):
    fromState = decimal_to_binary(fromState, numberOfGenes)
    toState = decimal_to_binary(toState, numberOfGenes)
    countDifferentGenes = 0
    for i in range(0, len(fromState)):
        if fromState[i] != toState[i]:
            countDifferentGenes += 1

    return countDifferentGenes

def decimal_to_binary(x, n):
    binary = bin(x).replace("0b", "")
    zeros = n - len(binary)

    return "0"*zeros + binary

def binary_to_decimal(x):
    return int(x, 2)

def main():
    booleanFunctions = np.array([[0, 0, 0],
                      [1, 1, 0],
                      [1, 1, 0],
                      [1, 1, 1],
                      [1, 0, 0],
                      [1, 0, 0],
                      [1, 0, 0],
                      [1, 0, 1]])

    transitionMatrix = generate_transition_matrix(3, booleanFunctions, 0.1)
    print(transitionMatrix)

if __name__ == "__main__":
    main()
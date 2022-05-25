import itertools
import pandas as pd
import numpy as np
from gene_network import GeneNetwork
from collections import defaultdict
from dfsAlgorithm import Graph, convertToNeighboursArray

def calculate_powers_with_all_combinations(genes, geneNetwork, numberOfPredictors):
    genesNew = range(10)
    allCombinations = []
    for L in range(1, len(genes)+1):
        for combination in itertools.combinations(genes, L):
            allCombinations.append(combination)

    steadyState = geneNetwork.get_steady_state()

    # have to change it to take the filtered sets of masters and slaves
    masterGenes = []
    slaveGenes = []
    regPowers = []
    incapPowers = []
    canalizingPowers = []
    for subset1 in allCombinations:
        for subset2 in allCombinations:
            masterGenes.append(subset1)
            slaveGenes.append(subset2)
            regPowers.append(geneNetwork.get_reg_power(subset1, subset2, numberOfPredictors, steadyState))
            incapPowers.append(geneNetwork.get_incap_power(subset1, subset2, numberOfPredictors, steadyState))
            canalizingPowers.append(geneNetwork.get_canalizing_power(subset1, subset2, numberOfPredictors, steadyState))
        
    data = {'Master genes': masterGenes, 'Slave genes': slaveGenes, 'Regulation Power': regPowers, 'Incapacitating Power': incapPowers, 'Canalizing Power': canalizingPowers}
    df = pd.DataFrame(data)
    df.to_excel('powers_with_combinations.xlsx')

def calculate_powers_with_filtered_genes(masterGenes, slaveGenes, geneNetwork, numberOfPredictors):
    allMasterCombinations = []
    for L in range(1, len(masterGenes)+1):
        for combination in itertools.combinations(masterGenes, L):
            allMasterCombinations.append(combination)

    allSlaveCombinations = []
    for L in range(1, len(slaveGenes)+1):
        for combination in itertools.combinations(slaveGenes, L):
            allSlaveCombinations.append(combination)

    steadyState = geneNetwork.get_steady_state()

    masterGenes = []
    slaveGenes = []
    regPowers = []
    incapPowers = []
    canalizingPowers = []
    for masterSubset in allMasterCombinations:
        for slaveSubset in allSlaveCombinations:
            masterGenes.append(masterSubset)
            slaveGenes.append(slaveSubset)
            regulatingPower = geneNetwork.get_reg_power(masterSubset, slaveSubset, numberOfPredictors, steadyState)
            regPowers.append(regulatingPower)

            incapacitatingPower = geneNetwork.get_incap_power(masterSubset, slaveSubset, numberOfPredictors, steadyState)
            incapPowers.append(incapacitatingPower)

            canalizingPower = regPower + incapPower
            canalizingPowers.append(canalizingPower)
        
    data = {'Master genes': masterGenes, 'Slave genes': slaveGenes, 'Regulation Power': regPowers, 'Incapacitating Power': incapPowers, 'Canalizing Power': canalizingPowers}
    df = pd.DataFrame(data)
    df.to_excel('powers_with_combinations.xlsx')


def filterGenes(neighbours, geneNetwork, cycles):
    masterGenesDict = {}
    slaveGenesDict = {}

    # mark all genes which are dependencies for other genes as master genes
    for key in range(len(neighbours)):
        if len(neighbours[key]) == 0 and geneNetwork.f_vars[key][0] != -1: # when neighbours[key] == 0???
            slaveGenesDict[key] = True

        for gene in neighbours[key]:
            if len(neighbours[key]) == 1 and key != gene:
                masterGenesDict[key] = True
            elif len(neighbours[key]) > 1:
                masterGenesDict[key] = True

            break

    # mark all genes which are part of a cycle to be both slave and master genes
    for cycle in cycles:
        if len(cycle) > 1: # is this check needed?
            for gene in cycle:
                masterGenesDict[gene] = True
                slaveGenesDict[gene] = True

    return masterGenesDict, slaveGenesDict

def main():

    f_vars4 = np.array([[1, -1, -1, -1],
        [0, -1, -1, -1],
        [0, 2, 3, -1],
        [0, 1, 2, 3]])
    funcs4 = np.array([[0, 0, 1, 0],
        [1, 1, 1, 0],
        [-1, 1, 0, 0],
        [-1, 1, 1, 0],
        [-1, -1, 1, 1],
        [-1, -1, 1, 1],
        [-1, -1, 1, 0],
        [-1, -1, 0, 0],
        [-1, -1, -1, 1],
        [-1, -1, -1, 1],
        [-1, -1, -1, 0],
        [-1, -1, -1, 0],
        [-1, -1, -1, 1],
        [-1, -1, -1, 0],
        [-1, -1, -1, 1],
        [-1, -1, -1, 0]])

    f_vars10_1 = np.array([[ 0,  4,  6, -1],
       [ 2,  4, -1, -1],
       [ 4, -1, -1, -1],
       [ 1,  5,  8,  9],
       [ 1,  2,  6,  9],
       [ 3,  5, -1, -1],
       [ 2,  5,  6, -1],
       [ 2,  5,  6, -1],
       [ 2,  6,  8, -1],
       [ 5, -1, -1, -1]])

    funcs10_1 = np.array([[ 1,  1,  0,  0,  0,  0,  0,  0,  1,  1],
       [ 0,  1,  1,  1,  0,  0,  0,  0,  1,  0],
       [ 1,  0, -1,  1,  1,  0,  1,  0,  0, -1],
       [ 1,  1, -1,  1,  1,  1,  1,  0,  0, -1],
       [ 0, -1, -1,  0,  0, -1,  1,  1,  1, -1],
       [ 0, -1, -1,  1,  0, -1,  0,  1,  0, -1],
       [ 0, -1, -1,  1,  1, -1,  1,  0,  1, -1],
       [ 0, -1, -1,  1,  0, -1,  1,  1,  1, -1],
       [-1, -1, -1,  0,  1, -1, -1, -1, -1, -1],
       [-1, -1, -1,  0,  0, -1, -1, -1, -1, -1],
       [-1, -1, -1,  1,  0, -1, -1, -1, -1, -1],
       [-1, -1, -1,  1,  0, -1, -1, -1, -1, -1],
       [-1, -1, -1,  1,  0, -1, -1, -1, -1, -1],
       [-1, -1, -1,  0,  0, -1, -1, -1, -1, -1],
       [-1, -1, -1,  1,  1, -1, -1, -1, -1, -1],
       [-1, -1, -1,  0,  0, -1, -1, -1, -1, -1]])

    f_vars10_2 = np.array([[ 0,  9, -1, -1],
       [ 5,  7, -1, -1],
       [ 1,  2,  5,  7],
       [ 2, -1, -1, -1],
       [ 6, -1, -1, -1],
       [ 2,  7, -1, -1],
       [ 2,  4,  5,  9],
       [ 3,  6,  7, -1],
       [ 0,  1, -1, -1],
       [ 6, -1, -1, -1]])

    funcs10_2 = np.array([[ 0,  0,  0,  1,  0,  1,  1,  0,  1,  1],
       [ 1,  0,  1,  0,  1,  0,  1,  1,  0,  0],
       [ 1,  1,  0, -1, -1,  1,  1,  1,  0, -1],
       [ 0,  0,  0, -1, -1,  1,  0,  1,  0, -1],
       [-1, -1,  1, -1, -1, -1,  0,  0, -1, -1],
       [-1, -1,  1, -1, -1, -1,  0,  0, -1, -1],
       [-1, -1,  1, -1, -1, -1,  1,  1, -1, -1],
       [-1, -1,  0, -1, -1, -1,  0,  1, -1, -1],
       [-1, -1,  0, -1, -1, -1,  1, -1, -1, -1],
       [-1, -1,  0, -1, -1, -1,  1, -1, -1, -1],
       [-1, -1,  1, -1, -1, -1,  1, -1, -1, -1],
       [-1, -1,  1, -1, -1, -1,  1, -1, -1, -1],
       [-1, -1,  1, -1, -1, -1,  1, -1, -1, -1],
       [-1, -1,  0, -1, -1, -1,  1, -1, -1, -1],
       [-1, -1,  0, -1, -1, -1,  1, -1, -1, -1],
       [-1, -1,  0, -1, -1, -1,  1, -1, -1, -1]])

    f_vars10_3 = np.array([[ 9, -1, -1, -1],
       [ 0,  9, -1, -1],
       [ 4,  5,  8, -1],
       [ 1,  2,  4,  5],
       [ 7, -1, -1, -1],
       [ 1,  6, -1, -1],
       [ 1,  2,  5,  7],
       [ 2,  4, -1, -1],
       [ 5,  9, -1, -1],
       [ 6, -1, -1, -1]])

    funcs10_3 = np.array([[ 1,  0,  0,  1,  1,  1,  0,  1,  0,  1],
       [ 0,  1,  1,  1,  0,  0,  1,  0,  1,  0],
       [-1,  0,  1,  0, -1,  1,  1,  1,  1, -1],
       [-1,  0,  1,  1, -1,  1,  0,  1,  1, -1],
       [-1, -1,  1,  0, -1, -1,  0, -1, -1, -1],
       [-1, -1,  1,  0, -1, -1,  0, -1, -1, -1],
       [-1, -1,  0,  1, -1, -1,  1, -1, -1, -1],
       [-1, -1,  1,  0, -1, -1,  1, -1, -1, -1],
       [-1, -1, -1,  1, -1, -1,  0, -1, -1, -1],
       [-1, -1, -1,  1, -1, -1,  1, -1, -1, -1],
       [-1, -1, -1,  1, -1, -1,  1, -1, -1, -1],
       [-1, -1, -1,  0, -1, -1,  0, -1, -1, -1],
       [-1, -1, -1,  0, -1, -1,  1, -1, -1, -1],
       [-1, -1, -1,  1, -1, -1,  1, -1, -1, -1],
       [-1, -1, -1,  0, -1, -1,  0, -1, -1, -1],
       [-1, -1, -1,  0, -1, -1,  0, -1, -1, -1]])

    f_vars10_4 = np.array([[ 9, -1, -1, -1],
       [ 3,  7, -1, -1],
       [ 0,  2,  8, -1],
       [ 4,  6,  7,  9],
       [ 6, -1, -1, -1],
       [ 5,  6, -1, -1],
       [ 2,  3,  5,  6],
       [ 1,  8, -1, -1],
       [ 2,  9, -1, -1],
       [ 8, -1, -1, -1]])

    funcs10_4 = np.array([[ 0,  1,  0,  0,  1,  0,  1,  1,  1,  1],
       [ 1,  0,  0,  0,  0,  1,  1,  0,  0,  0],
       [-1,  0,  1,  1, -1,  1,  0,  0,  0, -1],
       [-1,  0,  0,  1, -1,  1,  1,  0,  1, -1],
       [-1, -1,  0,  0, -1, -1,  0, -1, -1, -1],
       [-1, -1,  1,  0, -1, -1,  0, -1, -1, -1],
       [-1, -1,  1,  0, -1, -1,  0, -1, -1, -1],
       [-1, -1,  1,  0, -1, -1,  0, -1, -1, -1],
       [-1, -1, -1,  0, -1, -1,  0, -1, -1, -1],
       [-1, -1, -1,  0, -1, -1,  1, -1, -1, -1],
       [-1, -1, -1,  0, -1, -1,  1, -1, -1, -1],
       [-1, -1, -1,  1, -1, -1,  0, -1, -1, -1],
       [-1, -1, -1,  0, -1, -1,  0, -1, -1, -1],
       [-1, -1, -1,  0, -1, -1,  1, -1, -1, -1],
       [-1, -1, -1,  1, -1, -1,  1, -1, -1, -1],
       [-1, -1, -1,  1, -1, -1,  1, -1, -1, -1]])

    f_vars10_5 = np.array([[ 1, -1, -1],
       [ 5,  8, -1],
       [ 0,  6,  7],
       [ 0,  7,  9],
       [ 1, -1, -1],
       [ 5,  6, -1],
       [ 1,  3,  6],
       [ 0,  1, -1],
       [ 5,  6, -1],
       [ 5, -1, -1]])

    funcs10_5 = np.array([[ 0,  0,  0,  0,  0,  1,  1,  0,  0,  0],
       [ 1,  1,  1,  0,  1,  1,  0,  1,  1,  1],
       [-1,  1,  0,  1, -1,  1,  1,  0,  1, -1],
       [-1,  1,  1,  1, -1,  0,  1,  0,  1, -1],
       [-1, -1,  0,  1, -1, -1,  1, -1, -1, -1],
       [-1, -1,  0,  1, -1, -1,  0, -1, -1, -1],
       [-1, -1,  0,  0, -1, -1,  0, -1, -1, -1],
       [-1, -1,  1,  1, -1, -1,  0, -1, -1, -1]])

    neighbours = convertToNeighboursArray(f_vars10_5)
    g = Graph(len(f_vars10_5))
    g.graph = neighbours
    cycles = g.SCC()

    geneNetwork = GeneNetwork(len(f_vars10_5), 0.01, f_vars10_5, funcs10_5)
    print(filterGenes(neighbours, geneNetwork, cycles))
    

if __name__ == "__main__":
    main()
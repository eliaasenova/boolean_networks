import itertools
import pandas as pd
import numpy as np
from gene_network import GeneNetwork
from collections import defaultdict

def convertToNeighboursArray(f_vars):
    neighbours = defaultdict(list)

    for i in range(len(f_vars)):
        for node in f_vars[i]:
            if node != -1:
                neighbours[node].append(i)

            # print(neighbours[node]) 

    return neighbours

class Graph:
  
    def __init__(self,vertices):
        #No. of vertices
        self.V = vertices
         
        # default dictionary to store graph
        self.graph = defaultdict(list)
         
        self.Time = 0


    def SCCUtil(self, u, low, disc, stackMember, st, cycles):
 
        # Initialize discovery time and low value
        disc[u] = self.Time
        low[u] = self.Time
        self.Time += 1
        stackMember[u] = True
        st.append(u)
 
        # Go through all vertices adjacent to this
        for v in self.graph[u]:
             
            # If v is not visited yet, then recur for it
            if disc[v] == -1 :
             
                self.SCCUtil(v, low, disc, stackMember, st, cycles)
 
                # Check if the subtree rooted with v has a connection to
                # one of the ancestors of u
                # Case 1 (per above discussion on Disc and Low value)
                low[u] = min(low[u], low[v])
                         
            elif stackMember[v] == True:
 
                '''Update low value of 'u' only if 'v' is still in stack
                (i.e. it's a back edge, not cross edge).
                Case 2 (per above discussion on Disc and Low value) '''
                low[u] = min(low[u], disc[v])
 
        # head node found, pop the stack and print an SCC
        w = -1 #To store stack extracted vertices
        if low[u] == disc[u]:
            cycle = []
            counter = 0
            while w != u:
                w = st.pop()
                if w == u and counter == 0:
                    pass
                else:
                    cycle.append(w)

                stackMember[w] = False
                counter += 1
                 
            if len(cycle) != 0:
                cycles.append(cycle)

 
    #The function to do DFS traversal.
    # It uses recursive SCCUtil()
    def SCC(self):
        cycles = []
  
        # Mark all the vertices as not visited
        # and Initialize parent and visited,
        # and ap(articulation point) arrays
        disc = [-1] * (self.V)
        low = [-1] * (self.V)
        stackMember = [False] * (self.V)
        st =[]
         
 
        # Call the recursive helper function
        # to find articulation points
        # in DFS tree rooted with vertex 'i'
        for i in range(self.V):
            if disc[i] == -1:
                self.SCCUtil(i, low, disc, stackMember, st, cycles)

        return cycles
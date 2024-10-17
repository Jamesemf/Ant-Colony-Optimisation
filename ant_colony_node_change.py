from collections import defaultdict
import numpy as np
import random
import matplotlib.pyplot as plt
import sys

sys.setrecursionlimit(10000)


class Graph:

    def __init__(self):
        """
        Initializer for creating the construction graph to model BPP
        """
        self.nodes = dict()
        self.edge_pheremone = dict()
        
    def addNode(self, name):
        """
        Add a node to the graph
        """
        self.nodes[name] = []
    
    def getNode(self, name):
        return self.nodes[name]

    def addNodeAdjacents(self, node, adjacent):
        self.nodes[node].append(adjacent)

    def set_edge_pheromones(self, u: tuple, v: tuple, pheromone_value: float) -> None:
        """
        Initalise the pheromone on edges of the graph
        """
        self.edge_pheremone[(u,v)]= pheromone_value

    def get_edge_pheromone(self, u: int, v: int) -> float:
        """
        Get the pheromone located between the edges of two nodes
        """
        return self.edge_pheremone[(u,v)]
    

    # def get_edge_pheromones(self, current_bin):
    #     return [val for key, val in self.edge_pheremone.items() if (current_bin,_) in key]
    
    def update_edge_pheromone(self, u: tuple, v: tuple, delta ) -> None:
        """
        Update the amount of pheromone located on an edge by delta
        """
        self.edge_pheremone[(u,v)] += delta

    def evaporate_edge_pheromone(self, u: tuple, v:tuple, e) -> None:
        """
        Evaporate the amount of pheromone located on the edge of two nodes
        """
        self.edge_pheremone[(u,v)] *= e

    def printGraph(self):
        """
        Display the pheremone levels on all edges of the construction graph
        """
        for u, v in self.edge_pheremone:
            print(f"{u} -> {v} : Pheromone {self.edge_pheremone[(u,v)]}")



class AntColonyOptimization:
    """
    AntColonyOptimization class for ACO problem experiments and solutions.
    """
    best_path = []
    bestFitness = 100000
    num_eval = 0
    max_eval = 0
    graph = None
    bins = None
    items = None
    p = None
    e = None
    name=None
    BPP1=None

    def __init__(self, max_eval: int, bins: int, items: int, p:int, e:float, name:str, BPP1:bool):

        """
        Initializer for an Ant Colony Optimization (ACO) algorithm class. It sets up the essential
        parameters and data structures needed for the ACO process.

        :param trials: The number of trials of the ACO algorithm run
        :param bins: The number of bins 
        :param items: The number of items to be packed into bins
        :param p: The number of ants that explore paths
        :param e: The evaporation rate of pheromone
        """
        self.name = name
        self.max_eval = max_eval
        self.bins = bins
        self.items = items
        self.p = p
        self.e = e
        self.BPP1 = BPP1
        self.initalise_graph(bins, items)

    def reset(self):
        self.initalise_graph(self.bins, self.items)
        self.num_eval = 0
        self.bestFitness = 1000000

    def initalise_graph(self, bins, items):
        self.graph = Graph()

        self.graph.addNode('Start')
        self.graph.addNode('End')

        for item in range(1,items):
            for bin in range(1,bins+1):
                self.graph.addNode((item,bin))
                for nextBin in range(1,bins+1):
                    self.graph.addNodeAdjacents((item,bin), (item+1,nextBin))
                    self.graph.set_edge_pheromones((item,bin),(item+1,nextBin), np.random.random())

        for bin in range(1,bins+1):
            self.graph.addNode((items, bin))
            self.graph.addNodeAdjacents('Start', (1,bin))
            self.graph.addNodeAdjacents((items,bin), 'End')
            self.graph.set_edge_pheromones(('Start'),(1,bin), np.random.random())
            self.graph.set_edge_pheromones((items,bin), ('End'), np.random.random())
    
    def update_path_pheromone(self, path: list, fitness: float):
        """
        Update pheromone in the graph based on fitness achieved by the ant in its path
        :param path: List of vertex represents ant traval sequence.
        :param fitness: Float fitness of ant travel path.
        """
        for u, v in zip(path, path[1:]):
            self.graph.update_edge_pheromone(u, v, (100/fitness))

    def evaporate_pheromone(self):
        """
        Apply pheromone evaporation to the graph
        :return: None
        """
        for u, v in self.graph.edge_pheremone:
            self.graph.evaporate_edge_pheromone(u,v,self.e)

    def traverse_graph(self, cost , current_bin='Start', item=1, path=[]):

        adjacents = self.graph.getNode(current_bin)

        if 'End' in adjacents:
            path.append('End')
            return path, cost
            
        if item == 1:
            path = [('Start')]
            
        pheromones = [self.graph.get_edge_pheromone(current_bin, val) for val in self.graph.getNode(current_bin)]
        bias = [pheromone / sum(pheromones) for pheromone in pheromones]

        next_bin = random.choices(adjacents , bias)[0]
        current_bin = next_bin
        path.append(current_bin)

        if self.BPP1:
            cost[next_bin[1]] += item
        else:
            cost[next_bin[1]] += (item**2)/2
        
        item = item + 1
        
        return self.traverse_graph(cost,current_bin,item,path)
    
    def ant_path_generate(self):
        """
        Traverses the graph, then returns the path and fitness
        """
        cost = {key:0 for key in range(1,self.bins+1)}
        path, cost = self.traverse_graph(cost)
        fitness = max(cost.values()) - min(cost.values())

        return path, fitness
    
    def ant_colony_evaluation(self):
        """
        Main process function for ant colony optimization evaluation.
        Output the best path and its distance and the number of evaluation to find.
        :return: Tuple of (distance, number of evaluation for the best path, the best path)
        """
        while self.num_eval < self.max_eval:

            # "paths_info" as a list for tuple (path, distance, num_eva)
            path_information= []

            # Generate ant paths and compare for best fitness. 
            for _ in range(self.p):
                self.num_eval += 1
                path, fitness = self.ant_path_generate()
                path_information.append((path, fitness))

                if fitness < self.bestFitness:
                    print("New bestFitness: ",fitness," Evaluation: ", self.num_eval, "Decrease of ", \
                           round(((self.bestFitness-fitness)/self.bestFitness)*100),"%" )
                    self.bestFitness = fitness
                    self.best_path = path

            # Apply pheromone update.
            for info in path_information:
                self.update_path_pheromone(info[0],info[1])

            # Apply pheromone evaporation.
            self.evaporate_pheromone()

        print("[ACO] \"{}\" Evaluations Complete.".format(self.name))
        return self.best_path, self.bestFitness

if __name__ == "__main__":

    aco_test_1 = AntColonyOptimization(max_eval=10000,bins=10,items=500,p=10,e=0.9,name="BPP1",BPP1=True)
    
    Trial_paths = []
    Trial_fitness = []

    for x in range(1,5):
        path, fitness = aco_test_1.ant_colony_evaluation()
        print("Trial", x,"Final Fitness: ", fitness)
        Trial_paths.append(path)
        Trial_fitness.append(fitness)
        aco_test_1.reset()


    # plt.figure(figsize=(8, 6))
    # plt.imshow(aco_test_1.pheromone_matrix, cmap='hot', interpolation='nearest', vmin=0., vmax=0.01)
    # plt.colorbar()
    # plt.title("Pheromone Matrix Brazil")
    # plt.xlabel("City j")
    # plt.ylabel("City i")
    # plt.show()
    # plt.imshow(aco_test_2.pheromone_matrix, cmap='hot', interpolation='nearest', vmin=0., vmax=0.1)
    # plt.colorbar()
    # plt.title("Pheromone Matrices Burma 1")
    # plt.xlabel("City j")
    # plt.ylabel("City i")
    # plt.show()
    # plt.imshow(aco_test_3.pheromone_matrix, cmap='hot', interpolation='nearest', vmin=0., vmax=0.1)
    # plt.colorbar()
    # plt.title("Pheromone Matrices Burma 2")
    # plt.xlabel("City j")
    # plt.ylabel("City i")
    # plt.show()

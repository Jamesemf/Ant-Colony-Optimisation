from collections import defaultdict
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import os

sys.setrecursionlimit(10000)

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


class Graph:

    def __init__(self):
        """
        Initializer for creating the construction graph to model BPP
        """
        self.nodes = dict()
        self.edge_pheremone = dict()
        
    def addNode(self, name: tuple)-> None:
        """
        Add a node to the graph
        """
        self.nodes[name] = []
    
    def getAdjacentsToNode(self, name: tuple)-> list[tuple[int,int]]:
        """
        Get the list of adjacent nodes to the specified one
        """
        return self.nodes[name]

    def addNodeAdjacents(self, node: tuple, adjacent: tuple) -> None:
        """
        Add a node to the adjacency list of a specified node
        """
        self.nodes[node].append(adjacent)

    def set_edge_pheromones(self, u: tuple, v: tuple, pheromone_value: float) -> None:
        """
        Initalise the pheromone on the edge between node u and node v of the graph
        """
        self.edge_pheremone[(u,v)]= pheromone_value

    def get_edge_pheromone(self, u: int, v: int) -> float:
        """
        Get the pheromone located between the edge of node u and node v
        """
        return self.edge_pheremone[(u,v)]
    
    
    def update_edge_pheromone(self, u: tuple, v: tuple, delta: float ) -> None:
        """
        Update the amount of pheromone located on the edge between node u and v by delta
        """
        self.edge_pheremone[(u,v)] += delta

    def evaporate_edge_pheromone(self, u: tuple, v: tuple, e: float) -> None:
        """
        Evaporate the amount of pheromone located on the edge of two nodes
        """
        self.edge_pheremone[(u,v)] *= e

    def restore_edge_pheromone(self, u: tuple, v: tuple) -> None:
        """
        Evaporate the amount of pheromone located on the edge of two nodes
        """
        self.edge_pheremone[(u,v)] = np.random.random()

    def printGraph(self):
        """
        Display the pheremone levels on all edges of the construction graph
        """
        for u, v in self.edge_pheremone:
            print(f"{u} -> {v} : Pheromone {self.edge_pheremone[(u,v)]}")


class AntColonyOptimization:
    """
    AntColonyOptimization class for ACO problem experiments
    """
    best_path = []
    bestFitness = 100000000000000
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

        :param max_eval: The number of evaluations to be run each 
        :param bins: The number of bins 
        :param items: The number of items to be packed into bins
        :param p: The size of the colony of ants that explore paths 
        :param e: The evaporation rate of the pheromone
        :param name: The name of the environment for the experiment
        :param BPP1: Indicates whether the items are evaluated with cost BPP1 or BPP2 
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
        """
        Function perfroms actions required to re-run the algorithm. i.e restores the pheromone on all edges to random values
        """
        self.restore_pheromone()
        self.num_eval = 0
        self.bestFitness = 1000000000000

    def initalise_graph(self, bins: int, items: int)->None:
        """
        Function initalises the construction graph representing the BPP and adds pheromones to edges
        : param bins: The number of bins
        : param items: The number of items
        """
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
    
    def update_path_pheromone(self, path: list, fitness: float)->None:
        """
        Update pheromone in the graph based on fitness achieved by the ant in its path
        :param path: List of vertex representing the path of travelled by the ant
        :param fitness: fitness value of the path variable
        """
        for u, v in zip(path, path[1:]):
            self.graph.update_edge_pheromone(u, v, (100/fitness))

    def evaporate_pheromone(self)->None:
        """
        Apply pheromone evaporation to the graph
        """
        for u, v in self.graph.edge_pheremone:
            self.graph.evaporate_edge_pheromone(u,v,self.e)

    def restore_pheromone(self)->None:
        """
        Restores the pheromone to random values for all edges of the construction graph
        """
        for u, v in self.graph.edge_pheremone:
            self.graph.restore_edge_pheromone(u,v)

    def traverse_graph(self, cost , current_bin='Start', item=1, path=[]):
        """
        Recursive method for mapping a path for an ant through a network 
        : param cost: A dictionary that tracks the number of items in each bin
        : paran current_bin: The current bin when the method was called. Default is 'Start' for the first occurrence of the function called
        : param item: The current item that need to be placed in a bin 
        : param path: The path that has been traversed by the ant
        """

        adjacents = self.graph.getAdjacentsToNode(current_bin)

        if 'End' in adjacents:
            path.append('End')
            return path, cost   # If end is adjacent to the current node then recursion ends and path and cost is returned
            
        if item == 1:
            path = [('Start')]
            
        pheromones = [self.graph.get_edge_pheromone(current_bin, val) for val in adjacents] # Pheromone on the adjacent edges 
        bias = [pheromone / sum(pheromones) for pheromone in pheromones]                    # Work out biases

        next_bin = random.choices(adjacents , bias)[0]                                      # Next bin chosen accounting for bias
        current_bin = next_bin
        path.append(current_bin)

        if self.BPP1:
            cost[next_bin[1]] += item
        else:
            cost[next_bin[1]] += (item**2)/2
        
        item = item + 1
        
        return self.traverse_graph(cost,current_bin,item,path)      # Recursive call with new bin, updated path, and next item, and cost
    
    def ant_path_generate(self)->tuple[list[tuple[int,int]], float]:
        """
        Initalises the cost dictionary, aquires a traversal path by an ant and calculates fitness
        : return path: List containing the path of nodes traversed
        : return fitness: The fitness of the path

        """
        cost = {key:0 for key in range(1,self.bins+1)}
        path, cost = self.traverse_graph(cost)
        fitness = max(cost.values()) - min(cost.values())

        return path, fitness
    
    def ant_colony_evaluation(self)->tuple[list[tuple[int,int]], float]:
        """
        Main process function for ant colony optimization evaluations
        : return: The best path found
        : return: The fitness of the best path
        """
        while self.num_eval < self.max_eval:

            # "paths_information" as a list of tuple (path, fitness)
            path_information= []

            # Generate p ant paths and compare each one with the best fitness
            for _ in range(self.p):
                self.num_eval += 1
                path, fitness = self.ant_path_generate()
                path_information.append((path, fitness))

                if fitness < self.bestFitness:
                    print("BestFitness: ",fitness,"| Evaluation: ", self.num_eval, "| Decrease of ", \
                           round(((self.bestFitness-fitness)/self.bestFitness)*100),"%" )
                    self.bestFitness = fitness
                    self.best_path = path

            # Apply pheromone update
            for info in path_information:
                self.update_path_pheromone(info[0],info[1])

            # Apply pheromone evaporation
            self.evaporate_pheromone()

        return self.best_path, self.bestFitness

def plotExperiments(file_location:str, experiments:list[list[float]])->None:
    """
    Creates a visual representation of the best fitness achieved in each experiment
    """
    
    fig = plt.figure(1, figsize = (10,10))
    ax  = fig.add_subplot() # plot on a 2 row x 2 col grid, at cell 1

    labels = np.arange(4)  # the label locations
    colors = ["#24335c", "#275e8c", "#218cb9", "#26bde0", "#4befff"]
    width =0.1

    for x in range(0, len(experiments)):
        ax.bar(x-0.2, experiments[x][0], width, color=colors[0])
        ax.bar(x-0.1, experiments[x][1], width, color=colors[1])
        ax.bar(x,     experiments[x][2], width, color=colors[2])
        ax.bar(x+0.1, experiments[x][3], width, color=colors[3])
        ax.bar(x+0.2, experiments[x][4], width, color=colors[4])
    
    # plot data in grouped manner of bar type 
    plt.xticks(labels, ['p=100 | e=0.9', 'p=100 | e=0.6', 'p=10 | e=0.9', 'p=10 | e=0.6'])  
    plt.ylabel("Best Fitness") 
    fig.legend(["Trial_1","Trial_2","Trial_3","Trial_4","Trial_5"], loc='lower center', ncol = 5)
    # ax.set_ylim(0, 2000)
    
    fig.savefig(file_location)
    plt.close(fig)

def plotMeanOfTrialsForExperiments(file_location:str, experiments:list[list[float]])->None:
    """
    Creates a visual representation of the best fitness achieved in each experiment
    """
    
    fig, ax = plt.subplots(figsize=(10, 10))

    colours = ['#2e2f5c', '#963b76', '#ec5658', '#ffa600']
    labels = ["Experiment_1","Experiment_2","Experiment_3","Experiment_4"]

    labelpos = np.arange(4)

    for x in range(0, len(experiments)):
        value = np.mean(experiments[x])
        standard_deviation = np.std(experiments[x])
        ax.bar(x, value, width=0.5, color=colours[x], label=labels[x])
        ax.errorbar(x, value, standard_deviation, fmt="o", color="black", capsize=5, capthick=2)
        ax.text(x+0.1, value + 40 , str(f'{value}'))
    

    plt.xticks(labelpos, ['p=100 | e=0.9', 'p=100 | e=0.6', 'p=10 | e=0.9', 'p=10 | e=0.6'])  
    plt.ylabel("Mean Fitness In Trials") 

    fig.legend(loc='lower center', ncol = 4)
    # ax.set_ylim(0, 2000)
    
    fig.savefig(file_location)
    plt.close(fig)


if __name__ == "__main__":

    GRAPH_FILE = os.path.join(RUNS_DIR, f'Experiment_Bars_Graph.png')
    MEANGRAPH_FILE = os.path.join(RUNS_DIR, f'Experiment_Mean_Graph.png')


    STAT_FILE = os.path.join(RUNS_DIR, f'Result.txt')

    f = open(STAT_FILE, "w")

    Experiments = []
    Experiment_results = []

    random.seed(1)
    np.random.seed(1)
    Experiments.append(AntColonyOptimization(max_eval=10000,bins=10,items=500,p=100,e=0.9,name="BPP1 Exp_1",BPP1=True))
    random.seed(1)
    np.random.seed(1)
    Experiments.append(AntColonyOptimization(max_eval=10000,bins=10,items=500,p=100,e=0.6,name="BPP1 Exp_2",BPP1=True))
    random.seed(1)
    np.random.seed(1)
    Experiments.append(AntColonyOptimization(max_eval=10000,bins=10,items=500,p=10,e=0.9,name="BPP1 Exp_3",BPP1=True))
    random.seed(1)
    np.random.seed(1)
    Experiments.append(AntColonyOptimization(max_eval=10000,bins=10,items=500,p=10,e=0.6,name="BPP1 Exp_4",BPP1=True))

    # random.seed(1)
    # np.random.seed(1)
    # Experiments.append(AntColonyOptimization(max_eval=10000,bins=50,items=500,p=100,e=0.9,name="BPP2 Exp_1",BPP1=False))
    # random.seed(1)
    # np.random.seed(1)
    # Experiments.append(AntColonyOptimization(max_eval=10000,bins=50,items=500,p=100,e=0.6,name="BPP2 Exp_2",BPP1=False))
    # random.seed(1)
    # np.random.seed(1)
    # Experiments.append(AntColonyOptimization(max_eval=10000,bins=50,items=500,p=10,e=0.9,name="BPP2 Exp_3",BPP1=False))
    # random.seed(1)
    # np.random.seed(1)
    # Experiments.append(AntColonyOptimization(max_eval=10000,bins=50,items=500,p=10,e=0.6,name="BPP2 Exp_4",BPP1=False))


    for experiment in Experiments:
        print("[ACO] \"{}\" Evaluation Begin ".format(experiment.name))
        f.write(f'\nExperiment {experiment.name}\n')
        Trial_fitness = []
        for trial in range(0,5):
            random.seed()
            path, fitness = experiment.ant_colony_evaluation()
            Trial_fitness.append(fitness)
            experiment.reset()
            f.write(f"Trial: {trial+1} Final Fitness: {fitness}\n")
            print(f"Trial: {trial+1} Final Fitness: {fitness}")

        Experiment_results.append(Trial_fitness)
        f.write(f'Trial std = { np.std(Experiment_results[-1])}\n')
    

    # random.seed(0)
    # np.random.seed(0)
    # test = AntColonyOptimization(max_eval=10000,bins=10,items=500,p=10,e=0.9,name="Christian TEST",BPP1=True)
    # path, fitness = test.ant_colony_evaluation()
    # print(f"Trial 1 | Final Fitness: {fitness}")

    plotExperiments(GRAPH_FILE, Experiment_results)
    plotMeanOfTrialsForExperiments(MEANGRAPH_FILE, Experiment_results)

    f.close()

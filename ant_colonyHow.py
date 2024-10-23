"""
Ant Colony Optimization Algorithm for Experiments.

@author Haoyang Cui
"""
import xml.etree.ElementTree as Et
import numpy as np


def find_neighbors(path: list):
    """
    Find possible neighbor path for local search variations with Opt-2.
    :param path: List of Int index about ant path.
    :return: List of Lists about neighbor path.
    """
    neighbor_paths = []
    for i in range(1, len(path) - 1):
        for j in range(i + 1, len(path)):
            if i != j:
                new_path = path[:]
                new_path[i], new_path[j] = new_path[j], new_path[i]
                neighbor_paths.append(new_path)
    return neighbor_paths


class AntColonyOptimization:
    """
    AntColonyOptimization class for ACO problem experiments and solutions.
    """
    data_tree = None
    num_vertex = None
    distance_matrix = None
    heuristic_matrix = None
    pheromone_matrix = None
    shortest_path = []
    shortest_eva = None
    shortest_distance = None
    num_eva = 0

    def __init__(self, data_tree, max_eva=10000, alpha=1, beta=2, q_strength=1, eva_rate=0.5, colony_size=10, start=0,
                 local_iter=10, heu_var=1, local_search=None, elite_ant=None, mmas=None, name="ACO"):
        """
        Initializer for an Ant Colony Optimization (ACO) algorithm class. It sets up the essential
        parameters and data structures needed for the ACO process.

        :param data_tree: XML etree for distance data between cities.
        :param max_eva: Int value for maximum number of evaluations allows.
        :param alpha: Alpha parameter for probability function.
        :param beta: Beta parameter for probability function.
        :param q_strength: Q parameter represent strength of chemical for pheromone update.
        :param eva_rate: Float between 0 and 1, evaporation rate for pheromone.
        :param colony_size: Ant colony size for each evaluation, determine the number of iterations with 'max_eva'.
        :param start: Initial city Int index for start evaluations.
        :param local_iter: Int represents number of Local search iterations for tabu local search.
        :param heu_var: Parameter for tuning heuristic function.
        :param local_search: Local search method applied. Default 'None' for not use; "tabu-search" or "hill-climbing"
            for corresponding local search method.
        :param elite_ant: Float between 0 and 1 for elite ant rate, default 'None' as not applied.
        :param mmas: Tuple (min, max) for minimum and maximum boundaries for MMAS approach,
            default 'None' as not applied.
        :param name: String name for this ACO instance.
        """
        self.data_tree = data_tree
        self.heu_var = heu_var
        self.alpha = alpha
        self.beta = beta
        self.start = start
        self.max_eva = max_eva
        self.eva_rate = eva_rate
        self.q_strength = q_strength
        self.local_iter = local_iter
        self.local_search = local_search
        self.colony_size = colony_size
        self.elite_ant = elite_ant
        self.mmas = mmas
        self.initial_distance_matrix()
        self.initial_pheromone_matrix()
        self.initial_heuristic_matrix()
        self.name = name

    def initial_distance_matrix(self):
        """
        Initialize cost matrix and number of vertex for the construction graph from element tree data.
        :return: None
        """
        elem_graph = self.data_tree.getroot().find('graph')
        self.num_vertex = len(elem_graph)
        # Initialize the cost matrix.
        distance_matrix = np.zeros((self.num_vertex, self.num_vertex))
        # Fill the distance matrix with corresponding cost data.
        for i, vertex in enumerate(elem_graph):
            for edge in vertex:
                j = int(edge.text)
                cost = float(edge.get('cost'))
                distance_matrix[i, j] = cost
        self.distance_matrix = distance_matrix

    def initial_pheromone_matrix(self):
        """
        Initialize the pheromone matrix with random small positive values between 0 and 1 for all edges.
        :return: None
        """
        if self.num_vertex:
            # Random generate a matrix have value between 0 and 1 with the size of number of vertex.
            self.pheromone_matrix = np.random.rand(self.num_vertex, self.num_vertex)
            # Ensure the graph is symmetrical since cost matrix does.
            self.pheromone_matrix = (self.pheromone_matrix + self.pheromone_matrix.T) / 2
            # Ensure value on diagonal is 0 which represents the city itself.
            np.fill_diagonal(self.pheromone_matrix, 0)

    def initial_heuristic_matrix(self):
        """
        Initialize the heuristic matrix based on distance matrix.
        :return:
        """
        if self.num_vertex:
            # Initialize the heuristic matrix.
            heuristic_matrix = np.zeros((self.num_vertex, self.num_vertex))
            # Fill the heuristic matrix with corresponding data.
            for i in range(self.num_vertex):
                for j in range(self.num_vertex):
                    if i != j:
                        heuristic_matrix[i][j] = self.heu_var / self.distance_matrix[i][j]
            self.heuristic_matrix = heuristic_matrix

    def update_path_pheromone(self, path: list, fitness: float):
        """
        Update pheromone in the pheromone matrix based on Q parameter and fitness with the path of ant travel sequence.
        :param path: List of vertex represents ant traval sequence.
        :param fitness: Float fitness on ant travel path.
        :return:
        """
        delta = self.q_strength / fitness
        num_city = len(path)
        for i in range(num_city):
            start_city = path[i]
            # Index math for returning to the initial city
            target_city = path[(i + 1) % num_city]
            # Symmetrically update the pheromone.
            self.pheromone_matrix[start_city, target_city] += delta
            self.pheromone_matrix[target_city, start_city] += delta

    def evaporate_pheromone(self):
        """
        Apply pheromone evaporation to the pheromone matrix.
        :return: None
        """
        self.pheromone_matrix *= (1 - self.eva_rate)

    def transition_prob(self, start: int, allowed_cities: list):
        """
        Calculate transition probabilities for corresponding allowed citis in a list.
        :param start: Int index for initial city.
        :param allowed_cities: List of Int index for cities allowed to travel.
        :return: List of probabilities for traveling to corresponding city in allowed_cities list.
        """
        ph_array = self.pheromone_matrix[start, allowed_cities] ** self.alpha
        he_array = self.heuristic_matrix[start, allowed_cities] ** self.beta
        prob_array = ph_array * he_array
        prob_array /= prob_array.sum()
        return prob_array.tolist()

    def ant_path_generate(self, start: int):
        """
        Generate ant path for an ant with given initial city.
        :param start: Int index for initial city.
        :return: List of Int index for ant travel path.
        """
        travel_path = [start]
        allowed_cities = [i for i in range(self.num_vertex) if i != start]
        cur_city = start
        while allowed_cities:
            prob_list = self.transition_prob(cur_city, allowed_cities)
            next_city = np.random.choice(allowed_cities, p=prob_list)
            cur_city = next_city
            travel_path.append(next_city)
            allowed_cities.remove(cur_city)
        return travel_path

    def total_distance(self, path: list):
        """
        Calculate the total distance of ant path for fitness.
        :param path: List of Int index for ant travel path.
        :return: Float total distance.
        """
        total_distance = 0.
        num_path = len(path)
        for i in range(num_path):
            total_distance += self.distance_matrix[path[i], path[(i + 1) % num_path]]
        return total_distance

    def tabu_search(self, init_path_info: tuple):
        """
        A local search method, tabu search on given ant path for self.local_iter time.
        :param init_path_info: tuple of (path, distance).
        :return: List of Int index path for best result after search.
        """
        best_path = cur_path = init_path_info[0]
        best_dis = init_path_info[1]
        best_evo = init_path_info[2]
        tabu_list = []
        for _ in range(self.local_iter):
            neighbors = find_neighbors(cur_path)
            neighbors = [path for path in neighbors if path not in tabu_list]
            if neighbors:
                next_path = min(neighbors, key=lambda x: self.total_distance(x))
                next_dis = self.total_distance(next_path)
            else:
                break
            if next_dis < best_dis:
                best_path, best_dis = next_path, next_dis
            tabu_list.append(cur_path)
            cur_path = next_path
        return best_path, best_dis, best_evo

    def hill_climbing(self, init_path_info: tuple):
        """
        A local search method, hill climbing search on given ant path until no better neighbor path can be found.
        :param init_path_info: tuple of (path, distance).
        :return: List of Int index path for best result after search.
        """
        best_path = init_path_info[0]
        best_dis = init_path_info[1]
        best_evo = init_path_info[2]

        while True:
            neighbors = find_neighbors(best_path)
            next_path = min(neighbors, key=lambda x: self.total_distance(x))
            next_dis = self.total_distance(next_path)
            # If all neighbor path is worse than the current best solution, break.
            if next_dis >= best_dis:
                break
            best_path, best_dis = next_path, next_dis
        return best_path, best_dis, best_evo

    def ant_colony_evaluation(self):
        """
        Main process function for ant colony optimization evaluation.
        Output the best path and its distance and the number of evaluation to find.
        :return: Tuple of (distance, number of evaluation for the best path, the best path)
        """
        while self.num_eva < self.max_eva:

            # "paths_info" as a list for tuple (path, distance, num_eva)
            paths_info = []
            ant_paths = []
            # Generate ant path with colony size.
            for i in range(self.colony_size):
                ant_paths.append(self.ant_path_generate(self.start))

            # For each path, calculate its distance and zipped with path to store in paths_info.
            for path in ant_paths:
                self.num_eva += 1
                # Check if reach limit of evaluation.
                if self.num_eva > self.max_eva:
                    break
                distance = self.total_distance(path)
                paths_info.append((path, distance, self.num_eva))

            # Sort the paths by their distance for local search method and elite ant approach.
            paths_info.sort(key=lambda x: x[1])

            # Apply local search to the best path if assigned method.
            if self.local_search:
                if self.local_search == "tabu-search":
                    paths_info[0] = self.tabu_search(paths_info[0])
                elif self.local_search == "hill-climb":
                    paths_info[0] = self.hill_climbing(paths_info[0])

            # Update the shortest distance.
            if not self.shortest_distance:
                self.shortest_path = paths_info[0][0]
                self.shortest_distance = paths_info[0][1]
                self.shortest_eva = paths_info[0][2]
            else:
                if paths_info[0][1] < self.shortest_distance:
                    self.shortest_path = paths_info[0][0]
                    self.shortest_distance = paths_info[0][1]
                    self.shortest_eva = paths_info[0][2]

            # Apply pheromone evaporation.
            self.evaporate_pheromone()

            # Update the pheromone matrix, if assigned apply elite ant approach.
            if not self.elite_ant:
                for path_info in paths_info:
                    self.update_path_pheromone(path_info[0], path_info[1])
            else:
                # Only update first "elite_ant" percentage of paths according to their rank index.
                num_elites = int((len(paths_info) + 1) * self.elite_ant)
                rank_index = 0
                for path_info in paths_info[:num_elites]:
                    for _ in range(num_elites - rank_index):
                        self.update_path_pheromone(path_info[0], path_info[1])
                    rank_index += 1
                # Additionally, reinforce the global best path by number of elite ant times.
                for _ in range(num_elites):
                    self.update_path_pheromone(self.shortest_path, self.shortest_distance)

            # Apply Max-min ant system (MMAS) if assigned.
            if self.mmas:
                self.pheromone_matrix[self.pheromone_matrix < self.mmas[0]] = self.mmas[0]
                self.pheromone_matrix[self.pheromone_matrix > self.mmas[1]] = self.mmas[1]

        print("[ACO] \"{}\" Evaluations Complete.".format(self.name))
        return self.shortest_distance, self.shortest_eva, self.shortest_path,


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    brazil_etree = Et.parse("datasets/brazil58.xml")
    burma_etree = Et.parse("datasets/burma14.xml")

    aco_test_1 = AntColonyOptimization(brazil_etree, elite_ant=0.50, name="Test 1")

    aco_test_2 = AntColonyOptimization(burma_etree, elite_ant=0.75, name="Test 2")

    aco_test_3 = AntColonyOptimization(burma_etree, elite_ant=0.25, name="Test 3")

    print(aco_test_1.ant_colony_evaluation())
    print(aco_test_2.ant_colony_evaluation())
    print(aco_test_3.ant_colony_evaluation())

    plt.figure(figsize=(8, 6))
    plt.imshow(aco_test_1.pheromone_matrix, cmap='hot', interpolation='nearest', vmin=0., vmax=0.01)
    plt.colorbar()
    plt.title("Pheromone Matrix Brazil")
    plt.xlabel("City j")
    plt.ylabel("City i")
    plt.show()
    plt.imshow(aco_test_2.pheromone_matrix, cmap='hot', interpolation='nearest', vmin=0., vmax=0.1)
    plt.colorbar()
    plt.title("Pheromone Matrices Burma 1")
    plt.xlabel("City j")
    plt.ylabel("City i")
    plt.show()
    plt.imshow(aco_test_3.pheromone_matrix, cmap='hot', interpolation='nearest', vmin=0., vmax=0.1)
    plt.colorbar()
    plt.title("Pheromone Matrices Burma 2")
    plt.xlabel("City j")
    plt.ylabel("City i")
    plt.show()

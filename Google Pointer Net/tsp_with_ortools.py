from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2



# Distance callback
class CreateDistanceCallback(object):
  """Create callback to calculate distances between points."""
  def __init__(self, dist_matrix):
    """Array of distances between points."""
    self.matrix = dist_matrix

  def Distance(self, from_node, to_node):
    return self.matrix[from_node][to_node]


class Solver(object):

  def __init__(self, tsp_size):
    self.tsp_size = tsp_size          # The number of nodes
    self.num_routes = 1               # The number of routes, which is 1 for TSP
    self.depot = 0                    # The depot is the starting node of the route

  def run(self, dist_matrix):
    # Create routing model

    routing = pywrapcp.RoutingModel(self.tsp_size, self.num_routes, self.depot)
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()

    # Setting first solution heuristic: the
    # method for finding a first solution to the problem.
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    ################################################
    #                                              #
    #               TO EXTEND SEARCH               #
    #                                              #
    ################################################
    """            
    # Setting guided local search in order to find the optimal solution
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit_ms = 40000
    """

    # Create the distance callback, which takes two arguments (the from and to node indices)
    # and returns the distance between these nodes.
    dist_between_nodes = CreateDistanceCallback(10000*dist_matrix)
    dist_callback = dist_between_nodes.Distance

    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
    # Solve, returns a solution if any.
    assignment = routing.SolveWithParameters(search_parameters)

    return assignment.ObjectiveValue()/10000 # optimal distance
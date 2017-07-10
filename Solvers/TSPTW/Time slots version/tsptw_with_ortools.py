import math
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from scipy.spatial.distance import pdist, squareform
import numpy as np

# Distance callback
class CreateDistanceCallback(object):
  """Create callback to calculate distances and travel times between points."""

  def __init__(self, dist_matrix):
    """Initialize distance array."""
    self.matrix = dist_matrix


  def Distance(self, from_node, to_node):
    return self.matrix[from_node][to_node]


# Demand callback
class CreateDemandCallback(object):
  """Create callback to get demands at location node."""

  def __init__(self, demands):
    self.matrix = demands

  def Demand(self, from_node, to_node):
    return self.matrix[from_node]

# Service time (proportional to demand) callback.
class CreateServiceTimeCallback(object):
  """Create callback to get time windows at each location."""

  def __init__(self, demands, time_per_demand_unit):
    self.matrix = demands
    self.time_per_demand_unit = time_per_demand_unit

  def ServiceTime(self, from_node, to_node):
    return self.matrix[from_node] * self.time_per_demand_unit

# Create the travel time callback (equals distance divided by speed).
class CreateTravelTimeCallback(object):
  """Create callback to get travel times between locations."""

  def __init__(self, dist_callback, speed):
    self.dist_callback = dist_callback
    self.speed = speed

  def TravelTime(self, from_node, to_node):
    travel_time = self.dist_callback(from_node, to_node) / self.speed
    return travel_time

# Create total_time callback (equals service time plus travel time).
class CreateTotalTimeCallback(object):
  """Create callback to get total times between locations."""

  def __init__(self, service_time_callback, travel_time_callback):
    self.service_time_callback = service_time_callback
    self.travel_time_callback = travel_time_callback

  def TotalTime(self, from_node, to_node):
    service_time = self.service_time_callback(from_node, to_node)
    travel_time = self.travel_time_callback(from_node, to_node)
    return service_time + travel_time


class Solver(object):

    def __init__(self):
        self.depot = 0
        self.num_vehicles = 1
        self.search_time_limit = 4000000

    def run(self, dist_matrix, demands, start_times, end_times):

        # Setting the number of locations
        num_locations = len(demands)

        # Create routing model
        routing = pywrapcp.RoutingModel(num_locations, self.num_vehicles, self.depot)
        search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()

        # Setting first solution heuristic: the
        # method for finding a first solution to the problem.
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        # Create the distance callback
        dist_between_locations = CreateDistanceCallback(1000*dist_matrix)
        dist_callback = dist_between_locations.Distance

        routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
        demands_at_locations = CreateDemandCallback(demands)
        demands_callback = demands_at_locations.Demand

        # Adding capacity dimension constraints
        VehicleCapacity = 1000000000
        NullCapacitySlack = 0
        fix_start_cumul_to_zero = True
        capacity = "Capacity"
        routing.AddDimension(demands_callback, NullCapacitySlack, VehicleCapacity, fix_start_cumul_to_zero, capacity)


        # Add time dimension.
        time_per_demand_unit = 0
        horizon = 24 * 3600
        time = "Time"
        speed = 1000000000

        service_times = CreateServiceTimeCallback(demands, time_per_demand_unit)
        service_time_callback = service_times.ServiceTime

        travel_times = CreateTravelTimeCallback(dist_callback, speed)
        travel_time_callback = travel_times.TravelTime

        total_times = CreateTotalTimeCallback(service_time_callback, travel_time_callback)
        total_time_callback = total_times.TotalTime

        routing.AddDimension(total_time_callback, horizon, horizon, fix_start_cumul_to_zero, time)

        # Add time window constraints.
        time_dimension = routing.GetDimensionOrDie(time)
        for i in range(1,start_times.size):
            time_dimension.CumulVar(i).SetRange(start_times[i], end_times[i])

        """for start, end in zip(start_times, end_times):
            time_dimension.CumulVar(location).SetRange(start, end)"""


        # Solve and returns a solution if any.
        assignment = routing.SolveWithParameters(search_parameters)

        if assignment:
            total_distance = assignment.ObjectiveValue()/1000

            # Inspect solution.
            capacity_dimension = routing.GetDimensionOrDie(capacity);
            time_dimension = routing.GetDimensionOrDie(time);

            index = routing.Start(0)
            trip = []
            while not routing.IsEnd(index):
                node_index = routing.IndexToNode(index)
                time_var = time_dimension.CumulVar(index)
                tmin = assignment.Min(time_var)
                tmax = assignment.Max(time_var)
                index = assignment.Value(routing.NextVar(index))
                trip.append((node_index, tmin, tmax))

            node_index = routing.IndexToNode(index)
            time_var = time_dimension.CumulVar(index)
            tmin = assignment.Min(time_var)
            tmax = assignment.Max(time_var)
            trip.append((node_index, tmin, tmax))

            return total_distance, trip

        else:
            print "No solution found."
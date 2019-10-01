from collections import defaultdict
from math import inf
import random
import csv
import math

#import numpy as np
#import random
#import pandas as np


def point_avg(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    (points can have more dimensions than 2)
    
    Returns a new point which is the center of all the points.
    """
    assert len(points)>0
    
    res = [0]*len(points[0])
    
    for point in points:
        for ind, entry in enumerate(point):
            #print(type(entry))
            #print(res[ind])
            res[ind] += int(entry)

    res = [tmp/len(points) for tmp in res]
    
    
    return res

#print(point_avg([[1,2,3,4],[5,6,7,8]]))


def update_centers(data_set, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes 
    of both lists correspond to each other.
    Compute the center for each of the assigned groups.
    Return `k` centers in a list
    """

    dic = {}
    for ind, point in enumerate(data_set):
        if assignments[ind] not in dic:
            dic[assignments[ind]] = [point]
        else:
            dic[assignments[ind]].append(point)

    res = []
    for i in list(dic.keys()):
        res.append(point_avg(dic[i]))
    
    
    return res


def assign_points(data_points, centers):
    """
    """
    assignments = []
    for point in data_points:
        shortest = inf  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    Returns the Euclidean distance between a and b
    """
    
    #return np.linalg.norm(np.array(a)-np.array(b))
    #print(a)

    return math.sqrt(sum([(int(x) - int(y)) ** 2 for x, y in zip(a, b)]))


#print(distance([2,3,4],[4,5,6]))

def generate_k(data_set, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    """
    
    return [data_set[i] for i in random.sample(range(len(data_set)),k)]

print(generate_k([[2,3,4],[4,5,6],[7,8,9]],2))
    
def get_list_from_dataset_file(dataset_file):

    with open(dataset_file, 'r') as file:
        dataset = csv.reader(file)
        res = []
        for point in dataset:
            res.append(point)



    return res 




    #return pd.read_csv(dataset_file).values.toList()
    
def cost_function(clustering):
    cost = 0
    for cluster in clustering.keys():
        center = point_avg(clustering[cluster])
        for point in clustering[cluster]:
            cost += distance(point, center)
    return cost


def k_means(dataset_file, k):
    dataset = get_list_from_dataset_file(dataset_file)
    k_points = generate_k(dataset, k)
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    clustering = defaultdict(list)
    for assignment, point in zip(assignments, dataset):
        clustering[assignment].append(point)
    return clustering

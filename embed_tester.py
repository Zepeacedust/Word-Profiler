import json
import numpy

import math

with open("corrected.json") as file:
    mapping = json.load(file)

for word in mapping:
    mapping[word] = numpy.array(mapping[word])


def cos_sim(a,b):
    return numpy.dot(a, b) / (math.sqrt(numpy.dot(a, a)) * math.sqrt(numpy.dot(b, b)))

def find_closest(vec, n):
    closest = list(mapping.items())
    closest.sort(key = lambda x: - cos_sim(x[1], vec))
    return list(map(lambda x : (x[0], cos_sim(x[1], vec)), closest[:n]))
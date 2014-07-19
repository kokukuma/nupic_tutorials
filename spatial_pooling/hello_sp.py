#!/usr/bin/python

import numpy as np
from random import randrange, random
from nupic.research.spatial_pooler import SpatialPooler as SP

def get_random_input(size):
    return [ randrange(2) for i in range(size)]

def addNoise(example, noiseLevel):
    for i in range(int(noiseLevel * example.inputSize)):
        randomPosition = int(random() * example.inputSize)
        example.inputArray[randomPosition] = 1 if example.inputArray[randomPosition] == 0 else 0
    return example

def cos_value(v1_tpl, v2_tpl):
    v1 = np.array(v1_tpl)
    v2 = np.array(v2_tpl)
    dot = np.dot(v1, v2.T)
    len_v1 = np.linalg.norm(v1)
    len_v2 = np.linalg.norm(v2)
    return float(dot / (len_v1 * len_v2))

class Example():
    """
    """
    def __init__(self, inputShape, columnDimensions):
        self.inputShape       = inputShape
        self.columnDimensions = columnDimensions
        self.inputSize        = np.array(inputShape).prod()
        self.columnNumber     = np.array(columnDimensions).prod()
        self.inputArray       = np.zeros(self.inputSize)
        self.activeArray      = np.zeros(self.columnNumber)

        self.sp = SP(self.inputShape,
                self.columnDimensions,
                potentialRadius = self.inputSize,
                numActiveColumnsPerInhArea = int(0.02*self.columnNumber),
                globalInhibition = True,
                synPermActiveInc = 0.01
                )

    def run(self):
        self.sp.compute(self.inputArray, True, self.activeArray)


def main():
    print '## init example'
    #example = Example((5, 5), (10, 10))
    #example = Example((32, 32), (64, 64))
    example = Example((16, 16), (32, 32))
    print example.inputSize
    print example.columnNumber

    # make input binary
    print
    print '## make input'
    example.inputArray = get_random_input(example.inputSize)


    print
    print '## run no change'
    print "input  : ", example.inputArray
    for i in range(3):
        example.run()
        print "output : ", i, example.activeArray.nonzero()
        base = example.activeArray.nonzero()

    print
    print '## add noise 0.1'
    addNoise(example, 0.1)
    example.run()
    print "output : ", i, example.activeArray.nonzero()

    print
    print '## add noise 0.2'
    addNoise(example, 0.2)
    example.run()
    print "output : ", i, example.activeArray.nonzero()


    print
    print '## other input'
    example.inputArray = get_random_input(example.inputSize)
    example.run()
    print "input  : ", example.inputArray
    print "output : ", example.activeArray.nonzero()





if __name__ == "__main__":
    main()

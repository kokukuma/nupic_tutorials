#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy
from nupic.research.TP10X2 import TP10X2 as TP

def main():

    # create Temporal Pooler instance
    tp = TP(numberOfCols=50,           # カラム数
            cellsPerColumn=2,          # 1カラム中のセル数
            initialPerm=0.5,           # initial permanence
            connectedPerm=0.5,         # permanence の閾値
            minThreshold=10,           # 末梢樹状セグメントの閾値の下限?
            newSynapseCount=10,        # ?
            permanenceInc=0.1,         # permanenceの増加
            permanenceDec=0.0,         # permanenceの減少
            activationThreshold=8,     # synapseの発火がこれ以上かを確認している.
            globalDecay=0,             # decrease permanence?
            burnIn=1,                  # Used for evaluating the prediction score
            checkSynapseConsistency=False,
            pamLength=10               # Number of time steps
            )

    # create input vectors to feed to the temporal pooler.
    # Each input vector must be numberOfCols wide.
    # Here we create a simple sequence of 5 vectors
    # representing the sequence A -> B -> C -> D -> E
    x = numpy.zeros((5,tp.numberOfCols), dtype="uint32")
    x[0, 0:10] = 1     # A
    x[1,10:20] = 1     # B
    x[2,20:30] = 1     # C
    x[3,30:40] = 1     # D
    x[4,40:50] = 1     # E

    print x


    # repeat the sequence 10 times
    for i in range(10):
        # Send each letter in the sequence in order
        # A -> B -> C -> D -> E
        print
        print
        print '#### :', i
        for j in range(5):
            tp.compute(x[j], enableLearn = True, computeInfOutput=True)
            #tp.printCells(predictedOnly=False)
            tp.printStates(printPrevious = False, printLearnState = False)

        # sequenceの最後を教える. 絶対必要なわけではないが, あった方が学習速い.
        tp.reset()


    for j in range(5):
        print "\n\n--------","ABCDE"[j],"-----------"
        print "Raw input vector\n",formatRow(x[j])

        # Send each vector to the TP, with learning turned off
        tp.compute(x[j], enableLearn = False, computeInfOutput = True)

        # print predict state
        print "\nAll the active and predicted cells:"
        tp.printStates(printPrevious = False, printLearnState = False)

        # get predict state
        print "\n\nThe following columns are predicted by the temporal pooler. This"
        print "should correspond to columns in the *next* item in the sequence."
        predictedCells = tp.getPredictedState()
        print formatRow(predictedCells.max(axis=1).nonzero())



def formatRow(x):
  s = ''
  for c in range(len(x)):
    if c > 0 and c % 10 == 0:
      s += ' '
    s += str(x[c])
  s += ' '
  return s

if __name__ == "__main__":
    main()

#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy
import sys
import pprint
import random

from optparse import OptionParser
#from nupic.research.TP import TP
from nupic.research.TP10X2 import TP10X2 as TP

#############################################################################
def printOneTrainingVector(x):
    "Print a single vector succinctly."
    print ''.join('1' if k != 0 else '.' for k in x)


#############################################################################
def printAllTrainingSequences(trainingSequences, upTo = 99999):
    for i,trainingSequence in enumerate(trainingSequences):
        print "============= Sequence",i,"================="
        for j,pattern in enumerate(trainingSequence):
            printOneTrainingVector(pattern)

#############################################################################
def getSimplePatterns(numOnes, numPatterns, patternOverlap=0):
    """Very simple patterns. Each pattern has numOnes consecutive
    bits on. The amount of overlap between consecutive patterns is
    configurable, via the patternOverlap parameter.

    Parameters:
    -----------------------------------------------------------------------
    numOnes:        Number of bits ON in each pattern
    numPatterns:    Number of unique patterns to generate
    patternOverlap: Number of bits of overlap between each successive pattern
    retval:         patterns
    """

    assert (patternOverlap < numOnes)

    # How many new bits are introduced in each successive pattern?
    numNewBitsInEachPattern = numOnes - patternOverlap
    numCols = numNewBitsInEachPattern * numPatterns + patternOverlap

    p = []
    for i in xrange(numPatterns):
        x = numpy.zeros(numCols, dtype='float32')

        startBit = i*numNewBitsInEachPattern
        nextStartBit = startBit + numOnes
        x[startBit:nextStartBit] = 1

        p.append(x)

    return p


#############################################################################
def buildOverlappedSequences( numSequences = 2,
                              seqLen = 5,
                              sharedElements = [3,4],
                              numOnBitsPerPattern = 3,
                              patternOverlap = 0,
                              seqOverlap = 0,
                              **kwargs
                              ):
    """ Create training sequences that share some elements in the middle.

    Parameters:
    -----------------------------------------------------
    numSequences:         Number of unique training sequences to generate
    seqLen:               Overall length of each sequence
    sharedElements:       Which element indices of each sequence are shared. These
                            will be in the range between 0 and seqLen-1
    numOnBitsPerPattern:  Number of ON bits in each TP input pattern
    patternOverlap:       Max number of bits of overlap between any 2 patterns
    retval:               (numCols, trainingSequences)
                            numCols - width of the patterns
                            trainingSequences - a list of training sequences

    """

    # Total number of patterns used to build the sequences
    numSharedElements = len(sharedElements)
    numUniqueElements = seqLen - numSharedElements
    numPatterns = numSharedElements + numUniqueElements * numSequences

    # Create the table of patterns
    patterns = getSimplePatterns(numOnBitsPerPattern, numPatterns, patternOverlap)

    # Total number of columns required
    numCols = len(patterns[0])


    # -----------------------------------------------------------------------
    # Create the training sequences
    trainingSequences = []

    uniquePatternIndices = range(numSharedElements, numPatterns)
    for _ in xrange(numSequences):
        sequence = []

        # pattern indices [0 ... numSharedElements-1] are reserved for the shared
        #  middle
        sharedPatternIndices = range(numSharedElements)

        # Build up the sequence
        for j in xrange(seqLen):
            if j in sharedElements:
              patIdx = sharedPatternIndices.pop(0)
            else:
              patIdx = uniquePatternIndices.pop(0)
            sequence.append(patterns[patIdx])

        trainingSequences.append(sequence)


        print "\nTraining sequences"
        printAllTrainingSequences(trainingSequences)

    return (numCols, trainingSequences)


def main(SEED):
    # input 生成
    numOnBitsPerPattern = 3
    (numCols, trainingSequences) = buildOverlappedSequences(
            numSequences        = 2,        # 生成するsequenceの数
            seqLen              = 5,        # sequenceの長さ
            sharedElements      = [2,3],    # 異なるsequence間で同じものが含まれている番号
            numOnBitsPerPattern = 3,        # activeになるカラム数
            patternOverlap      = 0         # activeになるカラムが重なっている数
            )


    print numCols
    for sequence in trainingSequences:
        print sequence


    # TP生成
    tp = TP(
            numberOfCols          = numCols,
            cellsPerColumn        = 2,
            initialPerm           = 0.6,
            connectedPerm         = 0.5,
            minThreshold          = 3,
            newSynapseCount       = 3,
            permanenceInc         = 0.1,
            permanenceDec         = 0.0,
            activationThreshold   = 3,
            globalDecay           = 0.0,
            burnIn                = 1,
            seed                  = SEED,
            verbosity             = 0,
            checkSynapseConsistency  = True,
            pamLength                = 1
            )

    # TP学習
    for _ in range(10):
        for seq_num, sequence in enumerate(trainingSequences):
            for x in sequence:
                x = numpy.array(x).astype('float32')
                tp.compute(x, enableLearn = True, computeInfOutput=True)
                #tp.printStates(False, False)
            tp.reset()


    # TP 予測
    for seq_num, sequence in enumerate(trainingSequences):
        for x in sequence:
            x = numpy.array(x).astype('float32')
            tp.compute(x, enableLearn = False, computeInfOutput = True)
            tp.printStates(False, False)




if __name__ == "__main__":
    main(SEED=35)

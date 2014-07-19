#!/usr/bin/python
# _*_ coding: utf-8 _*_

import numpy as np

from nupic.research.TP import TP
from nupic.research.TP10X2 import TP10X2
from nupic.support.unittesthelpers.testcasebase import (TestCaseBase,
                                                        TestOptionParser)


def _printOneTrainingVector(x):
  "Print a single vector succinctly."
  print ''.join('1' if k != 0 else '.' for k in x)

def _getSimplePatterns(numOnes, numPatterns):
  """Very simple patterns. Each pattern has numOnes consecutive
  bits on. There are numPatterns*numOnes bits in the vector. These patterns
  are used as elements of sequences when building up a training set."""

  numCols = numOnes * numPatterns
  p = []
  for i in xrange(numPatterns):
    x = np.zeros(numCols, dtype='float32')
    x[i*numOnes:(i + 1)*numOnes] = 1
    p.append(x)

  return p


def main(SEED, VERBOSITY):
    # TP 作成
    tp = TP(
            numberOfCols          = 100,
            cellsPerColumn        = 1,
            initialPerm           = 0.3,
            connectedPerm         = 0.5,
            minThreshold          = 4,
            newSynapseCount       = 7,
            permanenceInc         = 0.1,
            permanenceDec         = 0.05,
            activationThreshold   = 5,
            globalDecay           = 0,
            burnIn                = 1,
            seed                  = SEED,
            verbosity             = VERBOSITY,
            checkSynapseConsistency  = True,
            pamLength                = 1000
            )

    print
    trainingSet = _getSimplePatterns(10, 10)
    for seq in trainingSet[0:5]:
        _printOneTrainingVector(seq)


    # TP学習
    print
    print 'Learning 1 ... A->A->A'
    for _ in range(2):
        for seq in trainingSet[0:5]:
            for _ in range(10):
                #tp.learn(seq)
                tp.compute(seq, enableLearn = True, computeInfOutput=False)
            tp.reset()

    print
    print 'Learning 2 ... A->B->C'
    for _ in range(10):
        for seq in trainingSet[0:5]:
            tp.compute(seq, enableLearn = True, computeInfOutput=False)
        tp.reset()


    # TP 予測
    # Learning 1のみだと, A->Aを出力するのみだが,
    # その後, Learning 2もやると, A->A,Bを出力するようになる.　
    print
    print 'Running inference'
    for seq in trainingSet[0:5]:
        # tp.reset()
        # tp.resetStats()
        tp.compute(seq, enableLearn = False, computeInfOutput = True)
        tp.printStates(False, False)


if __name__ == "__main__":

    # ただのデフォルト設定の取得, unittestのときに使うように設定してあるみたい.
    parser = TestOptionParser()
    options, _ = parser.parse_args()
    SEED      = options.seed
    VERBOSITY = options.verbosity

    np.random.seed(SEED)

    main(SEED, VERBOSITY)


#!/usr/bin/python
# coding: utf-8

"""
How to use nupic Network api

https://github.com/numenta/nupic/wiki/NuPIC-API---A-bird's-eye-view
https://github.com/numenta/nupic.core
"""

import os
import csv
import copy
import json
import datetime

from nupic.algorithms.anomaly import computeAnomalyScore
from nupic.data.datasethelpers import findDataset
from nupic.data.file_record_stream import FileRecordStream
from nupic.engine import Network
from nupic.encoders import MultiEncoder


#_DATA_PATH = "./rec-center-hourly.csv"
_DATA_PATH = "./gym.csv"
_OUTPUT_PATH = "test_output.csv"

SP_PARAMS = {
    "spVerbosity": 0,
    "spatialImp": "cpp",
    "globalInhibition": 1,
    "columnCount": 2048,
    "inputWidth": 0,                # set later
    "numActiveColumnsPerInhArea": 20,
    "seed": 1956,
    "potentialPct": 0.8,
    "synPermConnected": 0.1,
    "synPermActiveInc": 0.0001,
    "synPermInactiveDec": 0.0005,
    "maxBoost": 2.0,
}

TP_PARAMS = {
    "verbosity": 0,
    "columnCount": 2048,
    "cellsPerColumn": 32,
    "inputWidth": 2048,
    "seed": 1960,
    "temporalImp": "cpp",
    "newSynapseCount": 20,
    "maxSynapsesPerSegment": 32,
    "maxSegmentsPerCell": 128,
    "initialPerm": 0.21,
    "permanenceInc": 0.1,
    "globalDecay": 0.0,
    "maxAge": 0,
    "minThreshold": 9,
    "activationThreshold": 12,
    "outputType": "normal",
    "pamLength": 1,
}

CLASSIFIER_PARAMS = {
    "clVerbosity": 2,
    'alpha': 0.005,
    "steps": '1'
}

def createEncoder():
    # TODO: vector
    encoder = MultiEncoder()
    encoder.addMultipleEncoders({
        "consumption": {
            "clipInput": True,
            "type": "ScalarEncoder",
            "fieldname": u"consumption",
            "name": u"consumption",
            "maxval": 100.0,
            "minval": 0.0,
            "n": 50,
            "w": 21,
            },
        "timestamp_timeOfDay": {
            "fieldname": u"timestamp",
            "name": u"timestamp_timeOfDay",
            "timeOfDay": (21, 9.5),
            "type": "DateEncoder",
            },
        })
    return encoder

def createNetwork(dataSource):
    """
    networkを作成する.
    sensor, sp, tp
    """
    network = Network()

    # create sensor region

    # create sensor region
    network.addRegion("sensor", "py.RecordSensor",
            json.dumps({"verbosity": 0}))
    sensor = network.regions["sensor"].getSelf()
    sensor.encoder = createEncoder()
    sensor.dataSource = dataSource


    # create spacial pooler region
    print sensor.encoder.getWidth()
    SP_PARAMS["inputWidth"] = sensor.encoder.getWidth()
    network.addRegion("SP", "py.SPRegion", json.dumps(SP_PARAMS))

    # linke sensor input <-> SP Region
    # Resion毎のinput/output名は, regions下の, SPRegion.py, TPRegion.py, RecordSensor.py
    network.link("sensor", "SP", "UniformLink", "")
    network.link("sensor", "SP", "UniformLink", "",
            srcOutput="resetOut", destInput="resetIn")
    network.link("SP", "sensor", "UniformLink", "",                          # これ, なくしても何も変化なかったけど..
            srcOutput="spatialTopDownOut", destInput="spatialTopDownIn")
    network.link("SP", "sensor", "UniformLink", "",                          # これ, なくしても何も変化なかったけど..
            srcOutput="temporalTopDownOut", destInput="temporalTopDownIn")

    # create temporal pooler region
    network.addRegion("TP", "py.TPRegion",
            json.dumps(TP_PARAMS))

    network.link("SP", "TP", "UniformLink", "")
    network.link("TP", "SP", "UniformLink", "",                              # これ, なくしても何も変化なかったけど..
            srcOutput="topDownOut", destInput="topDownIn")

    # create classifier
    network.addRegion("Classifier", "py.CLAClassifierRegion",
            json.dumps(CLASSIFIER_PARAMS))

    network.link("TP", "Classifier", "UniformLink", "")
    network.link("sensor", "Classifier", "UniformLink", "",
            srcOutput="categoryOut", destInput="categoryIn")


    # initialize
    network.initialize()

    # setting sp
    SP = network.regions["SP"]
    SP.setParameter("learningMode", True)
    SP.setParameter("anomalyMode", True)

    # setting tp
    TP = network.regions["TP"]
    TP.setParameter("topDownMode", False)
    TP.setParameter("learningMode", True)
    TP.setParameter("inferenceMode", True)

    # OPFでやってるみたいな, AnomalyClassifierを追加するやり方とちがうのか.
    TP.setParameter("anomalyMode", False)

    # classifier regionを定義.
    classifier = network.regions["Classifier"]
    classifier.setParameter('inferenceMode', True)
    classifier.setParameter('learningMode', True)


    return network


def main():
    def calc(sensor, sp, tp):
        sensor.prepareInputs()
        sensor.compute()
        sp.prepareInputs()
        sp.compute()
        tp.prepareInputs()
        tp.compute()
    # ファイル渡すだけなら, 絶対パスに変換しているだけ.
    #trainFile = findDataset(_DATA_PATH)
    trainFile = os.path.abspath(_DATA_PATH)
    print trainFile

    # TODO: sensor layerには, dataSourceの形で渡す必要がある.
    # 後から渡すような形にするためには, DataBuffer@nupic/frameworks/opf/clamodel.py のように,
    # getNextRecordDict methodを持った, instanceを作れば良いのかな?
    dataSource = FileRecordStream(streamID=trainFile)
    # print dataSource
    # for i in range(2000):
    #     print dataSource.getNextRecordDict()

    network = createNetwork(dataSource)

    #
    sensorRegion = network.regions["sensor"]
    SPRegion     = network.regions["SP"]
    TPRegion     = network.regions["TP"]
    classifier   = network.regions["Classifier"]

    prevPredictedColumns = []


    # TODO: ここの内容, OPFとの違いをちゃんと押さえる.
    for calc_num in xrange(10000):
        # TODO: このrunって実際何をやってるんだ? どこに書いてあるか探す.
        #       多分, 各Regionの_computeでやってることをまとめて実行してるのだと思う.
        network.run(1)
        #calc(sensorRegion, SPRegion, TPRegion)

        #
        print
        print "####################################"
        print
        print "==== EC layer ===="
        consumption = sensorRegion.getOutputData("sourceOut")[0]
        print 'sourceOut',consumption
        categoryOut = sensorRegion.getOutputData("categoryOut")[0]
        print 'categoryOut',categoryOut
        # spatialTopDownOut = sensorRegion.getOutputData("spatialTopDownOut")[0]
        # print 'spatialTopDownOut',spatialTopDownOut
        # temporalTopDownOut = sensorRegion.getOutputData("temporalTopDownOut")[0]
        # print 'temporalTopDownOut',temporalTopDownOut

        # SPへの入力(encode後)
        print
        print "==== SP layer ===="
        sp_bottomUpIn  = SPRegion.getInputData("bottomUpIn").nonzero()[0]
        print 'bottomUpIn', sp_bottomUpIn[:10]
        sp_bottomUpOut = SPRegion.getOutputData("bottomUpOut").nonzero()[0]
        print 'bottomUpOut', sp_bottomUpOut[:10]
        # sp_input  = SPRegion.getInputData("topDownIn").nonzero()[0]
        # print 'topDownIn',sp_input[:10]
        # topDownOut = SPRegion.getOutputData("topDownOut").nonzero()[0]
        # print 'topDownOut', topDownOut[:10]
        # spatialTopDownOut = SPRegion.getOutputData("spatialTopDownOut").nonzero()[0]
        # print 'spatialTopDownOut', spatialTopDownOut[:10]
        # temporalTopDownOut = SPRegion.getOutputData("temporalTopDownOut").nonzero()[0]
        # print 'temporalTopDownOut', temporalTopDownOut[:10]
        # anomalyScore = SPRegion.getOutputData("anomalyScore").nonzero()[0]
        # print 'anomalyScore', anomalyScore[:10]

        # anomaly

        # TPへのinput前
        print
        print "==== TP layer ===="
        # TPへのinput後 ※ topdownmdoeをfalseにしたらできたが..
        tp_bottomUpIn = TPRegion.getInputData("bottomUpIn").nonzero()[0]
        print 'bottomUpIn', sorted(tp_bottomUpIn)[:10]
        tp_bottomUpOut = TPRegion.getOutputData("bottomUpOut").nonzero()[0]
        print 'bottomUpOut', tp_bottomUpOut[:10]
        tp_topDownOut = TPRegion.getOutputData("topDownOut").nonzero()[0]
        print 'topDownOut', tp_topDownOut[:10]
        # predictedColumns = TPRegion.getOutputData("lrnActiveStateT").nonzero()[0]
        # print 'lrnActiveStateT',predictedColumns[:10]
        # predictedColumns = TPRegion.getOutputData("anomalyScore").nonzero()[0]
        # print 'anomalyScore',predictedColumns[:10]

        #
        print
        print "==== Anomaly ===="
        print 'sp_bottomUpOut', sp_bottomUpOut[:10]
        print 'prevPredictedColumns', prevPredictedColumns[:10]
        anomalyScore = computeAnomalyScore(sp_bottomUpOut, prevPredictedColumns)
        print 'anomalyScore', anomalyScore

        print
        print "==== Predict ===="
        categoryIn = classifier.getInputData("categoryIn").nonzero()[0]
        print 'categoryIn', categoryIn[:10]
        cl_bottomUpIn = classifier.getInputData("bottomUpIn").nonzero()[0]
        print 'bottomUpIn', cl_bottomUpIn[:10]
        enc_list  = sensorRegion.getSelf().encoder.getEncoderList()
        bucketIdx = enc_list[0].getBucketIndices(consumption)[0]
        classificationIn = {
                'bucketIdx': bucketIdx,
                'actValue': float(consumption)
                }
        clResults = classifier.getSelf().customCompute(
                recordNum=calc_num,
                patternNZ=tp_bottomUpOut,
                classification=classificationIn
                )
        max_index = [i for i, j in enumerate(clResults[1] ) if j == max(clResults[1] )]
        print 'predict value: ',clResults['actualValues'][max_index[0]]




        print
        print "==== Other ===="
        # TODO: topdownの予測とclassifierの予測両方試す.
        # 下記, ex.1, 2, 3 どれでも予測できていない. むしろ別の部分を疑うべきか?

        # ex.1: sample
        predictedColumns = TPRegion.getOutputData("topDownOut").nonzero()[0]
        print 'ex.1 topDownOut', predictedColumns[:10]
        #prevPredictedColumns = copy.deepcopy(predictedColumns)

        # ex.2: l.506@nupic/regions/TPRegion.py 修正版
        #       l.506@nupic/regions/TPRegion.py間違ってないか?
        #       self._tfdr.topDownCompute() -> self.getSelf()._tfdr.topDownCompute()
        topdown_predict = TPRegion.getSelf()._tfdr.topDownCompute().copy().nonzero()[0]
        print 'ex.2 topDownCompute', sorted(topdown_predict)[:10]
        prevPredictedColumns = copy.deepcopy(topdown_predict)


        # ex.3: opf  , これはあくまで, TPのoutputじゃないか?
        tpOutput = TPRegion.getSelf()._tfdr.infActiveState['t'].nonzero()[0]
        print 'ex.3 infActiveState', tpOutput[:10]

        # ex.4: l.509@nupic/regions/TPRegion.py 修正版
        #       l.509@nupic/regions/TPRegion.py間違ってないか?
        #       ex.2と同じ. getSelf()が抜けている.
        activeLearnCells = TPRegion.getSelf()._tfdr.getLearnActiveStateT()
        size = activeLearnCells.shape[0] * activeLearnCells.shape[1]
        print 'ex.4 anomalyscore', activeLearnCells.reshape(size).nonzero()[0]




    # # 結果保存
    # outputPath = os.path.join(os.path.dirname(__file__), _OUTPUT_PATH)
    # with open(outputPath, "w") as outputFile:
    #     writer = csv.writer(outputFile)
    #     print "Writing output to %s" % outputPath


if __name__ == "__main__":
    main()


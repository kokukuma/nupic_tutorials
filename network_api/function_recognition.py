#!/usr/bin/python
# coding: utf-8

from pprint import pprint

class function_data(object):
    """
    sample:
        fd = function_data()
        ftype = fd.romdom_choice()
        data = fd.get_data(ftype)

        print ftype
        for d in data:
            print d
    """
    def __init__(self):
        import numpy
        """
        x = 0 - 100
        y = 0 - 100
        """
        self.max_x = 100
        self.function_list = {
                #'flat':    lambda x: 50,
                'plus':    lambda x: float(x),
                'minus':    lambda x: 100-float(x),
                # 'sin':       lambda x: numpy.sin(x * 4 * numpy.pi/self.max_x) * 50 + 50,
                # 'linear':    lambda x: float(x),
                # 'quadratic': lambda x: float(x*x)/self.max_x,
                # 'step':      lambda x: 100.0 if int(float(x)/15) % 2 == 0 else 0.0
                }

    def romdom_choice(self):
        import random
        ftype = random.choice(self.function_list.keys())
        return ftype

    def get_data(self, ftype):
        if ftype not in self.function_list.keys():
            return []
        result = []
        for x in range(self.max_x):
            y = self.function_list[ftype](x)
            result.append([float(x), y])
        return result


class FunctionRecogniter():

    def __init__(self):
        self.classifier_encoder_list = {}
        self.run_number = 0
        self.prevPredictedColumns = None
        self._createNetwork()

    def _createNetwork(self):
        from nupic.algorithms.anomaly import computeAnomalyScore
        from nupic.engine import Network
        import create_network as cn
        import json


        self.network = Network()

        # sensor
        self.network.addRegion("sensor", "py.RecordSensor",
                json.dumps({"verbosity": 0}))
        sensor = self.network.regions["sensor"].getSelf()
        sensor.encoder         = cn.createVectorEncoder()
        #sensor.disabledEncoder = cn.createCategoryEncoder(['sin', 'linear', 'quadratic', 'step'])
        #sensor.disabledEncoder = cn.createCategoryEncoder(['plus', 'minus', 'flat'])
        #sensor.disabledEncoder = cn.createScalarEncoder()

        sensor.dataSource      = cn.DataBuffer()

        # sp
        cn.SP_PARAMS["inputWidth"] = sensor.encoder.getWidth()
        self.network.addRegion("SP", "py.SPRegion", json.dumps(cn.SP_PARAMS))

        self.network.link("sensor", "SP", "UniformLink", "")
        self.network.link("sensor", "SP", "UniformLink", "",
                srcOutput="resetOut", destInput="resetIn")

        # tp
        self.network.addRegion("TP", "py.TPRegion",
                json.dumps(cn.TP_PARAMS))

        self.network.link("SP", "TP", "UniformLink", "")
        # self.network.link("TP", "SP", "UniformLink", "",
        #         srcOutput="topDownOut", destInput="topDownIn")

        # # secound sp/tp layer
        # cn.SP_PARAMS["inputWidth"] = 512 * 32
        # cn.SP_PARAMS["columnCount"] = 100
        # self.network.addRegion("SP2", "py.SPRegion", json.dumps(cn.SP_PARAMS))
        # self.network.link("TP", "SP2", "UniformLink", "")
        #
        # cn.TP_PARAMS["inputWidth"] = 100
        # cn.TP_PARAMS["cellsPerColumn"] = 10
        # self.network.addRegion("TP2", "py.TPRegion", json.dumps(cn.TP_PARAMS))
        # self.network.link("SP2", "TP2", "UniformLink", "")
        # self.network.link("TP2", "SP2", "UniformLink", "",
        #         srcOutput="topDownOut", destInput="topDownIn")


        # classifier
        cn.CLASSIFIER_PARAMS['steps'] = '0'
        self.network.addRegion("Classifier", "py.CLAClassifierRegion",
                json.dumps(cn.CLASSIFIER_PARAMS))
        self.network.link("TP", "Classifier", "UniformLink", "")
        # self.network.link("sensor", "Classifier", "UniformLink", "",
        #         srcOutput="categoryOut", destInput="categoryIn")
        self.classifier_encoder_list["Classifier"]  = cn.createCategoryEncoder(['plus', 'minus', 'flat'])


        cn.CLASSIFIER_PARAMS['steps'] = '1'
        self.network.addRegion("Classifier_y", "py.CLAClassifierRegion",
                json.dumps(cn.CLASSIFIER_PARAMS))
        self.network.link("TP", "Classifier_y", "UniformLink", "")

        self.classifier_encoder_list["Classifier_y"] = cn.createScalarEncoder()



        # initialize
        self.network.initialize()

        # setting sp
        SP = self.network.regions["SP"]
        SP.setParameter("learningMode", True)
        SP.setParameter("anomalyMode", True)

        # setting tp
        TP = self.network.regions["TP"]
        TP.setParameter("topDownMode", False)
        TP.setParameter("learningMode", True)
        TP.setParameter("inferenceMode", True)
        TP.setParameter("anomalyMode", False)

        # classifier regionを定義.
        classifier = self.network.regions["Classifier"]
        classifier.setParameter('inferenceMode', True)
        classifier.setParameter('learningMode', True)

        classifier_y = self.network.regions["Classifier_y"]
        classifier_y.setParameter('inferenceMode', True)
        classifier_y.setParameter('learningMode', True)


        # # setting secound layer
        # SP2 = self.network.regions["SP2"]
        # SP2.setParameter("learningMode", True)
        # SP2.setParameter("anomalyMode", True)
        # TP2 = self.network.regions["TP2"]
        # TP2.setParameter("topDownMode", False)
        # TP2.setParameter("learningMode", True)
        # TP2.setParameter("inferenceMode", True)
        # TP2.setParameter("anomalyMode", False)


        return

    def layer_output(self, input_data):
        sensorRegion = self.network.regions["sensor"]
        SPRegion = self.network.regions["SP"]
        TPRegion = self.network.regions["TP"]
        # SP2Region = self.network.regions["SP2"]
        # TP2Region = self.network.regions["TP2"]
        print
        print "####################################"
        print
        print "==== Input ===="
        print input_data['xy_value']
        print
        print "==== EC layer ===="
        print "output:     ", sensorRegion.getOutputData("dataOut").nonzero()[0][:10]
        print
        print "==== SP layer ===="
        print "input:  ", SPRegion.getInputData("bottomUpIn").nonzero()[0][:10]
        print "output: ", SPRegion.getOutputData("bottomUpOut").nonzero()[0][:10]
        print
        print "==== TP layer ===="
        print "input:  ", TPRegion.getInputData("bottomUpIn").nonzero()[0][:10]
        print "output: ", TPRegion.getOutputData("bottomUpOut").nonzero()[0][:10]
        print
        # print "==== SP2 layer ===="
        # print "input:  ", SP2Region.getInputData("bottomUpIn").nonzero()[0][:10]
        # print "output: ", SP2Region.getOutputData("bottomUpOut").nonzero()[0][:10]
        # print
        # print "==== TP2 layer ===="
        # print "input:  ", TP2Region.getInputData("bottomUpIn").nonzero()[0]
        # print "output: ", TP2Region.getOutputData("bottomUpOut").nonzero()[0]
        print
        print "==== Predict ===="
        print TPRegion.getSelf()._tfdr.topDownCompute().copy().nonzero()[0][:10]
        print

    def debug(self, input_data):
        TPRegion = self.network.regions["TP"]
        tp_output = TPRegion.getOutputData("bottomUpOut").nonzero()[0]

        if 23263 in tp_output:
            print input_data


    def learn(self, input_data):
        """
        input_data = {'xy_value': [1.0, 2.0], 'ftype': 'sin'}
        """
        # TODO: learn/predict, やってること同じだから一緒にする.
        #       学習するかしないか. classifierに値渡すか渡さないか違いのみ.

        self.enable_learning_mode(True)
        self.run_number += 1

        # calc encoder, SP, TP
        self.network.regions["sensor"].getSelf().dataSource.push(input_data)
        self.network.run(1)
        #self.layer_output(input_data)


        # learn classifier
        inferences = {}
        inferences['Classifier']   = self._learn_classifier_multi("Classifier", actValue=input_data['ftype'], pstep=0)
        inferences['Classifier_y'] = self._learn_classifier_multi("Classifier_y", input_data['xy_value'][1], pstep=1)

        # anomaly
        inferences["anomaly"] = self._calc_anomaly()

        # print input_data['xy_value'], inferences['Classifier_y']['best']['value'], inferences['Classifier_y']['best']['prob']
        # print 'actual value: ',input_data['xy_value'],
        # print input_data['ftype'],
        # print inferences['Classifier']['best']['value'],
        # print inferences["anomaly"],
        # print dict(inferences['Classifier']['likelihoodsDict'])

        print "%10s, %10s, %10.6f, %10.6f, %5s, %5s" % (
                input_data['xy_value'][0],
                input_data['xy_value'][1],
                inferences['Classifier']['likelihoodsDict']['plus'],
                inferences['Classifier']['likelihoodsDict']['minus'],
                input_data['ftype'], inferences['Classifier']['best']['value'])

        return inferences


    def predict(self, input_data):
        # calc encoder, SP, TP
        self.enable_learning_mode(False)
        self.run_number += 1

        self.network.regions["sensor"].getSelf().dataSource.push(input_data)
        self.network.run(1)
        if input_data["xy_value"] == [50.0, 50.0]:
        #if input_data["xy_value"][0] == 10.0:
            self.layer_output(input_data)
        #self.debug(input_data)

        # learn classifier
        inferences = {}
        inferences['Classifier']   = self._learn_classifier_multi("Classifier", actValue=None, pstep=0)
        inferences['Classifier_y'] = self._learn_classifier_multi("Classifier_y", actValue=None, pstep=1)

        # anomaly
        inferences["anomaly"] = self._calc_anomaly()

        return inferences

    def _learn_classifier_multi(self, region_name, actValue=None, pstep=0):

        # TODO: これはnetworkの中で計算できないのか?
        #       regions/CLAClassifierRegion.py の getSpec/outputにもないし, 無理そうだな.

        classifier     = self.network.regions[region_name]
        encoder        = self.classifier_encoder_list[region_name].getEncoderList()[0]
        tp_bottomUpOut = self.network.regions["TP"].getOutputData("bottomUpOut").nonzero()[0]
        #tp_bottomUpOut = self.network.regions["TP"].getSelf()._tfdr.infActiveState['t'].reshape(-1).nonzero()[0]

        if actValue is not None:
            bucketIdx = encoder.getBucketIndices(actValue)[0]
            classificationIn = {
                    'bucketIdx': bucketIdx,
                    'actValue': actValue
                    }
        else:
            classificationIn = {'bucketIdx': 0,'actValue': 'no'}
        clResults = classifier.getSelf().customCompute(
                recordNum=self.run_number,
                patternNZ=tp_bottomUpOut,
                classification=classificationIn
                )

        inferences= self._summay_clresult(clResults, pstep)
        return inferences

    def _summay_clresult(self, clResults, steps):
        from collections import defaultdict

        likelihoodsVec = clResults[steps]
        bucketValues   = clResults['actualValues']

        likelihoodsDict = defaultdict(int)
        bestActValue = None
        bestProb = None

        for (actValue, prob) in zip(bucketValues, likelihoodsVec):
            likelihoodsDict[actValue] += prob
            if bestProb is None or likelihoodsDict[actValue] > bestProb:
                bestProb = likelihoodsDict[actValue]
                bestActValue = actValue

        return {'likelihoodsDict': likelihoodsDict, 'best': {'value': bestActValue, 'prob':bestProb}}


    def _calc_anomaly(self):
        import copy
        from nupic.algorithms.anomaly import computeAnomalyScore

        anomalyScore = None
        sp_bottomUpOut = self.network.regions["SP"].getOutputData("bottomUpOut").nonzero()[0]
        if self.prevPredictedColumns is not None:
            anomalyScore = computeAnomalyScore(sp_bottomUpOut, self.prevPredictedColumns)
        #topdown_predict = self.network.regions["TP"].getSelf()._tfdr.topDownCompute().copy().nonzero()[0]
        topdown_predict = self.network.regions["TP"].getSelf()._tfdr.topDownCompute().nonzero()[0]
        self.prevPredictedColumns = copy.deepcopy(topdown_predict)

        return anomalyScore

    def reset(self):
        """
        reset sequence
        """
        self.network.regions["TP"].getSelf().resetSequenceStates()
        #self.network.regions["TP2"].getSelf().resetSequenceStates()

    def enable_learning_mode(self, enable):
        self.network.regions["SP"].setParameter("learningMode", enable)
        self.network.regions["TP"].setParameter("learningMode", enable)
        self.network.regions["Classifier"].setParameter("learningMode", enable)

        # self.network.regions["SP2"].setParameter("learningMode", enable)
        # self.network.regions["TP2"].setParameter("learningMode", enable)


def main():

    fd = function_data()
    recogniter = FunctionRecogniter()

    # トレーニング
    for i in range(50):
        print i,
        for num, ftype in enumerate(fd.function_list.keys()):
            data = fd.get_data(ftype)
            for x, y in data:
                input_data = {
                        'xy_value': [x, y],
                        'ftype': ftype
                        }
                inferences = recogniter.learn(input_data)

            recogniter.reset()


    # TODO: 合わない原因を考える.
    # TODO: 本当にclassifierの計算だけ別途行う必要があるのか? runを読んで調べる.
    # TODO: TP->SPと接続したら, 接続元はcellなのかcolumnなのか.

    # TODO: 複数の層対応


    # 予測1
    for ftype in fd.function_list.keys():
        print ftype
        data = fd.get_data(ftype)
        for x, y in data:
            input_data = {
                    'xy_value': [x, y]
                    }
            inferences = recogniter.predict(input_data)
            print "%10s, %10s, %10.6f, %10.6f, %5s, %5s" % (
                    x, y,
                    inferences['Classifier']['likelihoodsDict']['plus'],
                    inferences['Classifier']['likelihoodsDict']['minus'],
                    ftype, inferences['Classifier']['best']['value']
                    )


    # # 予測2, fixed-sin
    # import numpy
    # print
    # print "fiexed-sin"
    # fd.function_list['sin'] = lambda x: numpy.sin(x * 2 * numpy.pi/fd.max_x) * 50 + 50
    # data = fd.get_data('sin')
    # for x, y in data:
    #     input_data = {
    #             'xy_value': [x, y]
    #             }
    #     inferences = recogniter.predict(input_data)
    #     print 'sin', inferences['best']['value'], inferences['best']['prob'], inferences["anomaly"]
    #
    #
    # # 予測3, fixed-sin
    # print
    # print "fiexed-sin"
    # fd.function_list['sin'] = lambda x: numpy.sin(x * 4 * numpy.pi/fd.max_x) * 30 + 50
    # data = fd.get_data('sin')
    # for x, y in data:
    #     input_data = {
    #             'xy_value': [x, y]
    #             }
    #     inferences = recogniter.predict(input_data)
    #     print 'sin', inferences['best']['value'], inferences['best']['prob'], inferences["anomaly"]

if __name__ == "__main__":
    main()

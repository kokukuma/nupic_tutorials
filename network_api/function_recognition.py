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
                'flat':    lambda x: 50.0,
                'plus':    lambda x: float(x),
                'minus':    lambda x: 100-float(x),
                #'sin':       lambda x: numpy.sin(x * 4 * numpy.pi/self.max_x) * 50 + 50,
                # 'linear':    lambda x: float(x),
                # 'quad': lambda x: float(x*x)/self.max_x,
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
        from collections import OrderedDict

        self.classifier_encoder_list = {}
        self.classifier_input_list = {}
        self.run_number = 0
        self.prevPredictedColumns = {}

        # net structure
        self.net_structure = OrderedDict()
        self.net_structure['sensor1'] = ['region1']
        self.net_structure['sensor2'] = ['region2']
        self.net_structure['sensor3'] = ['region3']

        self.net_structure['region1'] = ['region4']
        self.net_structure['region2'] = ['region4']
        self.net_structure['region3'] = ['region4']

        # sensor change params
        self.sensor_params = {
                'sensor1': {
                    'xy_value': {
                        'maxval': 60.0,
                        'minval':  0.0
                        },
                    },
                'sensor2': {
                    'xy_value': {
                        'maxval': 80.0,
                        'minval': 20.0
                        },
                    },
                'sensor3': {
                    'xy_value': {
                        'maxval': 100.0,
                        'minval':  40.0
                        },
                    },
                }

        # region change params
        self.dest_resgion_data = {
                'region1': {
                    'TP_PARAMS':{
                        "cellsPerColumn": 8
                        },
                    },
                'region2': {
                    'TP_PARAMS':{
                        "cellsPerColumn": 8
                        },
                    },
                'region3': {
                    'TP_PARAMS':{
                        "cellsPerColumn": 8
                        },
                    },
                'region4': {
                    'SP_PARAMS':{
                        "inputWidth": 2024 * (8 + 8 + 8)
                        },
                    'TP_PARAMS':{
                        "cellsPerColumn": 16
                        },
                    },
                 }

        self._createNetwork()

    def _addRegion(self, src_name, dest_name, params):
        import json
        from nupic.encoders import MultiEncoder

        sensor     =  src_name
        sp_name    = "sp_" + dest_name
        tp_name    = "tp_" + dest_name
        class_name = "class_" + dest_name

        try:
            self.network.regions[sp_name]
            self.network.regions[tp_name]
            self.network.regions[class_name]

            self.network.link(sensor, sp_name, "UniformLink", "")

        except Exception as e:
            # sp
            self.network.addRegion(sp_name, "py.SPRegion", json.dumps(params['SP_PARAMS']))
            self.network.link(sensor, sp_name, "UniformLink", "")

            # tp
            self.network.addRegion(tp_name, "py.TPRegion", json.dumps(params['TP_PARAMS']))
            self.network.link(sp_name, tp_name, "UniformLink", "")

            # class
            self.network.addRegion( class_name, "py.CLAClassifierRegion", json.dumps(params['CLASSIFIER_PARAMS']))
            self.network.link(tp_name, class_name, "UniformLink", "")

            encoder = MultiEncoder()
            encoder.addMultipleEncoders(params['CLASSIFIER_ENCODE_PARAMS'])
            self.classifier_encoder_list[class_name]  = encoder
            self.classifier_input_list[class_name]    = tp_name

    def _initRegion(self, name):
        sp_name = "sp_"+ name
        tp_name = "tp_"+ name
        class_name = "class_"+ name

        # setting sp
        SP = self.network.regions[sp_name]
        SP.setParameter("learningMode", True)
        SP.setParameter("anomalyMode", True)

        # setting tp
        TP = self.network.regions[tp_name]
        TP.setParameter("topDownMode", False)
        TP.setParameter("learningMode", True)
        TP.setParameter("inferenceMode", True)
        TP.setParameter("anomalyMode", False)

        # classifier regionを定義.
        classifier = self.network.regions[class_name]
        classifier.setParameter('inferenceMode', True)
        classifier.setParameter('learningMode', True)


    def _createNetwork(self):

        def deepupdate(original, update):
            """
            Recursively update a dict.
            Subdict's won't be overwritten but also updated.
            """
            for key, value in original.iteritems():
                if not key in update:
                    update[key] = value
                elif isinstance(value, dict):
                    deepupdate(value, update[key])
            return update


        from nupic.algorithms.anomaly import computeAnomalyScore
        from nupic.encoders import MultiEncoder
        from nupic.engine import Network
        import create_network as cn
        import json
        import itertools


        self.network = Network()

        # sensor
        for sensor_name, change_params in self.sensor_params.items():
            self.network.addRegion(sensor_name, "py.RecordSensor", json.dumps({"verbosity": 0}))
            sensor = self.network.regions[sensor_name].getSelf()

            # set encoder
            params = deepupdate(cn.SENSOR_PARAMS, change_params)
            encoder = MultiEncoder()
            encoder.addMultipleEncoders( params )
            sensor.encoder         = encoder

            # set datasource
            sensor.dataSource      = cn.DataBuffer()


        # network
        print 'create network ...'
        for source, dest_list in self.net_structure.items():
            for dest in dest_list:
                change_params = self.dest_resgion_data[dest]
                params = deepupdate(cn.PARAMS, change_params)

                if source in self.sensor_params.keys():
                    sensor = self.network.regions[source].getSelf()
                    params['SP_PARAMS']['inputWidth'] = sensor.encoder.getWidth()
                    self._addRegion(source, dest, params)
                else:
                    self._addRegion("tp_" + source, dest, params)

        # initialize
        print 'initializing network ...'
        self.network.initialize()
        for name in set( itertools.chain.from_iterable( self.net_structure.values() )):
            self._initRegion(name)


        # TODO: 1-3-1構造で, TPのセル数をむやみに増やすことは逆効果になるのでは?

        return

    def layer_output(self, input_data):
        sensorRegion = self.network.regions["sensor"]
        SPRegion = self.network.regions["SP"]
        TPRegion = self.network.regions["TP"]
        SP2Region = self.network.regions["SP2"]
        TP2Region = self.network.regions["TP2"]
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
        print "==== SP2 layer ===="
        print "input:  ", SP2Region.getInputData("bottomUpIn").nonzero()[0][:10]
        print "output: ", SP2Region.getOutputData("bottomUpOut").nonzero()[0][:10]
        print
        print "==== TP2 layer ===="
        print "input:  ", TP2Region.getInputData("bottomUpIn").nonzero()[0]
        print "output: ", TP2Region.getOutputData("bottomUpOut").nonzero()[0]
        print
        print "==== Predict ===="
        print TPRegion.getSelf()._tfdr.topDownCompute().copy().nonzero()[0][:10]
        print

    def debug(self, input_data):
        TPRegion = self.network.regions["TP2"]
        tp_output = TPRegion.getOutputData("bottomUpOut").nonzero()[0]
        #print tp_output

        if 5 in tp_output:
            print input_data['xy_value']

    def print_inferences(self, input_data, inferences):
        import itertools

        print "%10s, %10s, %5s" % (
                int(input_data['xy_value'][0]),
                int(input_data['xy_value'][1]),
                input_data['ftype']),

        for name in set( itertools.chain.from_iterable( self.net_structure.values() )):
            print "%5s," % (inferences['classifier_'+name]['best']['value']),

        for name in set( itertools.chain.from_iterable( self.net_structure.values() )):
            print "%10.6f," % (inferences['classifier_'+name]['likelihoodsDict'][input_data['ftype']]),

        for name in set( itertools.chain.from_iterable( self.net_structure.values() )):
            print "%5s," % (str(inferences["anomaly"][name])),
        print

    def run(self, input_data, learn=True):
        """
        input_data = {'xy_value': [1.0, 2.0], 'ftype': 'sin'}
        """
        import itertools

        self.enable_learning_mode(learn)
        self.run_number += 1

        # calc encoder, SP, TP
        for sensor_name in self.sensor_params.keys():
            self.network.regions[sensor_name].getSelf().dataSource.push(input_data)
        self.network.run(1)
        #self.layer_output(input_data)
        #self.debug(input_data)


        # learn classifier
        inferences = {}
        for name in set( itertools.chain.from_iterable( self.net_structure.values() )):
            class_name = "class_" + name
            inferences['classifier_'+name]   = self._learn_classifier_multi(class_name, actValue=input_data['ftype'], pstep=0)

        # anomaly
        inferences["anomaly"] = self._calc_anomaly()


        return inferences


    def _learn_classifier_multi(self, region_name, actValue=None, pstep=0):

        # TODO: これはnetworkの中で計算できないのか?
        #       regions/CLAClassifierRegion.py の getSpec/outputにもないし, 無理そうだな.

        classifier     = self.network.regions[region_name]
        encoder        = self.classifier_encoder_list[region_name].getEncoderList()[0]
        class_input    = self.classifier_input_list[region_name]
        tp_bottomUpOut = self.network.regions[class_input].getOutputData("bottomUpOut").nonzero()[0]
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

        inferences= self._summay_clresult(clResults, pstep, summary_tyep='sum')
        return inferences

    def _summay_clresult(self, clResults, steps, summary_tyep='sum'):
        from collections import defaultdict

        likelihoodsVec = clResults[steps]
        bucketValues   = clResults['actualValues']

        likelihoodsDict = defaultdict(int)
        bestActValue = None
        bestProb = None

        if summary_tyep == 'sum':
            for (actValue, prob) in zip(bucketValues, likelihoodsVec):
                likelihoodsDict[actValue] += prob
                if bestProb is None or likelihoodsDict[actValue] > bestProb:
                    bestProb = likelihoodsDict[actValue]
                    bestActValue = actValue

        elif summary_tyep == 'best':
            for (actValue, prob) in zip(bucketValues, likelihoodsVec):
                if bestProb is None or prob > bestProb:
                    likelihoodsDict[actValue] = prob
                    bestProb = prob
                    bestActValue = actValue

        return {'likelihoodsDict': likelihoodsDict, 'best': {'value': bestActValue, 'prob':bestProb}}


    def _calc_anomaly(self):
        import copy
        import itertools
        from nupic.algorithms.anomaly import computeAnomalyScore

        score = 0
        anomalyScore = {}
        for name in set( itertools.chain.from_iterable( self.net_structure.values() )):
            sp_bottomUpOut = self.network.regions["sp_"+name].getOutputData("bottomUpOut").nonzero()[0]
            if self.prevPredictedColumns.has_key(name):
                score = computeAnomalyScore(sp_bottomUpOut, self.prevPredictedColumns[name])
            #topdown_predict = self.network.regions["TP"].getSelf()._tfdr.topDownCompute().copy().nonzero()[0]
            topdown_predict = self.network.regions["tp_"+name].getSelf()._tfdr.topDownCompute().nonzero()[0]
            self.prevPredictedColumns[name] = copy.deepcopy(topdown_predict)

            anomalyScore[name] = score

        return anomalyScore

    def reset(self):
        """
        reset sequence
        """
        import itertools
        for name in set( itertools.chain.from_iterable( self.net_structure.values() )):
            self.network.regions["tp_"+name].getSelf().resetSequenceStates()

    def enable_learning_mode(self, enable):
        import itertools
        for name in set( itertools.chain.from_iterable( self.net_structure.values() )):
            self.network.regions["sp_"+name].setParameter("learningMode", enable)
            self.network.regions["tp_"+name].setParameter("learningMode", enable)
            self.network.regions["class_"+name].setParameter("learningMode", enable)

def main():

    fd = function_data()
    recogniter = FunctionRecogniter()

    # トレーニング
    for i in range(100):
        print i,
        for num, ftype in enumerate(fd.function_list.keys()):
            data = fd.get_data(ftype)
            for x, y in data:
                input_data = {
                        'xy_value': [x, y],
                        'ftype': ftype
                        }
                inferences = recogniter.run(input_data, learn=True)

                # print
                recogniter.print_inferences(input_data, inferences)

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
                    'xy_value': [x, y],
                    'ftype': None
                    }
            inferences = recogniter.run(input_data, learn=False)

            # print
            input_data['ftype'] = ftype
            recogniter.print_inferences(input_data, inferences)

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

#!/usr/bin/python
# coding: utf-8

from pprint import pprint
from pylab import *
from collections import defaultdict

import matplotlib.pyplot as plt

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
                'flat':  lambda x: 50.0,
                'plus':  lambda x: float(x),
                'minus': lambda x: 100-float(x),
                # 'sin':   lambda x: numpy.sin(x *  4 * numpy.pi/self.max_x) * 50 + 50,
                # 'quad':  lambda x: float(x*x)/self.max_x,
                # 'step':  lambda x: 100.0 if int(float(x)/15) % 2 == 0  else 0.0
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


class NetworkEvaluation(object):
    def __init__(self):
        from collections import defaultdict
        self.cell_activity = defaultdict(lambda: defaultdict(int))

    def get_fired_rate(self):
        """
        セルの平均発火数/STD
        """
        import numpy
        fire_count  = []
        for cell, activity in self.cell_activity.items():
            fire_count.append(sum([x for x in activity.values()]))
        return numpy.mean(fire_count) , numpy.std(fire_count)

    def get_selectivity(self):
        """
        選択性
        """
        import numpy
        from collections import defaultdict
        selectivity  = defaultdict(lambda: defaultdict(int))

        mean, _ = self.get_fired_rate()

        for cell, activity in self.cell_activity.items():
            if sum(activity.values()) >= mean:
                for label, data in activity.items():
                    if not data == 0:
                        select_value = float(data) / sum(activity.values())
                        selectivity[label][int(select_value * 100)] += 1

        result  = defaultdict(lambda: defaultdict(list))
        for label, data in selectivity.items():
            result[label]['x'] = data.keys()
            result[label]['y'] = data.values()

        return result

    def get_selectivity_sum(self):
        """
        選択性summary
        """
        cell_count  = len(self.cell_activity)
        selectivity = self.get_selectivity()

        result = {}
        for label, data in selectivity.items():
            result[label] = sum([ rate * count for rate, count in zip(data['x'], data['y'])]) / cell_count

        return result


    def save_cell_activity(self, tp_cell_activity, label):
        """
        cellのアクティブ情報を保存

        classifierでも同じような個とやってるはずだからそっちからデータ取りたかったが,
        C++側で実装されていて, 直接アクセスできない.
        そっち側あまり修正したくないので, 自分で保存する.
        """
        for cell in tp_cell_activity:
            self.cell_activity[cell][label] += 1

    def print_summary(self):
        mean, std = self.get_fired_rate()
        print
        print '### mean/std'
        print 'mean : ', mean
        print 'std : ', std

        rate = self.get_selectivity_sum()
        print
        print "### selectivity"
        print "plus  : ", rate['plus']
        print "minus : ", rate['minus']
        print "flat  : ", rate['flat']
        print


class FunctionRecogniter():

    def __init__(self):
        from collections import OrderedDict

        self.run_number = 0

        # for classifier
        self.classifier_encoder_list = {}
        self.classifier_input_list   = {}
        self.prevPredictedColumns    = {}

        self.selectivity = "region1"

        # net structure
        self.net_structure = OrderedDict()
        self.net_structure['sensor1'] = ['region1']
        # self.net_structure['sensor2'] = ['region2']
        # self.net_structure['sensor3'] = ['region3']
        # self.net_structure['region1'] = ['region4']
        # self.net_structure['region2'] = ['region4']

        # sensor change params
        self.sensor_params = {
                'sensor1': {
                    'xy_value': {
                        'maxval': 100.0,
                        'minval':  0.0
                        },
                    },
                # 'sensor2': {
                #     'xy_value': {
                #         'maxval': 80.0,
                #         'minval': 20.0
                #         },
                #     },
                # 'sensor3': {
                #     'xy_value': {
                #         'maxval': 100.0,
                #         'minval':  40.0
                #         },
                #     },
                }

        # region change params
        self.dest_resgion_data = {
                'region1': {
                    'SP_PARAMS':{
                        "columnCount": 2024,
                        "numActiveColumnsPerInhArea": 20,
                        },
                    'TP_PARAMS':{
                        "cellsPerColumn": 16
                        },
                    },
                # 'region2': {
                #     'TP_PARAMS':{
                #         "cellsPerColumn": 8
                #         },
                #     },
                # 'region3': {
                #     'TP_PARAMS':{
                #         "cellsPerColumn": 8
                #         },
                #     },
                # 'region4': {
                #     'SP_PARAMS':{
                #         "inputWidth": 2024 * (4 + 8)
                #         },
                #     'TP_PARAMS':{
                #         "cellsPerColumn": 16
                #         },
                #     },
                 }

        self._createNetwork()


        # for evaluate netwrok accuracy
        self.evaluation = NetworkEvaluation()


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


    def run(self, input_data, learn=True):
        """
        networkの実行.
        学習したいときは, learn=True, ftypeを指定する.
        予測したいときは, learn=False, ftypeはNoneを指定する.
        学習しているときも, 予測はしているがな.

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

        # selectivity
        if input_data['ftype'] is not None and inferences["anomaly"][self.selectivity] < 0.7:
        #if input_data['ftype'] is not None and input_data['xy_value'][0] > 40 and input_data['xy_value'][0] < 60:
            tp_bottomUpOut = self.network.regions[ "tp_" + self.selectivity ].getOutputData("bottomUpOut").nonzero()[0]
            self.evaluation.save_cell_activity(tp_bottomUpOut, input_data['ftype'])

        return inferences


    def _learn_classifier_multi(self, region_name, actValue=None, pstep=0):
        """
        classifierの計算を行う.

        直接customComputeを呼び出さずに, network.runの中でやりたいところだけど,
        計算した内容の取り出し方法がわからない.
        """

        # TODO: networkとclassifierを完全に切り分けたいな.
        #       networkでは, sensor,sp,tpまで計算を行う.
        #       その計算結果の評価/利用は外に出す.

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

        inferences= self._get_inferences(clResults, pstep, summary_tyep='sum')

        return inferences

    def _get_inferences(self, clResults, steps, summary_tyep='sum'):
        """
        classifierの計算結果を使いやすいように変更するだけ.
        """
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
        """
        各層のanomalyを計算
        """
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
        """
        各層のSP, TP, ClassifierのlearningModeを変更
        """
        import itertools
        for name in set( itertools.chain.from_iterable( self.net_structure.values() )):
            self.network.regions["sp_"+name].setParameter("learningMode", enable)
            self.network.regions["tp_"+name].setParameter("learningMode", enable)
            self.network.regions["class_"+name].setParameter("learningMode", enable)


    def print_inferences(self, input_data, inferences):
        """
        計算結果を出力する
        """
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

    # def layer_output(self, input_data):
    #     sensorRegion = self.network.regions["sensor"]
    #     SPRegion = self.network.regions["SP"]
    #     TPRegion = self.network.regions["TP"]
    #     SP2Region = self.network.regions["SP2"]
    #     TP2Region = self.network.regions["TP2"]
    #     print
    #     print "####################################"
    #     print
    #     print "==== Input ===="
    #     print input_data['xy_value']
    #     print
    #     print "==== EC layer ===="
    #     print "output:     ", sensorRegion.getOutputData("dataOut").nonzero()[0][:10]
    #     print
    #     print "==== SP layer ===="
    #     print "input:  ", SPRegion.getInputData("bottomUpIn").nonzero()[0][:10]
    #     print "output: ", SPRegion.getOutputData("bottomUpOut").nonzero()[0][:10]
    #     print
    #     print "==== TP layer ===="
    #     print "input:  ", TPRegion.getInputData("bottomUpIn").nonzero()[0][:10]
    #     print "output: ", TPRegion.getOutputData("bottomUpOut").nonzero()[0][:10]
    #     print
    #     print "==== SP2 layer ===="
    #     print "input:  ", SP2Region.getInputData("bottomUpIn").nonzero()[0][:10]
    #     print "output: ", SP2Region.getOutputData("bottomUpOut").nonzero()[0][:10]
    #     print
    #     print "==== TP2 layer ===="
    #     print "input:  ", TP2Region.getInputData("bottomUpIn").nonzero()[0]
    #     print "output: ", TP2Region.getOutputData("bottomUpOut").nonzero()[0]
    #     print
    #     print "==== Predict ===="
    #     print TPRegion.getSelf()._tfdr.topDownCompute().copy().nonzero()[0][:10]
    #     print
    #
    # def debug(self, input_data):
    #     TPRegion = self.network.regions["TP2"]
    #     tp_output = TPRegion.getOutputData("bottomUpOut").nonzero()[0]
    #     #print tp_output
    #
    #     if 5 in tp_output:
    #         print input_data['xy_value']


class Plotter(object):

    def __init__(self):
        self.graphs = []
        self.data_y   = defaultdict(lambda: defaultdict(list))
        self.data_x   = defaultdict(lambda: defaultdict(list))

    def add(self, title, y_values, x_values=None):
        """
        y_values = {'plus':[1, 2, 3, ..], 'minus': [1, 2, 3, ...]}
        x_values = {'plus':[1, 2, 3, ..], 'minus': [1, 2, 3, ...]}
        """
        self.data_y[title].update(y_values)
        if x_values is not None:
            self.data_x[title].update(x_values)

    def show(self):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        self.fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(len(self.data_y), 1)

        # graph setting
        for idx, title in enumerate(self.data_y.keys()):
            self.graphs.append(self.fig.add_subplot(gs[idx, 0]))
            plt.title( title )

        for idx, data_dict in enumerate(self.data_y.values()):
            title = self.data_y.keys()[idx]
            sub_title_list = []
            for sub_title, data in data_dict.items():
                if self.data_x.has_key(title):
                    self.graphs[idx].plot(self.data_x[title][sub_title] , data)
                else:
                    self.graphs[idx].plot(data)

                sub_title_list.append(sub_title)

            self.graphs[idx].legend(tuple(sub_title_list), loc=3)

        plt.show()


def main():

    fd = function_data()
    recogniter = FunctionRecogniter()
    plotter    = Plotter()

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
                inferences = recogniter.run(input_data, learn=True)

                # print
                recogniter.print_inferences(input_data, inferences)

            recogniter.reset()


    # 予測1
    result = defaultdict(list)
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

            tmp = inferences[ "classifier_" + recogniter.selectivity]['likelihoodsDict'][ftype]
            result[ftype].append(tmp)


    # plot write
    import numpy
    plotter.add(title='result', y_values=result)
    print '### result'
    for title , data in result.items():
        print title , " : ",
        print numpy.mean(data)

    # print evaluation summary
    recogniter.evaluation.print_summary()

    # plot selectivity
    selectivity = recogniter.evaluation.get_selectivity()
    for title, data in selectivity.items():
        plotter.add( title = 'selectivity', y_values={title:data['y']}, x_values={title:data['x']} )

    plotter.show()


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

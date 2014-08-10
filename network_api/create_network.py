#!/usr/bin/python
# coding: utf-8

from nupic.encoders import MultiEncoder

SP_PARAMS = {
    "spVerbosity": 0,
    "spatialImp": "cpp",
    "globalInhibition": 1,
    "columnCount": 2048,
    "inputWidth": 0,             # set later
    "numActiveColumnsPerInhArea": 20,
    "seed": 1956,
    "potentialPct": 0.8,
    "synPermConnected": 0.1,
    "synPermActiveInc": 0.01,
    "synPermInactiveDec": 0.005,
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
    "steps": '0'
}


def createVectorEncoder():
    """
    you need delete [:] at l.298
    repos/nupic/build/release/lib/python2.7/site-packages/nupic/regions/RecordSensor.py

    """
    encoder = MultiEncoder()
    encoder.addMultipleEncoders({
        "xy_value": {
            #"type": "SimpleVectorEncoder",
            "clipInput": True,
            "type": "VectorEncoderOPF",
            "dataType": "float",
            "n": 100,
            "w": 21,
            "length": 2,
            "fieldname": u"xy_value",
            "name": u"xy_value",
            "maxval": 100.0,
            "minval": 0.0,
            },
        })
    return encoder

def createCategoryEncoder(type_list):
    encoder = MultiEncoder()
    encoder.addMultipleEncoders({
        "ftype": {
            "type": "CategoryEncoder",
            "fieldname": u"ftype",
            "name": u"ftype",
            "categoryList": type_list,
            "w": 21,
            },
        })
    return encoder

class DataBuffer(object):
    def __init__(self):
        self.stack = []

    def push(self, data):
        assert len(self.stack) == 0
        data = data.__class__(data)
        self.stack.append(data)

    def getNextRecordDict(self):
        assert len(self.stack) > 0
        return self.stack.pop()



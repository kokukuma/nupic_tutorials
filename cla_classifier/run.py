#!/usr/bin/python
# coding: utf-8

import csv
import datetime

from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.modelfactory import ModelFactory
#import nupic_output

from model_params import category_test

#DATE_FORMAT = "%m/%d/%y %H:%M"
# '7/2/10 0:00'
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
# '7/2/10 0:00'

def createModel():
    model = ModelFactory.create(category_test.config)
    model.enableInference({
        "predictedField": "category"
        })
    # model.enableInference({
    #     "predictedField": "kw_energy_consumption"
    #     })
    return model

def runModel(model):
    inputFilePath = "datasets/rec-center-hourly.csv"
    inputFile = open(inputFilePath, "rb")
    csvReader = csv.reader(inputFile)
    # skip header rows
    csvReader.next()
    csvReader.next()
    csvReader.next()

    shifter = InferenceShifter()
    #output = nupic_output.NuPICFileOutput(["Rec Center"])
    #output = nupic_output.NuPICPlotOutput(["Rec Center"])

    counter = 0
    for row in csvReader:
        counter += 1
        if (counter % 100 == 0) :
            print "Read %i lines..." % counter
        if counter > 900:
            #model.disableLearning()

            timestamp = datetime.datetime.strptime(row[0], DATE_FORMAT)
            consumption = float(row[1])
            category    = float(row[2])
            result = model.run({
                "timestamp": timestamp,
                "kw_energy_consumption": consumption,
                "category": 0.0,
                "_learning": False
            })
            #print model.isLearningEnabled()
        else:
            timestamp = datetime.datetime.strptime(row[0], DATE_FORMAT)
            consumption = float(row[1])
            category    = float(row[2])
            result = model.run({
                "timestamp": timestamp,
                "kw_energy_consumption": consumption,
                "category": category,
                "_learning": True
            })

        result = shifter.shift(result)
        #print timestamp, consumption, category, result.inferences['multiStepBestPredictions']
        print timestamp, category, result.inferences['multiStepBestPredictions'][1]
        #output.write([timestamp], [consumption], [prediction])

    inputFile.close()
    #output.close()

def runHotGym():
    model = createModel()
    runModel(model)



if __name__ == "__main__":
    runHotGym()

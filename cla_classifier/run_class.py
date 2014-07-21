#!/usr/bin/python
# coding: utf-8

import csv
import datetime

from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.modelfactory import ModelFactory
#import nupic_output

from model_params import model_params

DATE_FORMAT = "%m/%d/%y %H:%M"
# '7/2/10 0:00'

def createModel():
    model = ModelFactory.create(model_params.config)
    model.enableInference({
        "predictedField": "category"
        })
    return model

def runModel(model):
    inputFilePath = "datasets/category_TP_0.csv"
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
        if counter > 200:
            # こっちは予測
            print 'karino',
            model.disableLearning()
            category = row[1]
            field1   = float(row[2])
            result = model.run({
                "field1": field1
            })
        else:
            # こっちは学習
            category = row[1]
            field1   = float(row[2])
            result = model.run({
                "category": category,
                "field1": field1
            })

        result = shifter.shift(result)
        print category, field1, result.inferences['classification']
        #output.write([timestamp], [consumption], [prediction])

    inputFile.close()
    #output.close()

def runHotGym():
    model = createModel()
    runModel(model)



if __name__ == "__main__":
    runHotGym()

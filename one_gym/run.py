#!/usr/bin/python

import csv
import datetime

from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.modelfactory import ModelFactory
import nupic_output

from model_params import model_params

DATE_FORMAT = "%m/%d/%y %H:%M"
# '7/2/10 0:00'

def createModel():
    model = ModelFactory.create(model_params.MODEL_PARAMS)
    model.enableInference({
        "predictedField": "kw_energy_consumption"
        })
    return model

def runModel(model):
    inputFilePath = "./rec-center-hourly.csv"
    inputFile = open(inputFilePath, "rb")
    csvReader = csv.reader(inputFile)
    # skip header rows
    csvReader.next()
    csvReader.next()
    csvReader.next()

    shifter = InferenceShifter()
    #output = nupic_output.NuPICFileOutput(["Rec Center"])
    output = nupic_output.NuPICPlotOutput(["Rec Center"])

    counter = 0
    for row in csvReader:
        counter += 1
        if (counter % 100 == 0) :
            print "Read %i lines..." % counter
        timestamp = datetime.datetime.strptime(row[0], DATE_FORMAT)
        consumption = float(row[1])
        result = model.run({
            "timestamp": timestamp,
            "kw_energy_consumption": consumption
        })

        result = shifter.shift(result)

        prediction = result.inferences["multiStepBestPredictions"][1]
        output.write([timestamp], [consumption], [prediction])

    inputFile.close()
    output.close()

def runHotGym():
    model = createModel()
    runModel(model)



if __name__ == "__main__":
    runHotGym()

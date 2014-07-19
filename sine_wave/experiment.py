#!/usr/bin/python

import os
import generate_data
import pprint
import csv

#from nupic.data.inference_shifter import InferenceShifter
from nupic.swarming import permutations_runner
from nupic.frameworks.opf.modelfactory import ModelFactory
from search_def import description
import nupic_anomaly_output as nupic_output

def writeModelParams(modelParams):
    outDir = os.path.join(os.getcwd(), 'model_params')
    if not os.path.isdir(outDir):
        os.mkdir(outDir)
    outPath = os.path.join(outDir, "model_params.py")

    pp = pprint.PrettyPrinter(indent=2)

    with open(outPath, "wb") as outFile:
        modelParamsString = pp.pformat(modelParams)
        outFile.write("MODEL_PARAMS = \\\n%s" % modelParamsString)
    return outPath

def swarm_over_data():
    swarmWorkDir = os.path.abspath("swarm")
    if not os.path.exists(swarmWorkDir):
        os.mkdir(swarmWorkDir)
    modelParams = permutations_runner.runWithConfig(
        description,
        {'maxWorkers': 4, "overwrite": True},
        outDir= swarmWorkDir,
        permWorkDir=swarmWorkDir
    )
    writeModelParams(modelParams)

def run_experiment():
    # generate_data.run()
    # swarm_over_data()

    from model_params import model_params
    model = ModelFactory.create(model_params.MODEL_PARAMS)
    model.enableInference({"predictedField": "sine"})

    output = nupic_output.NuPICPlotOutput("sine")
    #output = nupic_output.NuPICFileOutput(["sine"])
    #shifter = InferenceShifter()

    with open("sine.csv", "rb") as sine_input:
        csv_reader = csv.reader(sine_input)
        csv_reader.next()
        csv_reader.next()
        csv_reader.next()

        for row in csv_reader:
            angle = float(row[0])
            sine_value = float(row[1])

            result = model.run({"sine": sine_value})
            print result
            #result = shifter.shift(result)

            prediction = result.inferences["multiStepBestPredictions"][1]
            anomalyScore = result.inferences["anomalyScore"]
            output.write(angle, sine_value, prediction, anomalyScore)

    output.close()


if __name__ == "__main__":
    run_experiment()

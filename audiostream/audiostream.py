#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

from nupic.encoders.bitmaparray import BitmapArrayEncoder
from nupic.research.TP10X2 import TP10X2 as TP

import matplotlib.pyplot as plt

from pylab import *

import wave
import numpy
import pyaudio

class Visualizations():
    def calcAnomaly(self, actual, predicted):
        """
        Calculates the anomaly of two SERs
        """
        combined = numpy.logical_and(actual, predicted)
        delta = numpy.logical_xor(actual, combined)
        delta_score = sum(delta)
        actual_score = float(sum(actual))
        return delta_score / actual_score



        pass
    def compareArray(self, actual, predicted):
        compare = []
        for i in range(actual.size):
            if actual[i] and predicted[i]:
                compare.append('E')
            elif actual[i]:
                compare.append('A')
            elif predicted[i]:
                compare.append('P')
            else:
                compare.append(' ')
        return compare

    def hashtagAnomaly(self, anomaly):
        """
        Basic printout method to visualize the anomaly score (scale: 1 -50)
        """
        hashcount = '#'
        for i in range(int(anomaly / 0.02)):
            hashcount += '#'
        for j in range(int((1 - anomaly) / 0.02)):
            hashcount += '.'
        return hashcount

class AudioStream:

    def printWaveInfo(self, wf):
        """
        WAVEファイルの情報を取得
        """
        print()
        print("チャンネル数:", wf.getnchannels()                        )
        print("サンプル幅:", wf.getsampwidth()                          )
        print("サンプリング周波数:", wf.getframerate()                  )
        print("フレーム数:", wf.getnframes()                            )
        print("パラメータ:", wf.getparams()                             )
        print("長さ（秒）:", float(wf.getnframes()) / wf.getframerate() )
        print()

    def __init__(self, wf=None):
        """
        wf = None : mic
        """
        # Visualizations of result
        self.vis = Visualizations()

        # network parameter
        self.numCols = 2**9     # 2**9 = 512
        sparsity     = 0.10
        self.numInput = int(self.numCols * sparsity)

        # encoder of audiostream
        self.e = BitmapArrayEncoder(self.numCols, 1)

        # setting audio
        p = pyaudio.PyAudio()
        if wf == None:
            self.wf = None
            channels = 1
            rate = 44100                # sampling周波数: １秒間に44100回
            secToRecord = .1            #
            self.buffersize = 2**12
            self.buffersToRecord=int(rate*secToRecord/self.buffersize)
            if not self.buffersToRecord:
                self.buffersToRecord = 1
            audio_format = pyaudio.paInt32

        else:
            self.printWaveInfo(wf)

            channels = wf.getnchannels()
            self.wf = wf
            rate =    wf.getframerate()
            secToRecord = wf.getsampwidth()
            self.buffersize = 1024
            self.buffersToRecord=int(rate*secToRecord/self.buffersize)
            if not self.buffersToRecord:
                self.buffersToRecord = 1
            audio_format = p.get_format_from_width(secToRecord)

        self.inStream = p.open(
                format=audio_format,
                channels=channels,
                rate=rate,
                input=True,
                output=True,
                frames_per_buffer=self.buffersize)
        self.audio = numpy.empty((self.buffersToRecord*self.buffersize), dtype="uint32")

        # filters in Hertz
        # max lowHertz = (buffersize / 2-1) * rate / buffersize
        highHertz = 500
        lowHertz = 10000

        # Convert filters from Hertz to bins
        self.highpass = max(int(highHertz * self.buffersize /rate), 1)
        self.lowpass  = min(int(lowHertz * self.buffersize / rate), self.buffersize/2 -1)

        # Temporal Pooler
        self.tp = TP(
                numberOfCols            = self.numCols,
                cellsPerColumn          = 4,
                initialPerm             = 0.5,
                connectedPerm           = 0.5,
                minThreshold            = 10,
                newSynapseCount         = 10,
                permanenceInc           = 0.1,
                permanenceDec           = 0.07,
                activationThreshold     = 8,
                globalDecay             = 0.02,
                burnIn                  = 2,
                checkSynapseConsistency = False,
                pamLength               = 100
                )

        print("Number of columns: ", str(self.numCols))
        print("Max size of input: ", str(self.numInput))
        print("Sampling rate(Hz): ", str(rate))
        print("Passband filter(Hz): ", str(highHertz), " - ", str(lowHertz))
        print("Passband filter(bin):", str(self.highpass), " - ", str(self.lowpass))
        print("Bin difference: ", str(self.lowpass - self.highpass))
        print("Buffersize: ", str(self.buffersize))

        # # setup plot
        # plt.ion()
        # bin = range(self.highpass, self.lowpass)
        # xs = numpy.arange(len(bin)*rate/self.buffersize + highHertz)
        # self.freqPlot = plt.plot(xs, xs)[0]
        # plt.ylim(0, 10**12)


    def plotPerformance(self, values, window=1000):
        plt.clf()
        plt.plot(values[-window:])
        plt.gcf().canvas.draw()
        # Without the next line, the pyplot plot won't actually show up.
        plt.pause(0.001)


    def playAudio(self):
        """
        指定されているwaveを再生
        同時に波形をplot
        """
        chunk = 22050     # 音源が0.5秒毎に切り替わっていたため.
        data = self.wf.readframes(chunk)
        #plt.ion()
        #data_list = []
        predictedInt = None
        plt.figure(figsize=(15, 5))
        while data != '':
            dat = numpy.fromstring(data, dtype = "uint32")
            #print(dat.shape, dat)

            # plot
            data_list = dat.tolist()
            self.plotPerformance(data_list, window=500)

            # plt.plot(dat)
            # plt.show(block = False)
            # plt.draw()

            # 音ならす.
            self.inStream.write(data)

            # sampling値 -> SDR
            actualInt, actual = self.encoder(data_list)


            # actualInt, predictedInt 比較
            if not predictedInt == None:
                compare = self.vis.compareArray(actualInt, predictedInt)
                print("." . join(compare) )
                anomaly = self.vis.calcAnomaly(actualInt, predictedInt)
                print(self.vis.hashtagAnomaly(anomaly) )

            # TP predict
            predictedInt = self.tp_learn_and_predict(actual)

            # 次のデータ
            data = self.wf.readframes(chunk)

        self.inStream.close()
        p.terminate()

    def tp_learn_and_predict(self, data):
        self.tp.compute(data, enableLearn = True, computeInfOutput = True)
        predictedInt = self.tp.getPredictedState().max(axis=1)
        return predictedInt

    def encoder(self, data):
        # sampling 値 -> 周波数成分
        ys = self.fft(data, self.highpass, self.lowpass)

        # 1. 強い周波数成分の上位numInputのindexを取得する.
        # 2. 数字のレンジをnumColsに合わせる.
        # 3. uniqにする. (いるの?)
        fs = numpy.sort(ys.argsort()[-self.numInput:])
        rfs = fs.astype(numpy.float32) / (self.lowpass - self.highpass) * self.numCols
        ufs = numpy.unique(rfs)

        # encode
        actualInt = self.e.encode(ufs)
        actual = actualInt.astype(numpy.float32)

        return actualInt, actual


    def fft(self, audio, highpass, lowpass):
        left, right = numpy.split(numpy.abs(numpy.fft.fft(audio)), 2)
        output = left[highpass:lowpass]
        return output

    def formatRow(self, x):
        s = ''
        for c in range(len(x)):
            if c > 0 and c % 10 == 0:
                s += ' '
            s += str(x[c])
        s += ' '
        return s


    def getAudioString(self):
        if self.wf == None:
            print(self.buffersToRecord)
            for i in range(self.buffersToRecord):
                try:
                    audioString = self.inStream.read(self.buffersize)
                except IOError:
                    print("Overflow error from 'audiostring = inStream.read(buffersize)'. Try decreasing buffersize.")
                    quit()
                self.audio[i*self.buffersize:(i+1)*self.buffersize] = numpy.fromstring(audioString, dtype = "uint32")
        else:
            for i in range(self.buffersToRecord):
                audioString = self.wf.readframes(self.buffersize)
                self.audio[i*self.buffersize:(i+1)*self.buffersize] = numpy.fromstring(audioString, dtype = "uint32")

    def plotWave(self):
        self.getAudioString()
        print(self.audio)
        plt.plot(audiostream.audio[0:1000])
        plt.show()


if __name__ == "__main__":
    wf = wave.open('./annoying_test.wav','rb')
    audiostream = AudioStream(wf)
    audiostream.playAudio()
    #audiostream.processAudio()
    #audiostream.plotWave()



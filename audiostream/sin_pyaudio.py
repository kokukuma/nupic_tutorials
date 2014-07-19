#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
## pyaudioの使い方とその周辺の知識を得るため以下をやってみる.

+ [波形を見る](http://aidiary.hatenablog.com/entry/20110607/1307449007)
+ [離散フーリエ変換](http://aidiary.hatenablog.com/entry/20110611/1307751369)
+ [高速フーリエ変換](http://aidiary.hatenablog.com/entry/20110514/1305377659)
"""

import wave
import numpy
import pyaudio
from pylab import *
import struct

def plotPerformance(values, window=100):
    ax311 = subplot(311)
    #plt.clf()
    ax311.plot(values[-window:])
    #plt.gcf().canvas.draw()
    # Without the next line, the pyplot plot won't actually show up.
    #plt.pause(0.001)

def plotAmplitudeSpectrum(fs, N, data):
    """
    fs  : サンプリング周波数
    N   : fftサンプル数
    data:
    """
    start = 0
    # X = numpy.fft.fft(data[start:start+N])
    # freqlist = numpy.fft.fftfreq(N, d=1.0/fs)
    X, right = numpy.split(numpy.abs(numpy.fft.fft(data[start:start+N])), 2)
    freqlist = numpy.fft.fftfreq(N/2, d=1.0/fs)

    # 振幅スペクトル
    amplitudeSpectrum = [numpy.sqrt(c.real ** 2 + c.imag ** 2) for c in X]
    ax312 = subplot(312)
    ax312.plot(freqlist,  amplitudeSpectrum, marker='o', linestyle='-')
    ax312.set_xlim(0, fs/2)
    ax312.set_xlabel("frequency [Hz]")
    ax312.set_ylabel("amplitude spectrum")


    # 位相スペクトル
    phaseSpectrum = [numpy.arctan2(int(c.imag), int(c.real)) for c in X]
    ax313 = subplot(313)
    ax313.plot(freqlist,  phaseSpectrum, marker='o', linestyle='-')
    ax313.set_xlim(0, fs/2)
    ax313.set_ylim(-numpy.pi, numpy.pi)
    ax313.set_xlabel("frequency [Hz]")
    ax313.set_ylabel("phase spectrum")


def createSineWabe(A, T, fs, length):
    """
    A   : 振幅
    T   : 基本周期  [s/2pi]
    fs  : サンプリング周波数 [n/s]
    length : 音の長さ
    """
    data = []
    # 1秒間のサンプリング数がfsだから, 全体のサンプリング数はfs * lenght.
    total_sampling_num = fs * length

    for n in numpy.arange(total_sampling_num):
        # sinのx軸の変換していくイメージ(x -> s -> n)
        # x = s/T , s = n/fs なので,
        # sin(2pi * x) -> sin(2pi * x / T) -> sin(2pi * x / T / fs)
        s = numpy.sin(2 * numpy.pi / T /fs * n)
        data.append(s)

    data = [int(x*32767.0) for x in data]
    data = struct.pack("h" * len(data), *data) # バイナリ変換

    return data

def play(data, fs, bit):
    p = pyaudio.PyAudio()
    stream = p.open(
            format=pyaudio.paInt32,
            channels=2,
            rate=int(fs),
            output=True)
    chunk = 40560
    sp = 0
    buffer = data[sp:sp+chunk]
    while buffer != "":
        stream.write(buffer)
        sp += chunk
        buffer = data[sp:sp+chunk]
    stream.close()
    p.terminate()

def main():
    sampling_rate = 44100
    # ドレミファソラシド周波数
    # 1オクターブ 261ってことか.
    #freqlist = [1, 33, 69, 88, 131, 179, 233, 262]
    freqlist = [262, 294, 330, 349, 392, 440, 494, 523]
    freqlist = [262, 523, 784, 1045, 1306, 1567, 1828]
    for f in freqlist:
        data = createSineWabe(1.0, 1./f, sampling_rate, 2.0)

        # plotする.
        glf_data = [x / 32767 for x in numpy.fromstring(data, dtype = "int32")]

        # TODO: もうちょっと綺麗に & plotAmplitudeSpectrum動いてない.
        plt.clf()
        plotPerformance(glf_data)
        #plotAmplitudeSpectrum(sampling_rate, len(glf_data),  glf_data)
        plotAmplitudeSpectrum(sampling_rate, 256,  glf_data)
        #plt.gcf().canvas.draw()  # これ, 何の意味があるか分からん
        plt.pause(0.001)

        # 音ならす.
        play(data, sampling_rate, 32)

if __name__ == "__main__":
    main()

nupic_tutorials
====

## 目的
+ tutorialを通して, Nupic の使い方を把握する

## CLA White Paper
+ 多分これ読んでないと何してるかよくわからんと思う.
+ [CLA White Paper](http://numenta.org/cla-white-paper.html)

## Install
+ だいたいここの通りにやればできた.
+ [nupic](https://github.com/numenta/nupic )

## tutorials
### 一覧
+ [Using NuPIC](nupic_tutorials)
  + ipython_notebook
  + one_gym
  + one_gym_anomaly
  + sine_wave
  + spatial_pooling
  + temporal_pooling
  + audiostream

### ipython_notebook
+ notebookにまとめてくれている. 
+ しかし, これは必要なもの探すときに使った方がいい.
```
cd ipython_notebook
ipython notebook
```

### one_gym
+ どっかのジムのエネルギー消費量のデータを使って予測をしてみる.
+ [online predict framework](https://github.com/numenta/nupic/wiki/Online-Prediction-Framework)を使っている.
+ また, OPFのモデル作成には, [swarm](https://github.com/numenta/nupic/wiki/Running-Swarms)を使っている. これは, OPFと一セットと考えても良さそう.
```
cd one_gym
python swarm.py
python run.py
```

### one_gym_anomaly
+ one_gymにanomalyをつけたもの
+ 最初は誤差が大きいが, 徐々に小さくなっていく.
+ また, 見慣れないパターンが出てきたとき誤差が大きくなるが, 次第に落ち着く状態が見られる.
```
cd one_gym_anomaly
python swarm.py
python run.py
```

### sine_wave
+ 自分で作ったsin波形をOPFを使って予測する.
+ 作られたのは, one_gym/one_gym_anomalyより後っぽい.
+ データの対象が異なるだけで, やってることは, one_gym_anomalyと同じ
```
cd sine_wave
python experiment.py
```

### spatial_pooling
+ OPFを使わず, spatial_poolingを単体やってみる.
+ これは, Sparse Distributed Representations(SDR)を作るための手法
```
cd one_gym_anomaly
python hello_sp.py
```

### temporal_pooling
+ OPFを使わず, temporal_poolingを単体やってみる.
+ これは, 時系列パターンの予測をするための手法
```
cd one_gym_anomaly
python hello_tp.py
```

### audiostream
+ マイクから入力した音をTPの入力として, 予測を行うtutorial.
+ 周波数成分を入力としてる
  + 音 -> sampling値 -> 含まれる周波数 -> input
+ OPFを使わず, temporal_pooling飲み使っている.
+ マイク持ってないから, 音源ファイルを取り込む形にした.
+ pyaudioを使っているため, 以下をinstallする必要があった.
```
brew install portaudio
pip install pyaudio
```
```
cd audiostream
python audiostream.py
```

+ pyaudioの使い方いまいち分からんかったから, この辺が役に立った.
  + [波形を見る](http://aidiary.hatenablog.com/entry/20110607/1307449007)
  + [離散フーリエ変換](http://aidiary.hatenablog.com/entry/20110611/1307751369)
  + [高速フーリエ変換](http://aidiary.hatenablog.com/entry/20110514/1305377659)
```
cd audiostream
python sin_pyaudio.py
```

### classification



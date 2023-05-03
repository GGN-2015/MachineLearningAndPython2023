## 项目目标

对于一个给定的音乐片段，我们要判断，这个音乐片段是否是间奏（一个长度比较长的，而且不含“人声”的一个片段）

## 基本流水线

1. 收集一些V+音乐（相当于就是流行音乐）
2. （细节特别说，比如说，我们得对音乐进行一些预处理——**统一采样率**，采样率都尽可能统一成  ）
3. 我们要把音乐切分成小的等长的片段（2s），给音乐打标签（一般来说得人工）
4. 计算音频频谱（梅尔倒谱），归一化，再去划分等长片段，形成数据集
5. 对每个数据帧中的数据计算每种梅尔频率区间上对数能量的最大值、最小值、均值和标准差

## 被抛弃的流水线

先对从音频中分离出人声，再对人声音频套用上述流水线。

## 操作流程

1. 将 `MP3` 文件放入文件夹 `MP3` 中
2. 将音频间奏区间描述文件放入 `TAG` 文件夹中
3. 运行 `SampleRateTransformer.py` 将文件夹 `MP3`  中的音频采样率统一，并将统一后的音频存入文件夹 `MODMP3` 中
4. 运行 `MusicSpliter.py` 对音频进行切割，切割后的音频存入文件夹 `SEG`
5. 运行 `CalcMFCC.py` 生成 `DATA/MFCC_ALPHA.txt` 文件
6. 运行 `MFCCStat.py` 生成 `MFCC_ALPHA_ABSTRACT.txt` 文件
7. 运行 `FeatureSelection.py` 生成 `MFCC_ABSTRACT_BEST_FEATURE_ID.json` 文件
8. 运行 `SvmOnSelectedFeature.py` 生成 `MFCC_ALPHA_ABSTRACT_SELECTED_PREFIX.json` 文件
9. 运行 `PlotLine.py` 绘制排名前 $x$ 的特征进行二分类的性能指标

其他没有提到的代码文件主要是在被抛弃的流水线中使用的。


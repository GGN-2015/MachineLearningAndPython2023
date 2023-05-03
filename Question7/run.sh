!/bin/bash
python ./ProjectData/CODE/SampleRateTransformer.py
python ./MusicSpliter.py
python ./CalcMFCC.py
python ./MFCCStat.py
python ./FeatureSelection.py
python ./SvmOnSelectedFeature.py
python ./PlotLine.py
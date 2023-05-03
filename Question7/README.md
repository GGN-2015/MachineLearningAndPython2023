##  Project Objective

The objective of the project is to determine whether a given music segment is an interlude (a relatively long segment without vocals).

## Basic Pipeline

1. Collect some Vocoid music (similar with popular music).
2. Perform some preprocessing on the music, such as **standardizing the sample rate** to ensure consistent sample rates.
3. Divide the music into small, equal-length segments (2 seconds) and label them (usually done manually).
4. Calculate the audio spectrum (Mel frequency cepstral coefficients - MFCC), normalize the data, and divide it into equal-length segments to create a dataset.
5. Compute the maximum, minimum, mean, and standard deviation of the logarithmic energy for each Mel frequency interval in each data frame.

## Discarded Pipeline

Initially separate vocals from the audio and then apply the above pipeline to the vocal audio.

## Data Labeling Process

1. Place all music files in the MP3 folder and all Python source code files in the CODE folder.
2. Convert the sampling rate of all audio files to 44100Hz and store them in the MODMP3 folder (modified MP3). Record the sampling rate of all original MP3 files in the project in the file `DATA/SAMPLE_RATE.txt`. Use `DATA/SAMPLE_RATE_BETA.txt` to record the sampling rate of the MP3 files after the sampling rate conversion (used for verifying the correctness of the sampling rate conversion process).
3. Listen to the music and identify the time periods of all instrumental parts. Record them in text files named `TAG_00**.txt`, where "00**" represents the song's name. All TAG files are stored in the TAG folder. (Each file contains multiple lines, and each line is formatted as `ff:ff tt:tt`, representing the start and end of an instrumental segment).
4. The basic principle for determining instrumental parts: Fragments dominated by lyrics with clear meaning are not considered instrumental parts, while sections without lyrics or with ambiguous lyrics are considered instrumental parts. The presence of vocals is not the criterion for determination.
5. Use a program to divide the music into segments of two seconds each. If a segment is completely covered by an instrumental part, it is considered an instrumental segment. If it is not covered at all by an instrumental part, it is considered a non-instrumental segment. Discard all other segments.
6. For the generated segments, name them in the format `SEG_00**_ff-ff_tt-tt.mp3`, where `ff-ff` indicates the start time and `tt-tt` indicates the end time. Store all time segments corresponding to instrumental parts in the SEG/POS folder, and store the remaining files in the SEG/NEG folder.

## Operation Procedure

1. Place the `MP3` files in the "`MP3`" folder.
2. Put the interlude interval description file in the "`TAG`" folder.
3. Run "`SampleRateTransformer.py`" to standardize the sample rate of the audio files in the "`MP3`" folder and save the standardized audio files in the "`MODMP3`" folder.
4. Run "`MusicSpliter.py`" to segment the audio files and store the segmented files in the "`SEG`" folder.
5. Run "`CalcMFCC.py`" to generate the "`DATA/MFCC_ALPHA.txt`" file.
6. Run "`MFCCStat.py`" to generate the "`MFCC_ALPHA_ABSTRACT.txt`" file.
7. Run "`FeatureSelection.py`" to generate the "`MFCC_ABSTRACT_BEST_FEATURE_ID.json`" file.
8. Run "`SvmOnSelectedFeature.py`" to generate the "`MFCC_ALPHA_ABSTRACT_SELECTED_PREFIX.json`" file.
9. Run "`PlotLine.py`" to plot the performance metrics of binary classification using the top-ranked features.

Other code files mentioned are primarily used in the discarded pipeline.

## Note

If you find steps 3-9 too complicated, we have provided a `run.sh` script for you to easily run the above processes.
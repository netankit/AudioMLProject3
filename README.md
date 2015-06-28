#Audo based ML Project 3: Emotion Recognition

Employ speaker detection classifiers for emotion recognition, a multiclass classification problem. VAD is conducted like in speaker detection: as a preprocessing step to filter out non-speech frames. The established baseline method is again MAP-adaptation of a general GMM, the UBM. Instead of using speaker-specific enrollment data to adapt the UBM to a speaker model, we now adapt the UBM with emotion-specific data (of multiple speakers).

## Goals
* Familiarize yourself with emotion corpora (structure and annotation)
* Mix corpora with noise, convolve with IR
* Extract MFCCs
* Modify your speaker detection classifiers for speaker-independent emotion recognition
* Present your results


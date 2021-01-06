# Seizure Prediction And Detection Engine (S.P.A.D.E)
**Keywords** -  Brain-Computer Interface, Seizure Prediction, Machine learning, Signal Processing, Python, Tkinter, Android App Development
### Quick Project Description -
  - Worked with complex, noisy and high dimensional EEG data
  - Developed a seizure prediction system in the emerging Brain-Computer Interface technology
  - Developed a system consisted of one desktop and one mobile application that analyses the brain signals in real-time to predict an upcoming seizure several minutes before the onset of the seizure. In the case of a prediction of a seizure, the user will be alerted for timely precautionary measures.
  - Achieved an accuracy of 99.3% with a model based on Support Vector Machine and was able to predict a seizure 64 minutes before its onset, on average.


### Dataset Description - [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/)
This database, collected at the Children’s Hospital Boston, consists of EEG recordings from pediatric subjects with intractable seizures. Subjects were monitored for up to several days following withdrawal of anti-seizure medication.

Recordings, grouped into 23 cases, were collected from 22 subjects (5 males, ages 3–22; and 17 females, ages 1.5–19). The file SUBJECT-INFO contains the gender and age of each subject.nEach case (chb01, chb02, etc.) contains between 9 and 42 continuous .edf files from a single subject. In most cases, the .edf files contain exactly one hour of digitized EEG signals; occasionally, files in which seizures are recorded are shorter. All signals were sampled at 256 samples per second with 16-bit resolution. Most files contain 23 EEG signals (24 or 26 in a few cases). The International 10-20 system of EEG electrode positions and nomenclature was used for these recordings.

The file RECORDS contains a list of all 664 .edf files included in this collection, and the file RECORDS-WITH-SEIZURES lists the 129 of those files that contain one or more seizures. In all, these records include 198 seizures (182 in the original set of 23 cases); the beginning ([) and end (]) of each seizure is annotated in the .seizure annotation files that accompany each of the files listed in RECORDS-WITH-SEIZURES. In addition, the files named chbnn-summary.txt contain information about the montage used for each recording, and the elapsed time in seconds from the beginning of each .edf file to the beginning and end of each seizure contained in it.

### Project Repo Navigation

To access the code, go in the `Code` folder
  - `spades.py` contains the code of the Tkinter-based desktop application, which provides the user with a user-friendly GUI to facilitate the user to enter the metadata required to train the model
  - `spade.py` contains the python code of SPADE library which is imported by the `spades.py` to access the model training functions. It contains the complete machine learning pipeline required to train the machine learning from the raw EEG data - data pre-processing, signal processing, feature extraction, model training and validation.


The following Documentation files of the project is available at the root location
  - Poster (`Project Poster.pdf`)
  - Project Presentation (`Project Presentation.pptx`)
  - Project Report (`Project Report.pdf`)

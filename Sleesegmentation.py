import os
import mne 
import warnings
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
import glob as glob
import os
import numpy as np
import mne 
import shutil
from mne.datasets import eegbci
from mne.datasets import sleep_physionet
import matplotlib.pyplot as plt
import pandas as pd
from mne.preprocessing import (ICA, corrmap, create_ecg_epochs,
                               create_eog_epochs)
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer

raw = (r'D:\Sleep-EDF Database Expanded\sleep-edf-database-expanded-1.0.0\sleep-telemetry/ST7011J0-PSG.edf')
hyp = (r'D:\Sleep-EDF Database Expanded\sleep-edf-database-expanded-1.0.0\sleep-telemetry/ST7011JP-Hypnogram.edf')

raw = mne.io.read_raw_edf(raw, preload=True, verbose=0, stim_channel='Event marker', misc=['Temp rectal']).crop(tmin=25, tmax=None)
annot = mne.read_annotations(hyp)
raw.set_annotations(annot)

filter_length = '30s'

raw.filter(0.3, 30, fir_design='firwin',filter_length=filter_length)
raw.apply_function(lambda x: (x - x.mean()) / x.std())
# raw.set_eeg_reference('average')

ica = ICA(n_components=5, max_iter='auto', random_state=123)

ica.fit(raw)

print(len(raw.annotations))
print(set(raw.annotations.duration))
print(set(raw.annotations.description))
print(raw.annotations.onset[0])

annotation_desc_2_event_id = { 'Sleep stage 1': 1, 'Sleep stage 2': 2, 'Sleep stage 3': 3, 'Sleep stage 4': 4, 'Sleep stage R': 5, 'Sleep stage W': 6}

# keep last 30-min wake events before sleep and first 30-min wake events after
# sleep and redefine annotations on raw data
annot.crop(annot[1]['onset'] - 30 * 60,
                 annot[-2]['onset'] + 30 * 60)

raw.set_annotations(annot, emit_warning=False)

events_train, _ = mne.events_from_annotations(
    raw, event_id=annotation_desc_2_event_id, chunk_duration=30.)

# create a new event_id that unifies stages 3 and 4
event_id = {'Sleep stage W': 1,
            'Sleep stage 1': 2,
            'Sleep stage 2': 3,
            'Sleep stage 3/4': 4,
            'Sleep stage R': 5 }

# plot events
fig = mne.viz.plot_events(events_train, event_id=event_id,
                          sfreq=raw.info['sfreq'],
                          first_samp=events_train[0, 0]);

# keep the color-code for further plotting
stage_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def eeg_power_band(epochs):
    """EEG relative power band feature extraction.

    This function takes an ``mne.Epochs`` object and creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn.

    Parameters
    ----------
    epochs : Epochs
        The data.

    Returns
    -------
    X : numpy array of shape [n_samples, 5]
        Transformed data.
    """
    # specific frequency bands
    FREQ_BANDS = {"delta": [0.3, 4],
                  "theta": [4, 8],
                  "alpha": [8, 13],
                  "sigma": [11.5, 15.5],
                  "beta": [13, 30]}

    spectrum = epochs.compute_psd(picks=raw.info['ch_names'], fmin=0.3, fmax=30., n_jobs=1)
    psds, freqs = spectrum.get_data(return_freqs=True)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)

tmax = 30. - 1. / raw.info['sfreq']  # tmax in included

# Create epochs from the raw data and events
epochs = mne.Epochs(raw=raw, events=events_train, event_id=event_id,
                     tmin=0., tmax=tmax, baseline=None, preload=True)

# Filter the epochs and apply other preprocessing steps
epochs.filter(l_freq=0.3, h_freq=30)

# Equalize event counts
# epochs.equalize_event_counts(event_id)

# Get the unique labels of the events in the data
labels = epochs.events[:, 2]
unique_labels = np.unique(labels)

# Use LabelEncoder to transform the labels into integers
le = LabelEncoder()
le.fit(unique_labels)
int_labels = le.transform(labels)

# Use OneHotEncoder to transform the integer labels into one-hot encoded vectors
ohe = OneHotEncoder(sparse=False)
ohe.fit(np.arange(len(unique_labels)).reshape(-1, 1))
one_hot_labels = ohe.transform(int_labels.reshape(-1, 1))

# Convert the one-hot encoded labels to a dense array
y = coo_matrix(one_hot_labels).toarray()

# Split the data into training and testing sets
indices = np.arange(len(epochs))
train_indices, test_indices, y_train, y_test = train_test_split(indices, y, test_size=0.3, random_state=123) 

train_epochs = epochs[train_indices]
test_epochs = epochs[test_indices]

print("Number of training epochs:", len(train_epochs))
print("Number of testing epochs:", len(test_epochs))

# define the pipeline
pipe = make_pipeline(FunctionTransformer(eeg_power_band, validate=False),
                     SimpleImputer(),
                     RandomForestClassifier(random_state=123))

# train the pipeline
y_train = train_epochs.events[:, 2]
pipe.fit(train_epochs, y_train)

# test the pipeline
y_pred = pipe.predict(test_epochs)

# assess the results
y_test = test_epochs.events[:, 2]

acc = accuracy_score(y_test, y_pred)

print("Accuracy score: {:.2f}%".format(acc * 100))

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred, target_names=event_id.keys()))

print(np.unique(y_test))
print(np.unique(y_pred))
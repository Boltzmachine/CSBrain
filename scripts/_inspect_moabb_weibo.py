"""Download + structurally inspect Weibo2014 via MOABB so we can write a
PhysioNet-consistent preprocessing script. Prints montage, sfreq, epoch
window, and event->label mapping. No writes to the benchmark dirs."""
import os
os.environ.setdefault('MNE_DATA', '/gpfs/radev/pi/ying_rex/wq44/CSBrain/data/raw/mne_data')
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import mne
mne.set_log_level('ERROR')
from moabb.datasets import Weibo2014

ds = Weibo2014()
print('=== Weibo2014 ===')
print('subject_list:', ds.subject_list)
print('code:', getattr(ds, 'code', None))
print('interval (epoch tmin,tmax):', getattr(ds, 'interval', None))
print('event_id:', getattr(ds, 'event_id', None))
print('paradigm:', getattr(ds, 'paradigm', None))

data = ds.get_data(subjects=[1])
for sub, sess in data.items():
    for sname, runs in sess.items():
        for rname, raw in runs.items():
            print(f'\n-- sub {sub} sess {sname} run {rname} --')
            print('sfreq:', raw.info['sfreq'], 'nchan:', raw.info['nchan'])
            print('ch_names:', raw.ch_names)
            eeg = mne.pick_types(raw.info, eeg=True)
            print('n eeg channels:', len(eeg))
            ev, evid = mne.events_from_annotations(raw)
            print('events_from_annotations id:', evid)
            print('n_events:', len(ev))
            # inter-event spacing to gauge trial length
            if len(ev) > 1:
                import numpy as np
                d = np.diff(ev[:, 0]) / raw.info['sfreq']
                print('median inter-event sec:', float(np.median(d)))
            break
        break
    break
print('\nDONE_INSPECT_WEIBO')

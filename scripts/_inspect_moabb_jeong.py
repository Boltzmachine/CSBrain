"""Inspect Jeong2020 (intuitive upper-limb MI) structure via MOABB."""
import os
os.environ.setdefault('MNE_DATA', '/gpfs/radev/pi/ying_rex/wq44/CSBrain/data/raw/mne_data')
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import mne
mne.set_log_level('ERROR')
import moabb.datasets as md

print('has Jeong2020:', hasattr(md, 'Jeong2020'))
ds = md.Jeong2020()
print('code:', getattr(ds, 'code', None))
print('subject_list:', ds.subject_list)
print('interval:', getattr(ds, 'interval', None))
print('event_id:', getattr(ds, 'event_id', None))
print('paradigm:', getattr(ds, 'paradigm', None))
data = ds.get_data(subjects=[1])
for sub, sess in data.items():
    for sname, runs in sess.items():
        for rname, raw in runs.items():
            print(f'\n-- sub {sub} sess {sname} run {rname} --')
            print('sfreq:', raw.info['sfreq'], 'nchan:', raw.info['nchan'])
            print('ch types:', {t: raw.get_channel_types().count(t) for t in set(raw.get_channel_types())})
            print('eeg ch:', [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)])
            ev, evid = mne.events_from_annotations(raw)
            print('evid:', evid, 'n_events:', len(ev))
            break
        print('  runs in sess:', list(runs.keys()))
        break
    print('  sessions:', list(sess.keys()))
    break
print('\nDONE_INSPECT_JEONG')

"""Artifact-reject the built Forenzo LMDB IN PLACE (no re-download, no copy).

Scans every window once; a window is dropped if its max |amplitude| exceeds
ARTIFACT_UV (electrode pops / movement in the continuous-pursuit recording,
~17% of windows). Rewrites only ``__keys__`` so the artifact records become
unreferenced (both cls 'label' and reg 'vel' targets travel with each record,
so the two variants stay aligned). Idempotent: re-running just re-filters the
already-filtered key set.
"""
import pickle

import lmdb
import numpy as np

SRC = 'data/preprocessed/forenzo'
ARTIFACT_UV = 800.0


def main():
    db = lmdb.open(SRC, map_size=64 * 1024 ** 3)
    with db.begin() as t:
        keys = pickle.loads(t.get(b'__keys__'))

    newkeys = {'train': [], 'val': [], 'test': []}
    dropped = total = 0
    with db.begin() as t:
        for split, klist in keys.items():
            for key in klist:
                total += 1
                s = pickle.loads(t.get(key.encode()))['sample']
                if float(np.abs(s).max()) > ARTIFACT_UV:
                    dropped += 1
                    continue
                newkeys[split].append(key)
            print(f'  {split}: kept {len(newkeys[split])} / {len(klist)}', flush=True)

    with db.begin(write=True) as t:
        t.put(b'__keys__', pickle.dumps(newkeys))
    db.close()
    print(f'DONE: dropped {dropped}/{total} ({100*dropped/total:.1f}%) | '
          f'splits {({s: len(v) for s, v in newkeys.items()})}', flush=True)


if __name__ == '__main__':
    main()

import lmdb
import pickle

env = lmdb.open("/home/bingxing2/ailab/group/ai4neuro/BrainLLM/Imagined_speech_classification/cbramod_processed", readonly=True, lock=False)
with env.begin() as txn:
    keys = txn.get(b'__keys__')
    if keys is not None:
        keys = pickle.loads(keys)
        print("Available keys:", keys.keys())
    else:
        print("__keys__ not found!")

# import lmdb
# import pickle
# import os 

# # 和你预处理脚本的保存路径保持一致
# lmdb_path = '/home/bingxing2/ailab/group/ai4neuro/BrainLLM/Imagined_speech_classification/cbramod_processed'

# # 打开数据库
# env = lmdb.open(lmdb_path, readonly=False, map_size=3000000000)

# # 重建 key 列表
# dataset = {'train': [], 'val': [], 'test': []}

# with env.begin() as txn:
#     cursor = txn.cursor()
#     for key, _ in cursor:
#         key_str = key.decode()
#         if key_str.startswith("train-"):
#             dataset['train'].append(key_str)
#         elif key_str.startswith("val-"):
#             dataset['val'].append(key_str)
#         elif key_str.startswith("test-"):
#             dataset['test'].append(key_str)

# print("样本数量统计：")
# for split in ['train', 'val', 'test']:
#     print(f"{split}: {len(dataset[split])} samples")

# # 写入 __keys__
# with env.begin(write=True) as txn:
#     txn.put(b'__keys__', pickle.dumps(dataset))
#     print("✅ 已成功写入 __keys__ 到 LMDB！")
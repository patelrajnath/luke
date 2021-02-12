import pickle
import zlib

import lmdb
db_file = 'D:\Downloads\wiki_dump_db'
env = lmdb.open(db_file, readonly=True, subdir=False, lock=False, max_dbs=3)
meta_db = env.open_db(b'__meta__')
page_db = env.open_db(b'__page__')
redirect_db = env.open_db(b'__redirect__')

with env.begin(db=page_db) as txn:
    key = 'Delhi'
    value = txn.get(key.encode('utf-8'))
    obj = pickle.loads(zlib.decompress(value))[0][-1]
    if len(obj) == 4:
        print(obj[3])

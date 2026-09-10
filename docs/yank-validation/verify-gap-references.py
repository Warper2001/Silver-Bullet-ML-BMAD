"""Independent native-array check of every retained MBO event reference."""
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import databento as db
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(4<<20),b''): h.update(part)
    return h.hexdigest()


def verify(report_path, native_root):
    report_path=Path(report_path);native_root=Path(native_root)
    raw=report_path.read_bytes();report=json.loads(raw)
    refs=defaultdict(dict)
    def visit(obj):
        if isinstance(obj,dict):
            if {'source','record_index','event_id','sequence'}<=obj.keys():
                key=obj['record_index'];source=obj['source']
                if key in refs[source] and refs[source][key]!=obj:
                    raise ValueError('inconsistent repeated reference')
                refs[source][key]=obj
            for value in obj.values():visit(value)
        elif isinstance(obj,list):
            for value in obj:visit(value)
    visit(report)
    checked=0;files={}
    for source,items in refs.items():
        path=native_root/source
        if digest(path)!=report['native_sha256'][source]:raise ValueError('native pin mismatch')
        found=set();offset=completed=0
        store=db.DBNStore.from_file(path)
        for chunk in store.to_ndarray(count=200000):
            indices=sorted(i for i in items if offset<=i<offset+len(chunk))
            for index in indices:
                row=chunk[index-offset];ref=items[index]
                event_id=completed+int(np.count_nonzero(chunk['flags'][:index-offset]&128))
                expected=dict(record_index=index,ts_recv_ns=int(row['ts_recv']),ts_event_ns=int(row['ts_event']),
                              sequence=int(row['sequence']),flags=int(row['flags']),action=row['action'].decode(),side=row['side'].decode(),
                              price=int(row['price'])/1e9,size=int(row['size']),event_id=f'{path.name}:{event_id}')
                if any(ref[k]!=v for k,v in expected.items()):raise ValueError(f'native reference mismatch: {source}:{index}')
                found.add(index);checked+=1
            completed+=int(np.count_nonzero(chunk['flags']&128));offset+=len(chunk)
            if len(found)==len(items):break
        if hasattr(store,'reader'):store.reader.close()
        if found!=set(items):raise ValueError('missing native index')
        if digest(path)!=report['native_sha256'][source]:raise ValueError('native file changed')
        files[source]={'sha256':digest(path),'unique_references':len(items)}
    if report_path.read_bytes()!=raw:raise ValueError('report changed')
    return dict(status='PASS_NATIVE_REFERENCE_CHECK',research_status='HOLD_VALIDATION',
                report_sha256=hashlib.sha256(raw).hexdigest(),unique_references=checked,files=files,
                python=platform.python_version(),versions={name:importlib.metadata.version(name) for name in ('databento','databento-dbn','numpy')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('report');p.add_argument('native_root');a=p.parse_args()
    print(json.dumps(verify(a.report,a.native_root),indent=2,sort_keys=True))

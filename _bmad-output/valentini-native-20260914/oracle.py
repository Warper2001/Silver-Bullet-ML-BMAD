"""Independent byte-offset oracle for two prespecified short native intervals."""
import collections, hashlib, json, struct, time
from pathlib import Path
import zstandard
root=Path('/root/Silver-Bullet-ML-BMAD/data/yank/databento-pilot-20260907/native/GLBX-20260907-NWS3PA9QPX')
# First hour of two UTC files: includes many multi-print events, varied sizes/prices.
checks={}
started=time.monotonic()
for day,start in [('20250520',1747699200000000000),('20250527',1748304000000000000)]:
 end=start+60*60_000_000_000
 bars={}
 with (root/f'glbx-mdp3-{day}.mbo.dbn.zst').open('rb') as f, zstandard.ZstdDecompressor().stream_reader(f) as stream:
  head=stream.read(8); length=struct.unpack('<I',head[4:])[0]
  assert head[:4]==b'DBN\x03' and len(stream.read(length))==length
  offset=0; done=False
  while not done:
   block=stream.read(56*65536)
   if not block:break
   assert len(block)%56==0
   for i in range(0,len(block),56):
    raw=block[i:i+56]
    flags=raw[36]
    if flags&32:continue
    recv=struct.unpack_from('<Q',raw,40)[0]
    if recv>=end:
     done=True;break
    if raw[38]!=ord('T') or recv<start:continue
    assert raw[0]==14 and raw[1]==160
    price,size=struct.unpack_from('<qI',raw,24)
    assert price%250_000_000==0
    minute=recv//60_000_000_000*60_000_000_000
    b=bars.setdefault(minute,{'ticks':collections.Counter(),'trade_count':0,'digest':hashlib.sha256()})
    b['ticks'][price//250_000_000]+=size;b['trade_count']+=1;b['digest'].update(raw)
   offset+=len(block)//56
 for minute,b in bars.items():
  checks[str(minute)]={'tick_volumes':sorted(b['ticks'].items()),'trade_count':b['trade_count'],'trade_sha256':b['digest'].hexdigest()}
print(json.dumps({'intervals_utc':['2025-05-20T00:00:00Z/2025-05-20T01:00:00Z','2025-05-27T00:00:00Z/2025-05-27T01:00:00Z'],'minutes':checks},sort_keys=True))

import importlib.util,json,time
from pathlib import Path
import zstandard
root=Path('/root/Silver-Bullet-ML-BMAD/.claude/worktrees/valentini-native')
spec=importlib.util.spec_from_file_location('_native_benchmark', root/'src/research/yank_native_minute/builder.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
p=Path('/root/Silver-Bullet-ML-BMAD/data/yank/databento-pilot-20260907/native/GLBX-20260907-NWS3PA9QPX/glbx-mdp3-20250519.mbo.dbn.zst')
b=m.Builder(); start=time.monotonic(); count=0
with p.open('rb') as f, zstandard.ZstdDecompressor().stream_reader(f) as stream:
 m.metadata(stream,'mbo','20250519')
 for a in m.chunks(stream):
  b.consume(a,p.name,count); count+=len(a)
  if count>=1048576:break
print(json.dumps({'records':count,'elapsed_seconds':time.monotonic()-start,'bars':len(b.bars),'included_T_records':b.counts['included_T_records'],'purpose':'decoder throughput only; incomplete slice has no session verdict'},sort_keys=True))

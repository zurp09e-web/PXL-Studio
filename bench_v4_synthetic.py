from PIL import Image
import numpy as np, tempfile, time, csv
from pathlib import Path
import pxl

rng=np.random.default_rng(42)
y,x=np.mgrid[0:256,0:256]
images={
 'gradient':(np.stack(((x*2+y)%256,(x+y*3)%256,(x*3+y*2)%256),-1).astype('uint8'),'RGB'),
 'block_mosaic':(np.tile(rng.integers(0,256,(8,8,3),dtype=np.uint8),(32,32,1)),'RGB'),
 'repeated_patch':(np.tile(rng.integers(0,256,(32,32,3),dtype=np.uint8),(8,8,1)),'RGB'),
 'noise':(rng.integers(0,256,(256,256,3),dtype=np.uint8),'RGB'),
 'flat':(np.full((256,256,3),(20,40,60),dtype=np.uint8),'RGB'),
}
rows=[]
with tempfile.TemporaryDirectory() as td:
  for name,(arr,mode) in images.items():
    src=Path(td)/(name+'.png'); Image.fromarray(arr,mode=mode).save(src)
    for version,fn in [('v3',pxl._compress_v3),('v4',pxl.compress)]:
      out=Path(td)/(name+'_'+version+'.pxl')
      t=time.perf_counter(); fn(src,out,tile_size=64,backend='cpu',workers=1,effort='balanced'); elapsed=time.perf_counter()-t
      restored=Path(td)/(name+'_'+version+'_restored.png')
      pxl.decompress(out,restored,backend='cpu',workers=1)
      with Image.open(restored) as image:
        exact=np.array_equal(np.asarray(image.convert(mode)),arr)
      rows.append({'image':name,'version':version,'size_bytes':out.stat().st_size,'encode_seconds':round(elapsed,4),'pixel_exact':exact})
      print(rows[-1])
with open('/mnt/data/pxl4_work/PXL_Studio_3.0/bench_v4_synthetic.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

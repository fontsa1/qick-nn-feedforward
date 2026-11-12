import numpy as np
rows=[l.split() for l in open('box_dump.txt')]
cyc=np.array([int(r[0]) for r in rows]); rst=np.array([int(r[1]) for r in rows])
def lanes(h):
    v=int(h,16); w=[(v>>(16*k))&0xFFFF for k in range(16)]
    w=[x-65536 if x>=32768 else x for x in w]
    return w
X=np.array([lanes(r[2]) for r in rows]); box=np.array([int(r[3],16) if "x" not in r[3] else -1 for r in rows]); bv=np.array([int(r[4]) if r[4]!="x" else -1 for r in rows]); cfg=np.array([int(r[5],16) for r in rows])
I=X[:,0::2].sum(1); Q=X[:,1::2].sum(1)    # lane k: I = word 2k (bits 32k..), Q = word 2k+1
def s16(a): return np.clip(a,-32768,32767)
def model(t,c):
    M=((c>>8)&0x1F)+1; sh=(c>>4)&0xF
    ai=I[t-M+1:t+1].sum(); aq=Q[t-M+1:t+1].sum()
    if sh: ai=(ai+(1<<(sh-1)))>>sh; aq=(aq+(1<<(sh-1)))>>sh
    ai,aq=int(s16(ai)),int(s16(aq))
    return ((aq&0xFFFF)<<16)|(ai&0xFFFF)
# segment boundaries
chg=np.flatnonzero(np.diff(cfg))+1
segs=list(zip(np.r_[0,chg],np.r_[chg,len(cfg)]))
best=None
for lat in range(3,30):
    s0,e0=segs[1]; ok=all(box[r]==model(r-lat,cfg[r-lat]) for r in range(s0+60,e0))
    if ok: best=lat; break
print("latency (input edge -> output visible):",best,"cycles")
tot=bad=0; extremes=0
for s,e in segs:
    if cfg[s]==0: continue
    c=cfg[s]; nb=0
    for r in range(s+60,e):
        tot+=1
        if box[r]!=model(r-best,c) or bv[r]!=1: nb+=1
    M=((c>>8)&0x1F)+1; sh=(c>>4)&0xF
    bad+=nb
    print(f"cfg M={M:2d} shift={sh:2d}: {e-s-60} outputs, mismatches {nb}")
print("total",tot,"mismatches",bad)
# image rejection at 100 MHz, fs = 552.96 MHz, box8 (sum of 8 at 4423.68 MSps) * MA11
fs=552.96; f=100.0
H8=abs(np.sum(np.exp(-2j*np.pi*f/(8*fs)*np.arange(8))))/8
H11=abs(np.sum(np.exp(-2j*np.pi*f/fs*np.arange(11))))/11
print("box8 at 100 MHz: %.2f dB; MA11: %.1f dB; combined %.1f dB; MA11 null at %.2f MHz"%(20*np.log10(H8),20*np.log10(H11),20*np.log10(H8*H11),2*fs/11))
print("group delay (M-1)/2 =5 samples = %.2f ns; pipeline 9 cycles = %.2f ns"%(5e3/fs,9e3/fs))
print("acc range M=16: max |acc| =",16*8*32768,"vs 23-bit signed min",-(1<<22))

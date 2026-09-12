import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
PRIMARY="#213448"; SECOND="#547792"; ACCENT="#649CA5"; OBJ="#8065A0"; RAY="#E69650"

# Hilbert (label fix)
def d2xy(n,d):
    x=y=0; t=d; s=1
    while s<n:
        rx=1&(t//2); ry=1&(t^rx)
        if ry==0:
            if rx==1: x=s-1-x; y=s-1-y
            x,y=y,x
        x+=s*rx; y+=s*ry; t//=4; s*=2
    return x,y
n=8; pts=np.array([d2xy(n,i) for i in range(n*n)])
fig,ax=plt.subplots(figsize=(5.2,5.4))
cmap=matplotlib.colors.LinearSegmentedColormap.from_list("c",[SECOND,OBJ,RAY])
for i,(x,y) in enumerate(pts):
    ax.add_patch(plt.Rectangle((x-0.5,y-0.5),1,1,fc=cmap(i/(n*n-1)),ec="white",lw=1.2,alpha=0.85))
ax.plot(pts[:,0],pts[:,1],color=PRIMARY,lw=2.6,solid_capstyle="round")
for i in [0,21,42,63]:
    x,y=pts[i]; ax.text(x,y,str(i),ha="center",va="center",fontsize=11,color="white",fontweight="bold",bbox=dict(boxstyle="circle,pad=0.25",fc=PRIMARY,ec="none"))
qx,qy=2,4
ax.add_patch(plt.Rectangle((qx-0.5,qy-0.5),2,2,fc="none",ec=RAY,lw=3.5,zorder=5))
ax.annotate("neighbours on the map\n= neighbours in the list: 28, 29, 30, 31",xy=(qx+1.5,qy+1.2),xytext=(4.2,8.6),fontsize=11.5,color=PRIMARY,fontweight="bold",ha="center",
            arrowprops=dict(arrowstyle="->",color=RAY,lw=2),bbox=dict(boxstyle="round",fc="white",ec=RAY))
ax.set_xlim(-0.6,n-0.4); ax.set_ylim(-0.6,n+2.2); ax.set_aspect("equal"); ax.axis("off")
fig.patch.set_alpha(0); plt.tight_layout(pad=0); plt.savefig("hilbert_index.png",dpi=200,transparent=True); plt.close()

# Terrain: proper fBm by summing octaves of value noise (clear octave story)
rng=np.random.default_rng(3)
N=256
def value_noise(freq):
    g=rng.normal(size=(freq+1,freq+1))
    xs=np.linspace(0,freq,N,endpoint=False); i=xs.astype(int); f=xs-i; f=f*f*(3-2*f)
    a=g[np.ix_(i,i)]; b=g[np.ix_(i+1,i)]; c=g[np.ix_(i,i+1)]; d=g[np.ix_(i+1,i+1)]
    return (a*(1-f[:,None])+b*f[:,None])*(1-f[None,:])+(c*(1-f[:,None])+d*f[:,None])*f[None,:]
octs=[value_noise(2**k)*(0.55**k) for k in range(7)]
from matplotlib.colors import LightSource
ls=LightSource(azdeg=315,altdeg=35)
terr=matplotlib.colors.LinearSegmentedColormap.from_list("t",["#2f5f8a","#4f8fb8","#7fb069","#a9c46c","#c8b57a","#8f6f4e","#e8e8e8","#ffffff"])
fig,axes=plt.subplots(1,3,figsize=(10.5,3.9))
for ax,k,lab in zip(axes,[1,3,7],["1 layer of bumps","3 layers","7 layers"]):
    h=sum(octs[:k]); h=(h-h.min())/(h.max()-h.min())
    ax.imshow(ls.shade(h,cmap=terr,vert_exag=8,blend_mode="soft"),interpolation="bilinear")
    ax.set_title(lab,fontsize=15,color=PRIMARY,fontweight="bold"); ax.axis("off")
fig.patch.set_alpha(0); plt.tight_layout(); plt.savefig("terrain_octaves.png",dpi=200,transparent=True); plt.close()

# Retina: transparent corners + zoom on vessels near optic disc
from skimage import data
r=data.retina()
mask=(r.max(axis=2)>25).astype(np.uint8)*255
rgba=np.dstack([r,mask])
x0,y0,w=330,330,330
fig,ax=plt.subplots(figsize=(5,5)); ax.imshow(rgba); ax.axis("off")
ax.add_patch(plt.Rectangle((x0,y0),w,w,fc="none",ec=RAY,lw=3.5))
fig.patch.set_alpha(0); plt.tight_layout(pad=0); plt.savefig("retina.png",dpi=200,transparent=True); plt.close()
fig,ax=plt.subplots(figsize=(3,3)); ax.imshow(r[y0:y0+w,x0:x0+w]); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_color(RAY); sp.set_linewidth(4)
fig.patch.set_alpha(0); plt.tight_layout(pad=0); plt.savefig("retina_zoom.png",dpi=200,transparent=True); plt.close()

# Nature collage tiles from the repo (real photos) -> pick 3 tiles
from PIL import Image
im=Image.open("../fractals.png"); W,H=im.size; print(W,H)
tiles={"aloe":(0,0,W/3,H/2),"snowflake":(W/3,0,2*W/3,H/2),"romanesco":(2*W/3,0,W,H/2),"nautilus":(0,H/2,W/3,H),"sunflower":(W/3,H/2,2*W/3,H),"pinecone":(2*W/3,H/2,W,H)}
for k,b in tiles.items(): im.crop(tuple(int(v) for v in b)).save(f"nature_{k}.png")

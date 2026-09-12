import numpy as np, matplotlib, json
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
PRIMARY="#213448"; SECOND="#547792"; ACCENT="#649CA5"; OBJ="#8065A0"; RAY="#E69650"

# ---- Richardson plot, two variants ----
d=json.load(open("coast_numbers.json"))["richardson"]
s=np.array(d["steps_km"]); L=np.array(d["L_km"])
def richardson(fname, limit=False):
    fig,ax=plt.subplots(figsize=(6.2,4.2))
    sel=slice(1,None)  # fit on 4..256 km
    sl,ic=np.polyfit(np.log(s[sel]),np.log(L[sel]),1); D=1-sl
    ax.loglog(s,L,"o-",color=PRIMARY,lw=2,ms=8,mfc=RAY,mec=PRIMARY,zorder=3)
    xs=np.array([s.min(),s.max()]); ax.loglog(xs,np.exp(ic)*xs**sl,"--",color=ACCENT,lw=1.6)
    ax.xaxis.set_major_locator(FixedLocator([2,4,8,16,32,64,128,256])); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f"{v:g}"))
    ax.yaxis.set_major_locator(FixedLocator([2000,3000,4000,6000,8000])); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f"{v:,.0f}"))
    ax.set_xlabel("ruler length (km)",fontsize=13,color=PRIMARY); ax.set_ylabel("measured coastline (km)",fontsize=13,color=PRIMARY)
    ax.tick_params(labelsize=11,colors=PRIMARY)
    if not limit:
        ax.text(0.04,0.08,f"straight line on log–log axes\nslope ≈ {sl:.2f}   →   D ≈ {D:.2f}",transform=ax.transAxes,fontsize=13,color=PRIMARY,fontweight="bold",bbox=dict(boxstyle="round",fc="white",ec=PRIMARY))
    else:
        ax.axvspan(1.6,3.2,color=RAY,alpha=0.18,lw=0)
        ax.annotate("curve flattens:\nno more detail\nin the data",xy=(2.05,L[0]),xytext=(6,7800),fontsize=12,color=PRIMARY,fontweight="bold",arrowprops=dict(arrowstyle="->",color=PRIMARY,lw=1.5))
        ax.text(60,6200,"fractal range\n(straight part)",fontsize=12,color=ACCENT,fontweight="bold",ha="center")
    for sp in ax.spines.values(): sp.set_color(PRIMARY)
    ax.grid(True,which="major",alpha=0.25); ax.set_facecolor("none")
    fig.patch.set_alpha(0); plt.tight_layout(); plt.savefig(fname,dpi=200,transparent=True); plt.close(); return D
print("D fit:",richardson("richardson.png"))
richardson("richardson_limit.png",limit=True)

# ---- Hilbert curve, order 3 (8x8) ----
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
fig,ax=plt.subplots(figsize=(5.2,5.2))
cmap=matplotlib.colors.LinearSegmentedColormap.from_list("c",[SECOND,OBJ,RAY])
for i,(x,y) in enumerate(pts):
    ax.add_patch(plt.Rectangle((x-0.5,y-0.5),1,1,fc=cmap(i/(n*n-1)),ec="white",lw=1.2,alpha=0.85))
ax.plot(pts[:,0],pts[:,1],color=PRIMARY,lw=2.6,solid_capstyle="round")
for i in [0,21,42,63]:
    x,y=pts[i]; ax.text(x,y,str(i),ha="center",va="center",fontsize=11,color="white",fontweight="bold",bbox=dict(boxstyle="circle,pad=0.25",fc=PRIMARY,ec="none"))
# query window: 2x2 block of cells, show their indices are contiguous
qx,qy=2,4
ax.add_patch(plt.Rectangle((qx-0.5,qy-0.5),2,2,fc="none",ec=RAY,lw=3.5,zorder=5))
ids=[i for i,(x,y) in enumerate(pts) if qx<=x<qx+1.5 and qy<=y<qy+1.5]
ax.text(qx+0.5,qy+2.05,f"cells {min(ids)}–{max(ids)}",ha="center",va="bottom",fontsize=13,color=RAY,fontweight="bold",zorder=6)
ax.set_xlim(-0.6,n-0.4); ax.set_ylim(-0.6,n+0.3); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("2-D map  →  one number per cell",fontsize=15,color=PRIMARY,fontweight="bold")
fig.patch.set_alpha(0); plt.tight_layout(); plt.savefig("hilbert_index.png",dpi=200,transparent=True); plt.close()
print("query ids",ids)

# ---- fBm terrain via midpoint displacement (diamond-square), octave stacking ----
rng=np.random.default_rng(7)
def diamond_square(k,rough=0.55):
    size=2**k+1; h=np.zeros((size,size)); h[0,0],h[0,-1],h[-1,0],h[-1,-1]=rng.normal(0,1,4)
    step=size-1; amp=1.0
    while step>1:
        half=step//2
        for i in range(0,size-1,step):
            for j in range(0,size-1,step):
                h[i+half,j+half]=np.mean([h[i,j],h[i+step,j],h[i,j+step],h[i+step,j+step]])+rng.normal(0,amp)
        for i in range(0,size,half):
            for j in range((i+half)%step,size,step):
                vals=[h[(i-half)%size,j],h[(i+half)%size,j],h[i,(j-half)%size],h[i,(j+half)%size]]
                h[i,j]=np.mean(vals)+rng.normal(0,amp)
        step=half; amp*=rough
    return h
from matplotlib.colors import LightSource
ls=LightSource(azdeg=315,altdeg=40)
terr=matplotlib.colors.LinearSegmentedColormap.from_list("t",["#3b6b8f","#6f9f5a","#c9b27a","#8e6b4a","#ffffff"])
fig,axes=plt.subplots(1,3,figsize=(9.6,3.4))
for ax,k,lab in zip(axes,[2,4,7],["1 split","3 splits","6 splits"]):
    h=diamond_square(k)
    # upsample low-order surfaces for display (nearest → visibly blocky, that's the point)
    big=np.kron(h,np.ones((2**(7-k),2**(7-k)))) if k<7 else h
    ax.imshow(ls.shade(big,cmap=terr,vert_exag=0.6,blend_mode="soft"),interpolation="nearest")
    ax.set_title(lab,fontsize=14,color=PRIMARY,fontweight="bold"); ax.axis("off")
fig.patch.set_alpha(0); plt.tight_layout(); plt.savefig("terrain_octaves.png",dpi=200,transparent=True); plt.close()

# ---- retina (CC0) with zoom-in inset ----
from skimage import data
r=data.retina()
fig,ax=plt.subplots(figsize=(5,5)); ax.imshow(r); ax.axis("off")
# zoom box near the optic disc branching
x0,y0,w=700,520,320
ax.add_patch(plt.Rectangle((x0,y0),w,w,fc="none",ec=RAY,lw=3))
fig.patch.set_alpha(0); plt.tight_layout(pad=0); plt.savefig("retina.png",dpi=200,transparent=True); plt.close()
fig,ax=plt.subplots(figsize=(3,3)); ax.imshow(r[y0:y0+w,x0:x0+w]); ax.axis("off")
for sp in ax.spines.values(): sp.set_visible(True)
fig.patch.set_alpha(0); plt.tight_layout(pad=0); plt.savefig("retina_zoom.png",dpi=200,transparent=True); plt.close()
print("retina",r.shape)

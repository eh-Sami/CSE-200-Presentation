import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import json

PRIMARY="#213448"; SECOND="#547792"; ACCENT="#649CA5"; OBJ="#8065A0"; RAY="#E69650"; BG="#F5F7FA"

def britain(res):
    m = Basemap(projection="tmerc", lon_0=-2, lat_0=54, llcrnrlon=-8.2, llcrnrlat=49.7, urcrnrlon=2.2, urcrnrlat=59.0, resolution=res)
    polys = m.coastpolygons; types = m.coastpolygontypes
    best=None; bestlen=0
    for p,t in zip(polys,types):
        if t!=1: continue
        x,y=np.array(p[0]),np.array(p[1])
        # pick largest land polygon by bbox area
        a=(x.max()-x.min())*(y.max()-y.min())
        if a>bestlen: bestlen=a; best=(x,y)
    return best
def length_km(x,y): return np.sum(np.hypot(np.diff(x),np.diff(y)))/1000

def ruler_walk(x,y,step):
    pts=np.column_stack([x,y]); n=len(pts); count=0; i=0
    cur=pts[0]
    while i<n-1:
        j=i+1
        while j<n and np.hypot(*(pts[j]-cur))<step: j+=1
        if j>=n: break
        count+=1; cur=pts[j]; i=j
    return count

data={}
fig,axes=plt.subplots(1,3,figsize=(9,4.2))
for ax,res,label in zip(axes,["c","i","f"],["Coarse ruler","Medium ruler","Fine ruler"]):
    x,y=britain(res); L=length_km(x,y); data[res]=L
    ax.fill(x,y,color=SECOND,alpha=0.25,lw=0); ax.plot(x,y,color=PRIMARY,lw=1.2)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(label,color=PRIMARY,fontsize=14,fontweight="bold")
    ax.text(0.5,-0.02,f"≈ {L:,.0f} km",transform=ax.transAxes,ha="center",va="top",fontsize=16,color=RAY,fontweight="bold")
fig.patch.set_alpha(0)
plt.tight_layout(); plt.savefig("britain_rulers.png",dpi=200,transparent=True); plt.close()

# Richardson plot on full-res polygon
x,y=britain("f")
steps=np.array([2,4,8,16,32,64,128,256])*1000.0
N=np.array([ruler_walk(x,y,s) for s in steps])
Ls=N*steps/1000
sl,ic=np.polyfit(np.log(steps),np.log(Ls),1); D=1-sl
data["richardson"]={"steps_km":(steps/1000).tolist(),"L_km":Ls.tolist(),"D":D}
fig,ax=plt.subplots(figsize=(6.2,4.2))
ax.loglog(steps/1000,Ls,"o-",color=PRIMARY,lw=2,ms=8,mfc=RAY,mec=PRIMARY)
xs=np.array([steps.min(),steps.max()]); ax.loglog(xs/1000,np.exp(ic)*xs**sl,"--",color=ACCENT,lw=1.5)
ax.set_xlabel("ruler length  (km)",fontsize=13,color=PRIMARY); ax.set_ylabel("measured coastline  (km)",fontsize=13,color=PRIMARY)
ax.tick_params(labelsize=11,colors=PRIMARY)
ax.text(0.04,0.10,f"slope = {sl:.2f}\nD = 1 − slope ≈ {D:.2f}",transform=ax.transAxes,fontsize=15,color=PRIMARY,fontweight="bold",bbox=dict(boxstyle="round",fc="white",ec=PRIMARY))
ax.set_facecolor("none")
for s in ax.spines.values(): s.set_color(PRIMARY)
ax.grid(True,which="both",alpha=0.25)
fig.patch.set_alpha(0); plt.tight_layout(); plt.savefig("richardson.png",dpi=200,transparent=True); plt.close()
json.dump(data,open("coast_numbers.json","w"),indent=1); print(json.dumps(data,indent=1))

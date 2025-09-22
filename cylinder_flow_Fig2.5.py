# Reference-style potential flow past a cylinder
# ψ = U sinθ ( r - R^2 / r )
# with ψ(x,y) = U*y*(1 - R^2/(x^2 + y^2))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib import patheffects as pe

# ---------------- Parameters ----------------
U = 1.0
R = 1.0
L = 3.0*R
N = 900

# ---------------- Grid & polar ----------------
x = np.linspace(-L, L, N)
y = np.linspace(-2.6, 2.6, N)
X, Y = np.meshgrid(x, y)
r  = np.hypot(X, Y)

# Streamfunction
den = (X**2 + Y**2)
eps = 1e-12
den = np.where(den < eps, eps, den)
psi = U*Y*(1.0 - (R**2)/den)
psi_nd = psi/(U*R)

inside = r < R

# ---------------- Figure ----------------
fig, ax = plt.subplots(figsize=(8.2, 5.6))
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-L, L); ax.set_ylim(-2.6, 2.6)
ax.axis('off')

# Cylinder boundary r = R
tt = np.linspace(0, 2*np.pi, 600)
ax.plot(R*np.cos(tt), R*np.sin(tt), color='k', linewidth=2.2, zorder=5)

# Centerline (ψ = 0)
ax.hlines(0.0, -L, L, colors='k', linewidth=2.0, zorder=2)

# ---------------- Outer streamlines ----------------
levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
psi_out = np.ma.array(psi_nd, mask=inside)
cs = ax.contour(X, Y, psi_out, levels=levels, colors='k',
                linewidths=1.8, zorder=1, linestyles='solid')

def arrow_and_label_at_right_end(contour_set, level_index, text, dx=0.25, ms=14, lw=1.6):
    segs_for_level = contour_set.allsegs[level_index]
    if not segs_for_level: return
    right_i = int(np.argmax([seg[:, 0].max() for seg in segs_for_level]))
    seg = segs_for_level[right_i]
    j = int(np.argmax(seg[:, 0]))
    xr, yr = seg[j]
    j2 = max(0, j-3)
    vx, vy = xr - seg[j2, 0], yr - seg[j2, 1]
    nrm = np.hypot(vx, vy) + 1e-12
    vx, vy = vx/nrm, vy/nrm
    x0, y0 = xr - dx*vx, yr - dx*vy
    ax.add_patch(FancyArrowPatch((x0, y0), (xr, yr),
                                 arrowstyle='->', mutation_scale=ms,
                                 lw=lw, color='k', zorder=4))
    ax.text(xr + 0.04, yr, text, fontsize=12, va='center', ha='left', zorder=6)

val_texts = [r'$-1$', r'$-\frac{1}{2}$', r'$0$', r'$+\frac{1}{2}$', r'$\frac{\psi}{UR}=+1$']
for i in range(len(levels)):
    arrow_and_label_at_right_end(cs, i, val_texts[i])

# ---------------- Inner loops (solid) ----------------
psi_top = np.ma.array(psi_nd, mask=~((r < R) & (Y > 0)))
psi_bot = np.ma.array(psi_nd, mask=~((r < R) & (Y < 0)))
cs_top = ax.contour(X, Y, psi_top, levels=[-1.0], colors='k',
                    linewidths=1.8, linestyles='solid', zorder=3)
cs_bot = ax.contour(X, Y, psi_bot, levels=[+1.0], colors='k',
                    linewidths=1.8, linestyles='solid', zorder=3)

def vel_from_psi(xp, yp):
    den = max(1e-12, xp**2 + yp**2)
    dpsidx = U * yp * (2*R**2*xp) / (den**2)
    dpsidy = U * (1.0 - (R**2)/den) + U * yp * (2*R**2*yp) / (den**2)
    u, v = dpsidy, -dpsidx
    n = (u*u + v*v)**0.5
    return (1.0, 0.0) if n < 1e-12 else (u/n, v/n)

def arrow_tangent_on_segment(seg, pick='rightmost', ds=0.22, ms=12, lw=1.6):
    k = int(np.argmax(seg[:,0])) if pick=='rightmost' else len(seg)//2
    x0, y0 = seg[k]
    ux, uy = vel_from_psi(float(x0), float(y0))
    ax.add_patch(FancyArrowPatch((x0, y0), (x0 + ds*ux, y0 + ds*uy),
                                 arrowstyle='-|>', mutation_scale=ms,
                                 lw=lw, color='k', zorder=6))

def polygon_centroid(seg):
    x = seg[:, 0]; y = seg[:, 1]
    x2 = np.r_[x, x[0]]; y2 = np.r_[y, y[0]]
    cross = x2[:-1]*y2[1:] - x2[1:]*y2[:-1]
    A = 0.5*np.sum(cross)
    if abs(A) < 1e-12: return float(np.mean(x)), float(np.mean(y))
    Cx = np.sum((x2[:-1] + x2[1:]) * cross) / (6*A)
    Cy = np.sum((y2[:-1] + y2[1:]) * cross) / (6*A)
    return float(Cx), float(Cy)

def place_center_label(seg, text, dx=0.0, dy=0.0, fs=12, z=7):
    cx, cy = polygon_centroid(seg)
    ax.text(cx + dx, cy + dy, text, fontsize=fs, ha='center', va='center', zorder=z)

if cs_top.allsegs and cs_top.allsegs[0]:
    seg_top = max(cs_top.allsegs[0], key=len)
    arrow_tangent_on_segment(seg_top, pick='rightmost', ds=0.22)
    place_center_label(seg_top, r'$-1$')

if cs_bot.allsegs and cs_bot.allsegs[0]:
    seg_bot = max(cs_bot.allsegs[0], key=len)
    arrow_tangent_on_segment(seg_bot, pick='rightmost', ds=0.22)
    place_center_label(seg_bot, r'$+1$')

# ---------------- r = R label styled like the reference ----------------
# Point on the circle slightly above the centerline on the left
phi = np.deg2rad(165)                  # where the leader touches the circle
xy  = (R*np.cos(phi), R*np.sin(phi))   # anchor on the boundary

# Place text to the LEFT with a short leader
# tweak dx, dy to nudge until it matches your reference
dx, dy = -0.85, +0.10                  # text offset from the anchor (in data units)
ann = ax.annotate(r'$r = R$',
                  xy=xy, xytext=(xy[0]+dx, xy[1]+dy),
                  textcoords='data',
                  ha='left', va='center', fontsize=12, zorder=10,
                  arrowprops=dict(arrowstyle='-', lw=1.2, shrinkA=0, shrinkB=0))
# add a subtle white halo for readability
ann.set_path_effects([pe.withStroke(linewidth=3.0, foreground='white')])

# "0" labels
ax.text(0.62*R,  0.06, r'$0$', fontsize=12, ha='center', va='bottom')
ax.text(0.62*R, -0.06, r'$0$', fontsize=12, ha='center', va='top')

# Callouts
ax.annotate('Streamlines converge,\nhigh-velocity region',
            xy=(0.10, 1.10), xytext=(0.10, 2.10),
            arrowprops=dict(arrowstyle='->', lw=1.2, color='k'),
            ha='center', va='bottom', fontsize=12)
ax.annotate('Singularity\nat origin',
            xy=(0.0, 0.0), xytext=(1.30, -1.55),
            arrowprops=dict(arrowstyle='->', lw=1.2, color='k'),
            ha='left', va='top', fontsize=12)
# Save the figure as a PDF
plt.savefig("cylinder_flow_fig2.5.pdf", format="pdf", bbox_inches="tight")
plt.show()

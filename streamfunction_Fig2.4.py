import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch
from matplotlib.lines import Line2D

# ---------------- Parameters ----------------
a = 1.0
L = 2.2
n = 400

# ---------------- Grid & fields ----------------
x = np.linspace(-L, L, n)
y = np.linspace(-L, L, n)
X, Y = np.meshgrid(x, y)

# Velocity field
U = a * (X**2 - Y**2)
V = -2 * a * X * Y

# Streamfunction ψ = a x^2 y - a y^3 / 3
PSI = a * (X**2 * Y - (Y**3)/3)

# ---------------- Helpers ----------------
def add_arrows_to_contour(ax, cs, level, n_arrows=2, size=12, lw=1.2,
                          color=None, seg_len=0.24):
    def vel(xp, yp):
        u = a*(xp**2 - yp**2)
        v = -2*a*xp*yp
        return np.array([u, v], float)

    if level not in cs.levels:
        return
    idx = list(cs.levels).index(level)

    for seg in cs.allsegs[idx]:
        if len(seg) < 3:
            continue
        ks = np.linspace(10, len(seg)-11, n_arrows).astype(int)
        for k in ks:
            x0, y0 = seg[k]
            t = seg[k+1] - seg[k-1]                 # local tangent
            if np.dot(t, vel(x0, y0)) < 0:
                t = -t                               # orient with flow
            t = t / (np.hypot(*t) + 1e-12)

            p1 = (x0 - 0.5*seg_len*t[0], y0 - 0.5*seg_len*t[1])
            p2 = (x0 + 0.5*seg_len*t[0], y0 + 0.5*seg_len*t[1])

            c = color if color is not None else (
                cs.cmap(cs.norm(level)) if cs.cmap else "black")
            ax.add_patch(FancyArrowPatch(
                p1, p2, arrowstyle='-|>', mutation_scale=size,
                lw=lw, color=c, shrinkA=0, shrinkB=0))

def arrow_on_ray(ax, theta_deg, r_mid, length=0.50,
                 ms=11, lw=1.25, color="black"):
    th = np.deg2rad(theta_deg)
    e = np.array([np.cos(th), np.sin(th)], float)
    p0 = r_mid * e
    p1 = p0 - 0.5*length*e
    p2 = p0 + 0.5*length*e
    # align with local velocity at midpoint
    u0 = a*(p0[0]**2 - p0[1]**2); v0 = -2*a*p0[0]*p0[1]
    if np.dot(p2 - p1, np.array([u0, v0])) < 0:
        p1, p2 = p2, p1
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle='-|>', mutation_scale=ms,
        lw=lw, color=color))

def endpoint_on_square(theta_deg, L):
    th = np.deg2rad(theta_deg)
    ct, st = np.cos(th), np.sin(th)
    # If |tan| >= 1, hit y = ±L first; otherwise x = ±L
    if abs(st) >= abs(ct):  # steeper than 45°
        y = np.sign(st) * L
        x = y * (ct / st)
    else:
        x = np.sign(ct) * L
        y = x * (st / ct)
    return x, y, th

def point_next_to_ray_end(theta_deg, L, offset_perp):
    x, y, th = endpoint_on_square(theta_deg, L)
    # unit normal (rotate tangent by +90°): n = (-sin, +cos)
    nx, ny = -np.sin(th), np.cos(th)
    return x + offset_perp*nx, y + offset_perp*ny

# ---------------- Figure ----------------
fig, ax = plt.subplots(figsize=(6.0, 6.0))

# Streamlines (book-like)
psi_levels_main = a*np.array([-2.0, -1.0, 1.0, 2.0])
cs = ax.contour(X, Y, PSI, levels=psi_levels_main,
                linewidths=1.6, cmap='viridis')

# Arrows centered on ψ-contours
for lvl in psi_levels_main:
    add_arrows_to_contour(ax, cs, level=lvl,
                          n_arrows=2, size=14, lw=1.2, seg_len=0.26)

# ---------------- Separatrices (ψ=0) ----------------
xx = np.linspace(-L, L, 200)
sep_color = "black"; sep_lw = 1.4
ax.plot(xx, 0*xx,           ls="--", lw=sep_lw, color=sep_color)  # y = 0
ax.plot(xx,  np.sqrt(3)*xx, ls="--", lw=sep_lw, color=sep_color)  # y = +√3 x
ax.plot(xx, -np.sqrt(3)*xx, ls="--", lw=sep_lw, color=sep_color)  # y = -√3 x

# Thick direction arrows along the diagonals (for reference)
arrow_on_ray(ax,  60,  r_mid= 1.2, length=0.60, ms=10)
arrow_on_ray(ax, -60,  r_mid= 1.2, length=0.60, ms=10)
arrow_on_ray(ax,  60,  r_mid=-1.2, length=0.60, ms=10)
arrow_on_ray(ax, -60,  r_mid=-1.2, length=0.60, ms=10)
arrow_on_ray(ax,   0,  r_mid=-1.2, length=0.60, ms=10)   # x-axis left
arrow_on_ray(ax,   0,  r_mid= 1.2, length=0.60, ms=10)   # x-axis right

# ---------------- Angle arcs 60° ----------------
r_circ = 0.45
for th1, th2 in [(0,60),(60,120),(120,180),
                 (-180,-120),(-120,-60),(-60,0)]:
    ax.add_patch(Arc((0,0), 2*r_circ, 2*r_circ, angle=0,
                     theta1=th1, theta2=th2,
                     lw=1.2, color="black"))
    thm = np.deg2rad(0.5*(th1+th2))
    ax.text(0.62*r_circ*np.cos(thm), 0.62*r_circ*np.sin(thm),
            "60°", ha="center", va="center", fontsize=10)

# ---------------- Axes (independent lengths) ----------------
arrowprops = dict(arrowstyle="->", linewidth=1.3,
                  color="black", mutation_scale=10)
x_len = 0.75
y_len = 0.75
ax.annotate("", xy=( x_len, 0), xytext=(-x_len, 0), arrowprops=arrowprops)  # x-axis
ax.annotate("", xy=( 0, y_len), xytext=( 0,-y_len), arrowprops=arrowprops)  # y-axis
ax.text(x_len*0.98, -0.08, "x", fontsize=9, weight="bold", ha="right")
ax.text(0.08, y_len*0.98, "y", fontsize=9, weight="bold", va="top")

# ---------------- '0' labels at the ENDS of the dashed lines (but not on them) ----------------
# Small perpendicular nudge so the labels sit beside the dashed lines
off_perp_UL = -0.06  # upper-left (θ=120°): negative to keep label outward
off_perp_LL = +0.06  # lower-left (θ=240°): positive to keep label outward

x0, y0 = point_next_to_ray_end(120, L, offset_perp=off_perp_UL)   # upper-left end
x1, y1 = point_next_to_ray_end(240, L, offset_perp=off_perp_LL)   # lower-left end

ax.text(x0, y0, r'$0$', fontsize=12, ha='center', va='center')
ax.text(x1, y1, r'$0$', fontsize=12, ha='center', va='center')

# ---------------- Legend ----------------
colors = [cs.cmap(cs.norm(Lv)) for Lv in psi_levels_main]
labels = [r'$\psi=-2a$', r'$\psi=-a$', r'$\psi=a$', r'$\psi=2a$']
handles = [Line2D([0],[0], color=c, lw=2) for c in colors]
ax.legend(handles, labels, title="Streamlines",
          loc="center left", bbox_to_anchor=(1.05, 0.5),
          frameon=False, fontsize=9, title_fontsize=9)

# ---------------- Cosmetics ----------------
ax.set_aspect('equal', 'box')
ax.set_xlim(-L, L); ax.set_ylim(-L, L)
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
# Save the figure as PDF
plt.savefig("streamlines_Fig2.4.pdf", format="pdf", bbox_inches="tight")
plt.show()

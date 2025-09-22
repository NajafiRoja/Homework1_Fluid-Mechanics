import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def phi_psi_image_form(X, Y, m=1.0, a=1.0):
    """
    psi = m[ atan2(y, x+a) - atan2(y, x-a) ]
    phi = (m/2) * ln( ((x+a)^2 + y^2) / ((x-a)^2 + y^2) )
    """
    psi = m * (np.arctan2(Y, X + a) - np.arctan2(Y, X - a))
    phi = (m / 2.0) * np.log(((X + a)**2 + Y**2) / ((X - a)**2 + Y**2))
    return phi, psi

# ---------- helpers to place arrows exactly where requested ----------
def _add_arrow(ax, p0, p1, color='tab:blue', arrowsize=12, lw=1.0):
    ax.add_patch(
        FancyArrowPatch(p0, p1, arrowstyle='->',
                        mutation_scale=arrowsize, color=color,
                        linewidth=lw, zorder=5)
    )

def _arrow_along_path(ax, verts, i0, forward=True, **kw):
    """Draw a small arrow along path vertices around index i0."""
    n = len(verts)
    if n < 2:
        return
    j = i0 + 1 if forward else i0 - 1
    if j < 0 or j >= n:
        return
    p0 = (verts[i0, 0], verts[i0, 1])
    p1 = (verts[j, 0], verts[j, 1])
    _add_arrow(ax, p0, p1, **kw)

def add_streamline_arrows_custom(cs_psi, ax, xlim, color='tab:blue', arrowsize=12):
    """
    Place arrows as specified:
      • Closed paths: one arrow with x<0,y>0 and one with x<0,y<0.
      • Open paths: one arrow at the leftmost point; one at the point with x>0 and
        minimal |x| (incoming on the right side of the y-axis). Handles upper/lower halves.
    """
    xmin, xmax = xlim
    tol_close = 1e-8

    for coll in cs_psi.collections:
        for path in coll.get_paths():
            verts = path.vertices
            if len(verts) < 3:
                continue

            closed = np.allclose(verts[0], verts[-1], atol=tol_close)
            x = verts[:, 0]
            y = verts[:, 1]

            if closed:
                # one arrow left of y-axis, above; one left of y-axis, below
                for selector in [(x < 0) & (y > 0), (x < 0) & (y < 0)]:
                    idxs = np.where(selector)[0]
                    if len(idxs) < 3:
                        continue
                    target = 2
                    i0 =idxs[np.argmin(np.abs(x[idxs] + target))]
                    #i0 = idxs[3*len(idxs)//4]                    # middle of that half
                    _arrow_along_path(ax, verts, i0, forward=True,
                                      color=color, arrowsize=arrowsize)
            else:
                # open path: leftmost arrow (exit on left)
                i_left = np.argmin(x)
                _arrow_along_path(ax, verts, i_left, forward=True,
                                  color=color, arrowsize=arrowsize)

                # incoming arrow on right side of y-axis, for upper and lower parts
                for selector in [(x > 0) & (y > 0), (x > 0) & (y < 0)]:
                    idxs = np.where(selector)[0]
                    if len(idxs) < 3:
                        continue
                    # choose the point with smallest positive x (closest to y-axis)
                    target = 3
                    i0 = idxs[np.argmin(np.abs(x[idxs] - target))]
                    _arrow_along_path(ax, verts, i0, forward=True,
                                      color=color, arrowsize=arrowsize)

# ---------- main plot (unchanged aside from calling the arrow helper) ----------
def plot_source_sink_image_form(m=1.0, a=1.0,
                                xlim=(-3.5, 3.5), ylim=(-3, 3), n=800,
                                psi_levels=np.linspace(-2.5, 2.5, 20),
                                phi_levels=np.linspace(-1.5, 1.5, 15),
                                streamline_color='tab:blue',
                                equipotential_color='k'):
    x = np.linspace(*xlim, n)
    y = np.linspace(*ylim, n)
    X, Y = np.meshgrid(x, y)

    phi, psi = phi_psi_image_form(X, Y, m=m, a=a)

    fig, ax = plt.subplots(figsize=(7, 7))

    # streamlines (solid) and equipotentials (dashed)
    cs_psi = ax.contour(X, Y, psi, levels=psi_levels,
                        linestyles='solid', linewidths=1.0, colors=streamline_color)
    ax.contour(X, Y, phi, levels=phi_levels,
               linestyles='dashed', linewidths=1.0, colors=equipotential_color)

    # add arrows exactly as requested
    add_streamline_arrows_custom(cs_psi, ax, xlim, color=streamline_color, arrowsize=12)

    # source/sink markers
    ax.plot([-a], [0], 'o', ms=6, color='k')                 # source (filled)
    ax.plot([+a], [0], 'o', ms=6, mfc='white', color='k')    # sink (open)

    # solid axes
    ax.axhline(0, color='k', linewidth=1)
    ax.axvline(0, color='k', linewidth=1)

    ax.set_aspect('equal', 'box')
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    # ax.set_xlabel('x')   # remove label
    # ax.set_ylabel('y')   # remove label
    # ax.set_title('Source–sink (image form): solid ψ (streamlines), dashed φ (equipotentials)')

    # remove box and numbers
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    return fig, ax

# --- run ---
if __name__ == "__main__":
    fig, ax = plot_source_sink_image_form(m=1.0, a=1.0)
    fig.savefig("source_sink.png", dpi=300, bbox_inches="tight")
    fig.savefig("source_sink_fig2.7.pdf", bbox_inches="tight")
    plt.show()











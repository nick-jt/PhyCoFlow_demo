"""PyVista dataset-representation figures.

1. Wing: transonic case surface (wing colored by Cp, fuselage ghosted) with
   spanwise velocity-magnitude contour slices along the span.
2. JHU: 125^3 cube with orthogonal midplane slices of Ux.
3. FireBench: truth vs reconstruction flame isosurfaces (needs
   Save_reconstruction_files/firebench_field3d.npz from the GPU job).

Run from this directory: python make_dataset_figs.py [wing|jhu|firebench|all]
"""

import sys
import json
import numpy as np
import pyvista as pv

OUT = "."
RAW = ("/projects/ammoniacomb/generative_reconstruction/shift_wing/data/"
       "OnShape_luminary_crm_version001/sample_001840/")
JHU_H5 = ("/home/ntricard/generative_reconstruction/temp/"
          "PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/"
          "Dataset/JHU_TurbulenceDataset.h5")
FB_NPZ = ("/home/ntricard/generative_reconstruction/temp/"
          "PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/"
          "Save_TrainedModel/firebench/pointcloud_ffm/"
          "iclr_firebench_v2_DemoN8_20260814_221528/Evaluation/"
          "firebench_field3d.npz")

pv.global_theme.window_size = [1800, 1200]
pv.global_theme.transparent_background = True


def fig_wing():
    params = json.load(open(RAW + "params.json"))
    p_inf = float(params["pressure"])
    q_inf = 0.5 * float(params["air_density"]) * float(params["stream_velocity"]) ** 2
    u_inf = float(params["stream_velocity"])

    surf = pv.read(RAW + "merged_surfaces.vtp")
    surf["Cp"] = (np.asarray(surf.point_data["Pressure (Pa)"]) - p_inf) / q_inf
    y_signed = surf.points[:, 1]
    body = surf.extract_points(y_signed <= 3.2, adjacent_cells=True)
    wing = surf.extract_points(y_signed > 3.2, adjacent_cells=True)

    vol = pv.read(RAW + "merged_volumes.vtu")
    vol["Umag"] = np.linalg.norm(np.asarray(vol.point_data["Velocity (m/s)"]),
                                 axis=1) / u_inf

    b = np.asarray(wing.bounds)
    pl = pv.Plotter(off_screen=True)
    pl.set_background("white")
    pl.add_mesh(body, color="lightgray", opacity=0.12, smooth_shading=True)
    pl.add_mesh(wing, scalars="Cp", cmap="coolwarm", clim=[-1.2, 0.6],
                smooth_shading=True,
                scalar_bar_args=dict(title="surface Cp", vertical=True,
                                     position_x=0.03, position_y=0.25,
                                     height=0.5, width=0.05))
    # Spanwise stations (one wing side); each slice windowed to the local
    # chord via an explicit point mask (the wing is swept, so the window
    # follows the section).
    print("wing bounds:", np.round(b, 1))
    stations = np.linspace(6.0, 0.92 * b[3], 5)
    for i, ys in enumerate(stations):
        wp = wing.points
        near = np.abs(wp[:, 1] - ys) < 0.8
        if near.sum() < 10:
            continue
        x_le, x_te = wp[near, 0].min(), wp[near, 0].max()
        z_lo, z_hi = wp[near, 2].min(), wp[near, 2].max()
        chord = max(x_te - x_le, 3.0)
        # Clean rectangular slice: sample the unstructured solution onto a
        # structured in-plane grid; points outside the fluid mesh (inside
        # the body / beyond the window) become transparent via NaN.
        xc = 0.5 * (x_le + x_te) + 0.2 * chord
        zc = 0.5 * (z_lo + z_hi) + 0.1 * chord
        plane = pv.Plane(center=(xc, ys, zc), direction=(0, 1, 0),
                         i_size=1.9 * chord, j_size=1.35 * chord,
                         i_resolution=260, j_resolution=190)
        sampled = plane.sample(vol)
        um = np.asarray(sampled["Umag"]).copy()
        um[np.asarray(sampled["vtkValidPointMask"]) == 0] = np.nan
        sampled["Umag"] = um
        pl.add_mesh(sampled, scalars="Umag", cmap="viridis",
                    clim=[0.3, 1.45], opacity=0.9, nan_opacity=0.0,
                    show_scalar_bar=(i == 0),
                    scalar_bar_args=dict(title="|U| / U_inf", vertical=True,
                                         position_x=0.92, position_y=0.25,
                                         height=0.5, width=0.05))
    span = b[3]
    focus = (0.5 * (b[0] + b[1]) + 3.0, 0.42 * span, 0.5 * (b[4] + b[5]))
    pl.camera_position = [(focus[0] - 1.05 * span, focus[1] + 1.05 * span,
                           focus[2] + 1.0 * span),
                          focus, (0, 0, 1)]
    pl.screenshot(f"{OUT}/dataset_wing.png")
    print("wrote dataset_wing.png")


def fig_jhu():
    import h5py
    with h5py.File(JHU_H5, "r") as f:
        u = f["fields"][0, 300, :, 0, 0, 0].reshape(125, 125, 125)
    g = pv.ImageData(dimensions=(125, 125, 125))
    g["Ux"] = u.ravel(order="F")
    pl = pv.Plotter(off_screen=True)
    pl.set_background("white")
    lim = float(np.percentile(np.abs(u), 99))
    pl.add_mesh(g.slice_orthogonal(), scalars="Ux", cmap="RdBu_r",
                clim=[-lim, lim], lighting=False,
                scalar_bar_args=dict(title="Ux (m/s)", vertical=True,
                                     position_x=0.88, height=0.5, width=0.05))
    pl.add_mesh(g.outline(), color="black", line_width=2)
    pl.camera_position = "iso"
    pl.screenshot(f"{OUT}/dataset_jhu.png")
    print("wrote dataset_jhu.png")


def fig_firebench():
    d = np.load(FB_NPZ)
    coords = d["coords"]
    nx, nh, ny = 152, 126, 192  # x streamwise, lateral, height
    def grid(vals):
        g = pv.ImageData(dimensions=(nx, nh, ny))
        sp = (coords[:, 0].max() - coords[:, 0].min()) / (nx - 1)
        g.spacing = (3.0, 2.0, 1.0)
        g[vals[1]] = vals[0].reshape(nx, nh, ny).ravel(order="F")
        return g

    # Absolute flame levels from the truth's upper tail (z-scored units),
    # applied identically to both panels.
    th_true = d["truth"][:, 3]
    iso_warm = float(np.percentile(th_true, 99.2))
    iso_hot = float(np.percentile(th_true, 99.9))
    th_max = float(th_true.max())
    rf_max = float(np.percentile(d["truth"][:, 4], 99.9))

    pl = pv.Plotter(off_screen=True, shape=(1, 2))
    for col, (fld, title) in enumerate([(d["truth"], "LES truth"),
                                        (d["sample"], "posterior sample "
                                         "(wind-only sensing)")]):
        pl.subplot(0, col)
        pl.set_background("white")
        th = grid((fld[:, 3], "theta"))
        rf = grid((fld[:, 4], "rho_f"))
        # ground fuel bed: light = unburned fuel, dark = burned/no fuel
        ground = rf.slice(normal="z", origin=(0, 0, 0.5))
        pl.add_mesh(ground, scalars="rho_f", cmap="YlOrBr_r",
                    clim=[-0.4, rf_max], lighting=False,
                    show_scalar_bar=False)
        # flame isosurfaces: translucent warm envelope + bright hot core
        for iso, op, colr in [(iso_warm, 0.30, "#e2662a"),
                              (iso_hot, 0.95, "#ffd21f")]:
            c = th.contour([iso])
            if c.n_points:
                pl.add_mesh(c, color=colr, opacity=op, smooth_shading=True,
                            specular=0.3, show_scalar_bar=False)
        pl.add_mesh(th.outline(), color="gray", line_width=1)
        pl.add_text(title, font_size=13, color="black")
        pl.camera_position = [(-420, -380, 300), (100, 126, 45), (0, 0, 1)]
    pl.screenshot(f"{OUT}/dataset_firebench.png")
    print("wrote dataset_firebench.png")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("wing", "all"):
        fig_wing()
    if which in ("jhu", "all"):
        fig_jhu()
    if which in ("firebench", "all"):
        fig_firebench()

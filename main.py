import argparse

from reciprocalspaceship.dataset import DataSet


def map_refl_to_panel(
    data: str,
    x_det: int = 3840,
    y_det: int = 3840,
    x_panel: int = 5,
    y_panel: int = 12,
    x_subpanel: int = 0,
    y_subpanel: int = 0,
    x_interpanel: int | None = 0,
    y_interpanel: int | None = 0,
    refl: str | None = None,
    out_refl: str = "integrated_panel_ids.refl",
    out_mtz: str | None = "unmerged_panel_ids.mtz",
    plot_centroids: bool = False,
    show_plot: bool = False,
    save_plot: bool = False,
    plot_filename: str = "panel_centroids.png",
) -> None:
    from itertools import product

    import matplotlib.pyplot as plt
    import numpy as np
    import reciprocalspaceship as rs
    from dials.array_family import flex
    from matplotlib.figure import Figure
    from scipy.spatial import KDTree

    def plot_panel_centroids(
        panel_centroids: np.ndarray,
        y_det: int | None = None,
        x_det: int | None = None,
        show: bool = True,
        save_plot: bool = False,
        out_filename: str | None = None,
    ) -> Figure:
        plt.clf()
        plt.scatter(panel_centroids[:, 0], panel_centroids[:, 1])

        if y_det is not None:
            plt.ylim(0, y_det)

        if x_det is not None:
            plt.xlim(0, x_det)

        plt.title("Detector panel centroids")
        plt.ylabel("Y")
        plt.xlabel("X")

        if save_plot:
            print("Saving plot")
            plt.savefig(
                "panel_centroids.png" if out_filename is None else out_filename,
                dpi=300,
            )

        if show:
            plt.show()

        return plt.gcf()

    def _get_xy_array_mtz(mtz: str) -> tuple[DataSet, np.ndarray] | None:
        df = rs.read_mtz(mtz)

        if "XDET" and "YDET" in df.columns:
            coords = df[["XDET", "YDET"]].to_numpy()
            return df, coords
        elif "xcal" and "ycal" in df.columns:
            coords = df[["xcal", "ycal"]].to_numpy()
            return df, coords
        else:
            print("Unknown coordinate keys in .mtz")

    def map_to_centroids(panel_centroids: np.ndarray, xy_array: np.ndarray):
        kdtree = KDTree(data=panel_centroids)
        query = kdtree.query(xy_array)
        return query

    def _get_xy_array_refl(refl_path: str):
        """function to get x,y coordinates from a reflection file"""
        # read reflection table
        tbl = flex.reflection_table.from_file(refl_path)
        # predicted centroids
        arr = np.array(tbl["xyzcal.px"])
        if arr.shape[-1] == 3:
            xy_arr = arr[:, [0, 1]]
            return tbl, xy_arr
        elif arr.shape[-2] == 2:
            xy_arr = arr
            return tbl, xy_arr
        else:
            raise ValueError("Invalid coordinate dimensions. `arr` must be 2D or 3D")

    def _get_panel_centroids(
        npix_sm_x: float,
        npix_sm_y: float,
        npix_inter_x: int,
        npix_inter_y: int,
        npanels_x,
        npanels_y,
    ) -> np.ndarray:
        """
        Assigns centroids to each detector panel.
        """
        panel_centroids = np.array(
            list(
                product(
                    [
                        npix_sm_x / 2 + i * npix_inter_x + i * npix_sm_x
                        for i in range(x_panel)
                    ],
                    [
                        npix_sm_y / 2 + i * npix_inter_y + i * npix_sm_y
                        for i in range(y_panel)
                    ],
                )
            )
        )
        return panel_centroids

    # get xy_array
    if data.endswith("refl"):
        tbl, xy_array = _get_xy_array_refl(refl_path=data)
    elif data.endswith("mtz"):
        df, xy_array = _get_xy_array_mtz(mtz=data)
    else:
        raise ValueError("Invalid filetype! Use a .refl or .mtz file.")

    # number of gaps between panels
    x_gaps = x_panel - 1
    y_gaps = y_panel - 1

    # number of pixels in between modules in x and y
    x_interpanel = 0 if x_interpanel is None else x_interpanel
    y_interpanel = 0 if y_interpanel is None else y_interpanel

    # number of pixels per panel in x and y
    n_pix_x = x_det / x_panel
    n_pix_y = x_det / x_panel

    # dimensions of each panel
    panel_dim = (x_panel, y_panel)

    # get centroids of each panel
    panel_centroids = _get_panel_centroids(
        npix_sm_x=n_pix_x,
        npix_sm_y=n_pix_y,
        npix_inter_x=x_interpanel,
        npix_inter_y=y_interpanel,
        npanels_x=n_pix_x,
        npanels_y=n_pix_y,
    )

    dist_panel, panel_id = map_to_centroids(
        panel_centroids=panel_centroids, xy_array=xy_array
    )

    if plot_centroids:
        # plotting the centroids
        plot_panel_centroids(
            panel_centroids=panel_centroids,
            x_det=x_det,
            y_det=y_det,
            show=show_plot,
            save_plot=save_plot,
            out_filename=plot_filename,
        )

    if data.endswith("refl"):
        print(f"Saving reflection table to: {out_refl}")
        tbl["panel_id"] = flex.int(panel_id)
        # write a new reflection file with panel_id column
        tbl.as_file(out_refl)

    if data.endswith("mtz"):
        print(f"Saving unmerged mtz to: {out_mtz}")
        df["PANEL_ID"] = panel_id
        df["PANEL_ID"] = df["PANEL_ID"].astype(rs.dtypes.MTZIntDtype())
        df.write_mtz(mtzfile=out_mtz)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="map-refl-to-panel",
        description="Map reflections (.mtz/.refl) to nearest detector panel centroids and optionally write outputs.",
    )
    # Positional
    p.add_argument(
        "data",
        help="path to integrated.refl or unmerged.mtz file",
    )
    # Detector geometry
    p.add_argument(
        "--x-det",
        type=int,
        default=3840,
        help="Number of pixels along horizontal axis (default: 3840)",
    )
    p.add_argument(
        "--y-det",
        type=int,
        default=3840,
        help="Number of pixels along vertical axis (default: 3840)",
    )
    p.add_argument(
        "--x-panel",
        type=int,
        default=16,
        help="number of panels along horizontal axis (default: 16)",
    )
    p.add_argument(
        "--y-panel",
        type=int,
        default=16,
        help="number of panels along vertical axis (default: 16)",
    )
    p.add_argument(
        "--x-subpanel",
        type=int,
        default=0,
        help="number of subpanels along horizontal axis (default: 0)",
    )
    p.add_argument(
        "--y-subpanel",
        type=int,
        default=0,
        help="number of subpanels along vertical axis (default: 0)",
    )
    p.add_argument(
        "--x-interpanel",
        type=int,
        default=0,
        help="number of pixels in panel gap horizontally (default: 0)",
    )
    p.add_argument(
        "--y-interpanel",
        type=int,
        default=0,
        help="number of pixels in panel gap vertically (default: 0)",
    )
    # I/O
    p.add_argument(
        "--refl",
        type=str,
        default=None,
        help="Path to integrated.refl",
    )
    p.add_argument(
        "--out-refl",
        type=str,
        default="integrated_panel_ids.refl",
        help="Name of the output .refl file (default: integrated_panel_ids.refl)",
    )
    p.add_argument(
        "--out-mtz",
        type=str,
        default="unmerged_panel_ids.mtz",
        help="Name of the output .mtz file (default: unmerged_panel_ids.mtz)",
    )
    # Plotting
    p.add_argument(
        "--plot-centroids",
        action="store_true",
        help="Set this option to plot panel centroids",
    )
    p.add_argument(
        "--show-plot",
        action="store_true",
        help="Set this option to show the plot",
    )
    p.add_argument(
        "--save-plot",
        action="store_true",
        help="Set this option to save the plot",
    )
    p.add_argument(
        "--plot-filename",
        type=str,
        default="panel_centroids.png",
        help="Optional filename if saving plot (default: panel_centroids.png)",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    map_refl_to_panel(**vars(args))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert PLY geometry to STL or OBJ with optional isosurface reconstruction.

The script supports three common workflows:
- Direct mesh cleanup, decimation, and smoothing for triangle-mesh PLY files.
- Point-cloud or mesh-to-volume reconstruction using a smoothed occupancy grid
  and marching cubes.
- Final scaling to a user-provided target size, either uniform or anisotropic.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy import ndimage
from skimage import measure


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def load_geometry(input_path: Path) -> tuple[str, object]:
    mesh = o3d.io.read_triangle_mesh(str(input_path))
    if not mesh.is_empty() and len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
        return "mesh", mesh

    point_cloud = o3d.io.read_point_cloud(str(input_path))
    if not point_cloud.is_empty() and len(point_cloud.points) > 0:
        return "point_cloud", point_cloud

    raise ValueError(f"Could not read a mesh or point cloud from {input_path}")


def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def sample_points_from_geometry(
    geometry_type: str,
    geometry: object,
    sample_points: int,
) -> np.ndarray:
    if geometry_type == "point_cloud":
        points = np.asarray(geometry.points, dtype=np.float64)
        if points.size == 0:
            raise ValueError("Input point cloud has no points")
        return points

    mesh = geometry
    if len(mesh.triangles) == 0:
        raise ValueError("Input mesh has no triangles")

    sample_count = max(int(sample_points), len(mesh.triangles))
    points = mesh.sample_points_uniformly(number_of_points=sample_count)
    return np.asarray(points.points, dtype=np.float64)


def reconstruct_isosurface(
    points: np.ndarray,
    grid_resolution: int,
    voxel_size: float | None,
    padding: float,
    volume_sigma: float,
    iso_level: float,
) -> o3d.geometry.TriangleMesh:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must have shape (N, 3)")
    if len(points) < 4:
        raise ValueError("Need at least four points for reconstruction")

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extents = maxs - mins
    extents = np.maximum(extents, 1e-9)
    mins = mins - extents * float(padding)
    maxs = maxs + extents * float(padding)
    extents = maxs - mins

    if voxel_size is not None:
        grid_shape = np.ceil(extents / float(voxel_size)).astype(int)
    else:
        longest = float(extents.max())
        base = max(int(grid_resolution), 16)
        grid_shape = np.ceil(base * extents / max(longest, 1e-9)).astype(int)

    grid_shape = np.maximum(grid_shape, 16)
    spacing = extents / grid_shape

    edges = [np.linspace(mins[i], maxs[i], int(grid_shape[i]) + 1) for i in range(3)]
    volume, _ = np.histogramdd(points, bins=edges)
    volume = volume.astype(np.float32)

    if volume.max() <= 0:
        raise ValueError("Reconstruction volume is empty; increase sample count or padding")

    volume /= volume.max()
    if volume_sigma > 0:
        volume = ndimage.gaussian_filter(volume, sigma=float(volume_sigma))
        if volume.max() > 0:
            volume /= volume.max()

    volume = np.pad(volume, 1, mode="constant")
    origin = mins - spacing

    if not (float(volume.min()) <= iso_level <= float(volume.max())):
        raise ValueError(
            f"iso_level={iso_level} is outside the volume range "
            f"[{float(volume.min()):.4f}, {float(volume.max()):.4f}]"
        )

    vertices, faces, _, _ = measure.marching_cubes(
        volume,
        level=float(iso_level),
        spacing=tuple(float(x) for x in spacing),
    )
    vertices = vertices + origin

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices.astype(np.float64)),
        o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    return clean_mesh(mesh)


def decimate_mesh(
    mesh: o3d.geometry.TriangleMesh,
    target_triangles: int | None,
    decimate_ratio: float,
) -> o3d.geometry.TriangleMesh:
    current = len(mesh.triangles)
    if current == 0:
        raise ValueError("Mesh has no triangles to decimate")

    if target_triangles is None:
        if decimate_ratio >= 0.999:
            return mesh
        target_triangles = max(4, int(current * decimate_ratio))

    target_triangles = int(target_triangles)
    if target_triangles >= current:
        return mesh

    logging.info("Decimating mesh from %d to %d triangles", current, target_triangles)
    mesh = mesh.simplify_quadric_decimation(target_triangles)
    return clean_mesh(mesh)


def smooth_mesh(
    mesh: o3d.geometry.TriangleMesh,
    method: str,
    iterations: int,
) -> o3d.geometry.TriangleMesh:
    if method == "none" or iterations <= 0:
        return mesh

    logging.info("Smoothing mesh using %s (%d iterations)", method, iterations)
    if method == "taubin":
        mesh = mesh.filter_smooth_taubin(number_of_iterations=iterations)
    elif method == "laplacian":
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=iterations)
    elif method == "simple":
        mesh = mesh.filter_smooth_simple(number_of_iterations=iterations)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")

    return clean_mesh(mesh)


def center_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    bbox = mesh.get_axis_aligned_bounding_box()
    mesh.translate(-bbox.get_center())
    return mesh


def scale_mesh_to_target(
    mesh: o3d.geometry.TriangleMesh,
    target_size: list[float] | None,
    scale_mode: str,
) -> o3d.geometry.TriangleMesh:
    if not target_size:
        return mesh

    target = np.asarray(target_size, dtype=np.float64)
    if target.size not in (1, 3):
        raise ValueError("target_size must contain either 1 or 3 values")

    bbox = mesh.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent(), dtype=np.float64)
    extent = np.maximum(extent, 1e-12)

    if target.size == 1 or scale_mode == "uniform":
        desired = float(target.max())
        factor = desired / float(extent.max())
        mesh.scale(factor, center=(0.0, 0.0, 0.0))
        logging.info("Applied uniform scale factor %.6f", factor)
        return mesh

    factors = target / extent
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices *= factors.reshape(1, 3)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    logging.info("Applied anisotropic scale factors %s", factors.tolist())
    return mesh


def mesh_to_trimesh(mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.triangles, dtype=np.int64),
        process=False,
    )


def resolve_export_path(output: Path, export_format: str, input_stem: str) -> Path:
    if export_format == "obj" and output.suffix.lower() != ".obj":
        output.mkdir(parents=True, exist_ok=True)
        return output / f"{input_stem}.obj"

    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def export_obj_with_mtl(mesh: trimesh.Trimesh, obj_path: Path) -> Path:
    material_name = "material_0"
    mtl_path = obj_path.with_suffix(".mtl")

    mtl_path.write_text(
        "\n".join(
            [
                f"newmtl {material_name}",
                "Ka 0.200000 0.200000 0.200000",
                "Kd 0.800000 0.800000 0.800000",
                "Ks 0.000000 0.000000 0.000000",
                "d 1.000000",
                "illum 2",
                "",
            ]
        ),
        encoding="utf-8",
    )

    obj_text = mesh.export(file_type="obj")
    if isinstance(obj_text, bytes):
        obj_text = obj_text.decode("utf-8")

    obj_lines = obj_text.splitlines()
    header = [f"mtllib {mtl_path.name}", f"usemtl {material_name}"]
    if obj_lines and obj_lines[0].startswith("#"):
        obj_lines = [obj_lines[0], *header, *obj_lines[1:]]
    else:
        obj_lines = [*header, *obj_lines]

    obj_path.write_text("\n".join(obj_lines) + "\n", encoding="utf-8")
    return mtl_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PLY file to STL with optional decimation, smoothing, and isosurface reconstruction."
    )
    parser.add_argument("input", type=Path, help="Input PLY file")
    parser.add_argument("output", type=Path, help="Output STL file")
    parser.add_argument(
        "--export-format",
        choices=["stl", "obj"],
        default="stl",
        help="Export format. Use obj to write an OBJ file into the output directory.",
    )
    parser.add_argument(
        "--reconstruct",
        action="store_true",
        help="Rebuild an isosurface from a smoothed occupancy grid before export",
    )
    parser.add_argument(
        "--sample-points",
        type=int,
        default=200000,
        help="Number of points to sample from a mesh when reconstructing",
    )
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=256,
        help="Base resolution for marching-cubes reconstruction",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Optional voxel size for reconstruction. Overrides --grid-resolution.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="Padding fraction added around the sampled geometry before reconstruction",
    )
    parser.add_argument(
        "--volume-sigma",
        type=float,
        default=1.25,
        help="Gaussian smoothing sigma, in grid cells, applied to the reconstruction volume",
    )
    parser.add_argument(
        "--iso-level",
        type=float,
        default=0.12,
        help="Isosurface threshold for marching cubes on the normalized volume",
    )
    parser.add_argument(
        "--target-triangles",
        type=int,
        default=None,
        help="Optional triangle count for quadric decimation",
    )
    parser.add_argument(
        "--decimate-ratio",
        type=float,
        default=1.0,
        help="Fallback decimation ratio when --target-triangles is not set",
    )
    parser.add_argument(
        "--smooth-method",
        choices=["none", "simple", "laplacian", "taubin"],
        default="taubin",
        help="Mesh smoothing method",
    )
    parser.add_argument(
        "--smooth-iterations",
        type=int,
        default=5,
        help="Smoothing iterations to apply after reconstruction or decimation",
    )
    parser.add_argument(
        "--target-size",
        type=float,
        nargs="+",
        default=None,
        help="Desired final size. Provide 1 value for uniform scaling or 3 values for x y z extents.",
    )
    parser.add_argument(
        "--scale-mode",
        choices=["uniform", "anisotropic"],
        default="uniform",
        help="How to match --target-size when 3 values are provided",
    )
    parser.add_argument(
        "--no-center",
        action="store_true",
        help="Preserve the input position instead of centering the mesh before scaling",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Write ASCII STL instead of binary STL",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    geometry_type, geometry = load_geometry(args.input)
    logging.info("Loaded %s from %s", geometry_type, args.input)

    if geometry_type == "mesh":
        mesh = geometry
    else:
        mesh = None

    if args.reconstruct or geometry_type == "point_cloud":
        points = sample_points_from_geometry(
            geometry_type=geometry_type,
            geometry=geometry,
            sample_points=args.sample_points,
        )
        mesh = reconstruct_isosurface(
            points=points,
            grid_resolution=args.grid_resolution,
            voxel_size=args.voxel_size,
            padding=args.padding,
            volume_sigma=args.volume_sigma,
            iso_level=args.iso_level,
        )
        logging.info(
            "Reconstructed mesh with %d vertices and %d triangles",
            len(mesh.vertices),
            len(mesh.triangles),
        )
    else:
        mesh = clean_mesh(mesh)

    mesh = smooth_mesh(
        mesh=mesh,
        method=args.smooth_method,
        iterations=args.smooth_iterations,
    )

    if not args.no_center:
        mesh = center_mesh(mesh)

    mesh = scale_mesh_to_target(
        mesh=mesh,
        target_size=args.target_size,
        scale_mode=args.scale_mode,
    )

    mesh = decimate_mesh(
        mesh=mesh,
        target_triangles=args.target_triangles,
        decimate_ratio=args.decimate_ratio,
    )

    mesh.compute_vertex_normals()

    export_path = resolve_export_path(args.output, args.export_format, args.input.stem)
    tri_mesh = mesh_to_trimesh(mesh)
    if args.export_format == "obj":
        mtl_path = export_obj_with_mtl(tri_mesh, export_path)
        logging.info(
            "Wrote %s and %s with %d vertices and %d triangles",
            export_path,
            mtl_path,
            len(mesh.vertices),
            len(mesh.triangles),
        )
    else:
        tri_mesh.export(export_path, file_type=args.export_format)
        logging.info(
            "Wrote %s with %d vertices and %d triangles",
            export_path,
            len(mesh.vertices),
            len(mesh.triangles),
        )


if __name__ == "__main__":
    main()
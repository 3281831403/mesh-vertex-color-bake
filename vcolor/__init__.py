"""Vertex-color baking package for the AO bake bugfix exercise."""

from .bake import (
    VColorError,
    ao_samples,
    bake_colors,
    bake_vcolor_batch,
    occlusion,
    shade_by_normal,
    smooth_colors,
    validate,
    vertex_normals,
)

__all__ = [
    "VColorError",
    "ao_samples",
    "bake_colors",
    "bake_vcolor_batch",
    "occlusion",
    "shade_by_normal",
    "smooth_colors",
    "validate",
    "vertex_normals",
]

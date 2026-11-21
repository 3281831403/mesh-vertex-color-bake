"""Contract tests for the vertex-color AO bake exercise.

The suite starts red: every test drives at least one stub from ``vcolor.bake``
and the stubs raise ``NotImplementedError``, so a fresh checkout reports
``15 failed`` and no errors.  Colors, AO terms and light terms are compared
with an absolute tolerance of ``1e-6``.
"""

import pytest

from vcolor.bake import (
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

TOL = 1e-6


def _mesh(positions, normals, indices):
    return {
        "positions": [tuple(float(c) for c in p) for p in positions],
        "normals": [tuple(float(c) for c in n) for n in normals],
        "indices": [tuple(int(i) for i in face) for face in indices],
    }


def _flat_plane(size=2.0, y=0.0, shift_x=0.0):
    """Up-facing square in the XZ plane: 4 vertices, 2 triangles."""
    half = size / 2.0
    positions = [
        (-half + shift_x, y, -half),
        (half + shift_x, y, -half),
        (half + shift_x, y, half),
        (-half + shift_x, y, half),
    ]
    return _mesh(positions, [(0.0, 1.0, 0.0)] * 4, [(0, 1, 2), (0, 2, 3)])


def _small_triangle(x=0.0):
    """Three up-facing vertices forming a small triangle offset on X."""
    return _mesh(
        [(x, 0.0, 0.0), (x + 1.0, 0.0, 0.0), (x, 0.0, 1.0)],
        [(0.0, 1.0, 0.0)] * 3,
        [(0, 1, 2)],
    )


def _big_and_tiny_faces():
    """A 50.0-area +Y face and a 0.00005-area +X face sharing vertex 0."""
    return _mesh(
        [
            (0.0, 0.0, 0.0),   # 0, shared by both faces
            (0.0, 0.0, 10.0),  # 1
            (10.0, 0.0, 0.0),  # 2
            (0.0, 0.01, 0.0),  # 3
            (0.0, 0.0, 0.01),  # 4
        ],
        [(0.0, 1.0, 0.0)] * 5,
        [(0, 1, 2), (0, 3, 4)],
    )


def test_vertex_normals_are_area_weighted():
    """Small faces must not outvote large faces in the vertex normal."""
    normals = vertex_normals(_big_and_tiny_faces())

    assert len(normals) == 5
    nx, ny, nz = normals[0]

    # Equal weighting would give roughly (0.707, 0.707, 0.0) here.
    assert abs(nx) <= 1e-3
    assert ny == pytest.approx(1.0, abs=TOL)
    assert abs(nz) <= TOL
    assert (nx * nx + ny * ny + nz * nz) ** 0.5 == pytest.approx(1.0, abs=TOL)


def test_ao_samples_follow_the_surface_normal():
    """Samples must cover the hemisphere the normal points into."""
    normal = (0.0, 1.0, 0.0)

    samples = ao_samples(normal, 16, seed=20261119)

    assert len(samples) == 16
    for sample in samples:
        sx, sy, sz = (float(c) for c in sample)
        assert (sx * sx + sy * sy + sz * sz) ** 0.5 == pytest.approx(1.0, abs=1e-3)
        assert sy >= -TOL

    again = ao_samples(normal, 16, seed=20261119)
    assert [[float(c) for c in s] for s in again] == [
        [float(c) for c in s] for s in samples
    ]


def test_occlusion_ignores_the_surface_itself():
    """A ray leaving the surface must not count its own starting point."""
    samples = [(0.0, 1.0, 0.0), (0.6, 0.8, 0.0), (0.0, 0.8, 0.6)]

    blocked = occlusion(_flat_plane(), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), samples)

    assert blocked == pytest.approx(0.0, abs=TOL)


def test_shade_never_returns_negative():
    """Back-facing vertices must be clamped to 0, never negative."""
    assert shade_by_normal((0.0, 0.0, -1.0)) == pytest.approx(0.0, abs=TOL)
    assert shade_by_normal((1.0, 0.0, -1.0)) == pytest.approx(0.0, abs=TOL)
    for direction in [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0)]:
        assert shade_by_normal(direction) >= 0.0


def test_baked_colors_stay_within_unit_range():
    """Unnormalised bakes overshoot 1 and must be clamped before export."""
    mesh = _flat_plane()

    colors = bake_colors(mesh, 32)

    assert len(colors) == len(mesh["positions"])
    for color in colors:
        assert len(color) == 3
        for channel in color:
            assert 0.0 - TOL <= channel <= 1.0 + TOL


def test_smoothing_leaves_boundary_colors_untouched():
    """Boundary vertices must not average in neighbours and bleed color."""
    colors = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    indices = [(0, 1, 2)]  # every edge is an open-border edge

    smoothed = smooth_colors(colors, indices, iterations=3)

    assert len(smoothed) == 3
    for got, want in zip(smoothed, colors):
        assert tuple(got) == pytest.approx(want, abs=TOL)
    assert colors == [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]


def test_validate_flags_out_of_range_colors():
    """Out-of-range colors must be reported, not silently clipped."""
    mesh = _flat_plane()
    colors = [(0.0, 0.0, 0.0)] * 3 + [(1.5, 0.0, 0.0)]

    problems = validate(mesh, colors)

    assert problems != []
    assert any("color" in str(issue).lower() for issue in problems)


def test_validate_flags_color_count_mismatch():
    """A color count mismatch must be reported instead of shifting indices."""
    mesh = _flat_plane()
    colors = [(0.2, 0.2, 0.2)] * 3  # the mesh has 4 vertices

    problems = validate(mesh, colors)

    assert problems != []


def test_batch_hands_each_mesh_its_own_color_list():
    """Batches must not share one color array between meshes."""
    meshes = [_flat_plane(), _small_triangle(x=5.0)]

    result = bake_vcolor_batch(meshes, {"samples": 8, "smooth_iterations": 1})
    colors = result["colors"]

    assert len(colors) == 2
    assert colors[0] is not colors[1]

    before = tuple(colors[1][0])
    colors[0][0] = (1.0, 1.0, 1.0)
    assert tuple(colors[1][0]) == before


def test_zero_sample_count_is_rejected():
    """samples=0 must raise instead of being accepted and dividing by zero."""
    with pytest.raises(VColorError):
        ao_samples((0.0, 1.0, 0.0), 0, seed=1)


def test_flat_up_facing_mesh_is_fully_open():
    """A flat up-facing mesh is unoccluded, so its AO value is 1.0."""
    mesh = _flat_plane()
    normal = (0.0, 1.0, 0.0)

    samples = ao_samples(normal, 24, seed=7)
    ao = 1.0 - occlusion(mesh, (0.25, 0.0, 0.25), normal, samples)

    assert ao == pytest.approx(1.0, abs=TOL)


def test_light_facing_normal_is_the_brightest():
    """The normal pointing at the light must win with full brightness."""
    toward = shade_by_normal((0.0, 0.0, 1.0))
    away = shade_by_normal((0.0, 0.0, -1.0))
    side = shade_by_normal((1.0, 0.0, 0.0))

    assert toward == pytest.approx(1.0, abs=TOL)
    assert toward > away
    assert toward > side


def test_empty_mesh_raises_vcolor_error():
    """An empty mesh is invalid input, not an empty bake."""
    empty = {"positions": [], "normals": [], "indices": []}

    with pytest.raises(VColorError):
        bake_colors(empty, 8)


def test_zero_samples_bake_raises_vcolor_error():
    """samples=0 must fail fast before any per-sample division."""
    with pytest.raises(VColorError):
        bake_colors(_flat_plane(), 0)


def test_successful_batch_preserves_input_order():
    """A fully valid batch returns meshes and colors in the input order."""
    first = _flat_plane()
    second = _small_triangle(x=5.0)

    result = bake_vcolor_batch(
        [first, second], {"samples": 8, "smooth_iterations": 0}
    )

    assert result["meshes"] == [first, second]
    assert [len(colors) for colors in result["colors"]] == [4, 3]

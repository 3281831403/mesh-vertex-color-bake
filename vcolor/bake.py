"""Vertex-color baking primitives for the AO bake bugfix exercise.

Conventions: colors are ``(r, g, b)`` triples whose channels are in the closed
range [0, 1]; ambient occlusion is a float in [0, 1] where 1 means fully open
and 0 means fully blocked.  A mesh dictionary contains ``positions`` (3D
vertex coordinates), ``normals`` (per-vertex normal hints) and ``indices``
(triangles indexing the vertex arrays).  Floating point comparisons use an
absolute tolerance of 1e-6.
"""

import math

TOL = 1e-6

_MESH_KEYS = ("positions", "normals", "indices")


class VColorError(Exception):
    """Raised when a mesh, color list or bake configuration is invalid."""


# ---------------------------------------------------------------------------
# small vector helpers (plain tuples, no dependencies)


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(v):
    return math.sqrt(_dot(v, v))


def _normalize(v):
    n = _length(v)
    if n <= 0.0:
        return None
    return (v[0] / n, v[1] / n, v[2] / n)


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _check_mesh(mesh):
    """Raise VColorError unless ``mesh`` is structurally usable."""
    if not isinstance(mesh, dict):
        raise VColorError("mesh must be a dict with positions/normals/indices")
    for key in _MESH_KEYS:
        if key not in mesh:
            raise VColorError(f"mesh is missing key {key!r}")
    positions = mesh["positions"]
    normals = mesh["normals"]
    indices = mesh["indices"]
    if not isinstance(positions, (list, tuple)) or not isinstance(
        indices, (list, tuple)
    ):
        raise VColorError("mesh positions/indices must be sequences")
    for p in positions:
        if (
            not isinstance(p, (list, tuple))
            or len(p) != 3
            or not all(_is_number(c) for c in p)
        ):
            raise VColorError(f"mesh positions entry is not a numeric 3-vector: {p!r}")
    n = len(positions)
    for face in indices:
        if not isinstance(face, (list, tuple)) or len(face) != 3:
            raise VColorError(f"mesh indices entry is not a triangle: {face!r}")
        for i in face:
            if not isinstance(i, int) or isinstance(i, bool):
                raise VColorError(f"mesh index is not an integer: {i!r}")
    return positions, normals, indices


def _iter_valid_faces(mesh):
    """Yield ``(a, b, c)`` position triples for each non-degenerate face."""
    positions = mesh["positions"]
    n = len(positions)
    for face in mesh["indices"]:
        i, j, k = face
        if len({i, j, k}) != 3:
            continue  # repeated vertex indices
        if not (0 <= i < n and 0 <= j < n and 0 <= k < n):
            continue  # out-of-range indices
        a, b, c = positions[i], positions[j], positions[k]
        if _length(_cross(_sub(b, a), _sub(c, a))) <= 0.0:
            continue  # zero-area triangle
        yield (i, j, k), (a, b, c)


# ---------------------------------------------------------------------------
# public API


def vertex_normals(mesh) -> list:
    """Compute one area-weighted unit normal per vertex of ``mesh``.

    For each triangle, compute its face normal scaled by the triangle area, so
    that a large face outweighs a small one (equal-weight averaging lets a
    tiny sliver dominate the average and is the bug to fix).  Sum the weighted
    contributions per incident vertex and normalise the result to unit length.
    Degenerate triangles (zero area, repeated vertex indices or out-of-range
    indices) contribute nothing.  A vertex without any usable incident face
    keeps ``(0.0, 0.0, 0.0)``.  Return exactly ``len(mesh["positions"])``
    normals in vertex order; the mesh is not mutated.  The mesh fields are
    positions/normals/indices.  The returned normals feed the color pipeline,
    where colors are in 0..1 and AO values are in 0..1; numeric comparisons use
    tolerance 1e-6.  A structurally malformed mesh raises VColorError.
    """
    positions, _, _ = _check_mesh(mesh)
    sums = [[0.0, 0.0, 0.0] for _ in positions]
    for (i, j, k), (a, b, c) in _iter_valid_faces(mesh):
        # cross(b-a, c-a) has magnitude 2*area and points along the face
        # normal: weighting by it makes large faces dominate, not slivers.
        wx, wy, wz = _cross(_sub(b, a), _sub(c, a))
        for idx in (i, j, k):
            sums[idx][0] += wx
            sums[idx][1] += wy
            sums[idx][2] += wz
    result = []
    for s in sums:
        unit = _normalize((s[0], s[1], s[2]))
        result.append(unit if unit is not None else (0.0, 0.0, 0.0))
    return result


def ao_samples(normal, count, seed) -> list:
    """Return ``count`` deterministic hemisphere sample directions around ``normal``.

    ``normal`` is normalised first (a zero-length normal raises VColorError).
    Each sample is a unit-length ``(x, y, z)`` tuple in the hemisphere that the
    normal points into, i.e. ``dot(sample, normal) >= 0``: the hemisphere must
    follow the surface normal instead of always pointing at -Z (the bug to
    fix).  Sampling is deterministic -- the same ``normal``, ``count`` and
    ``seed`` always return the same sequence and no RNG state is used.  The
    AO values produced from these directions are floats in 0..1 with tolerance
    1e-6.  ``count`` must be a positive integer; ``0`` or a negative value
    raises VColorError so later code never divides by zero.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise VColorError(f"samples count must be a positive integer, got {count!r}")
    n = _normalize(tuple(float(c) for c in normal))
    if n is None:
        raise VColorError("normal must have non-zero length")

    # Orthonormal basis (t, b, n) so the hemisphere follows the normal.
    hint = (0.0, 0.0, 1.0) if abs(n[2]) < 0.9 else (1.0, 0.0, 0.0)
    t = _normalize(_cross(hint, n))
    b = _cross(n, t)

    # Deterministic Fibonacci spiral over the hemisphere; the seed only
    # rotates the spiral around the normal, so results are reproducible.
    golden = math.pi * (3.0 - math.sqrt(5.0))
    phase = (float(seed) % 4096.0) * golden
    samples = []
    for i in range(count):
        cos_theta = 1.0 - (i + 0.5) / count  # in (0, 1): strictly above the surface
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        phi = phase + i * golden
        lx, ly = sin_theta * math.cos(phi), sin_theta * math.sin(phi)
        sample = (
            t[0] * lx + b[0] * ly + n[0] * cos_theta,
            t[1] * lx + b[1] * ly + n[1] * cos_theta,
            t[2] * lx + b[2] * ly + n[2] * cos_theta,
        )
        samples.append(_normalize(sample))
    return samples


def _ray_hits_triangle(origin, direction, a, b, c):
    """Möller–Trumbore intersection; True when hit strictly in front."""
    edge1 = _sub(b, a)
    edge2 = _sub(c, a)
    pvec = _cross(direction, edge2)
    det = _dot(edge1, pvec)
    if abs(det) < 1e-12:
        return False
    inv = 1.0 / det
    tvec = _sub(origin, a)
    u = _dot(tvec, pvec) * inv
    if u < -TOL or u > 1.0 + TOL:
        return False
    qvec = _cross(tvec, edge1)
    v = _dot(direction, qvec) * inv
    if v < -TOL or u + v > 1.0 + TOL:
        return False
    t = _dot(edge2, qvec) * inv
    return t > 1e-9


def occlusion(mesh, point, normal, samples) -> float:
    """Return the fraction of shadow rays from ``point`` blocked by ``mesh``.

    Cast one ray per direction in ``samples`` starting at
    ``point + normal * 1e-6``: the tiny offset keeps the surface that the point
    lies on from blocking its own ray (counting the origin as a hit is the bug
    that turns every bake black).  A ray stops at the first triangle hit; the
    result is ``blocked / len(samples)``, an AO-style float in 0..1 where 0.0
    means fully open and 1.0 fully blocked, compared with tolerance 1e-6.  A
    flat mesh whose normals point up and whose samples stay in that hemisphere
    blocks nothing.  An empty ``samples`` sequence or a malformed mesh raises
    VColorError.  The mesh uses positions/normals/indices and is not mutated.
    """
    _check_mesh(mesh)
    if not samples:
        raise VColorError("samples must be a non-empty sequence of directions")
    n = _normalize(tuple(float(c) for c in normal))
    if n is None:
        raise VColorError("normal must have non-zero length")
    # Offset the origin along the normal so the surface never occludes itself.
    origin = (
        float(point[0]) + n[0] * 1e-6,
        float(point[1]) + n[1] * 1e-6,
        float(point[2]) + n[2] * 1e-6,
    )
    faces = [tri for _, tri in _iter_valid_faces(mesh)]
    blocked = 0
    for sample in samples:
        d = _normalize(tuple(float(c) for c in sample))
        if d is None:
            raise VColorError("sample direction must have non-zero length")
        for a, b, c in faces:
            if _ray_hits_triangle(origin, d, a, b, c):
                blocked += 1
                break  # first hit stops the ray
    return blocked / len(samples)


def shade_by_normal(normal) -> float:
    """Return the Lambert term of the fixed directional light for ``normal``.

    The light direction is fixed at (0, 0, 1) (shining along +Z).  Normalise
    ``normal`` and return ``max(0.0, dot(normal, light))`` clamped into 0..1:
    a back-facing vertex must never receive a negative brightness (leaving the
    dot product unclamped is the bug to fix).  A zero-length normal returns
    0.0.  Vectors are 3D and numeric tolerance is 1e-6.
    """
    n = _normalize(tuple(float(c) for c in normal))
    if n is None:
        return 0.0
    return min(1.0, max(0.0, _dot(n, (0.0, 0.0, 1.0))))


def _clamp01(x):
    return min(1.0, max(0.0, x))


def bake_colors(mesh, samples) -> list:
    """Bake one RGB vertex color per vertex of ``mesh``.

    For every vertex, combine its AO term (from ``occlusion`` over ``samples``
    directions) with ``shade_by_normal`` and clamp/normalise the mix so that
    each channel ends up in 0..1 (an unnormalised bake overshoots 1 and is the
    bug to fix).  Return a list of ``(r, g, b)`` triples with exactly
    ``len(mesh["positions"])`` entries in vertex order; the mesh is not
    mutated.  ``samples`` must be a positive integer and ``0`` raises
    VColorError instead of dividing by zero.  An empty or structurally
    malformed mesh raises VColorError.  The mesh uses
    positions/normals/indices; color and AO comparisons use tolerance 1e-6.
    """
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise VColorError(f"samples must be a positive integer, got {samples!r}")
    _check_mesh(mesh)
    positions = mesh["positions"]
    if len(positions) == 0:
        raise VColorError("mesh has no vertices")

    normals = vertex_normals(mesh)
    colors = []
    for idx, position in enumerate(positions):
        normal = normals[idx]
        if normal == (0.0, 0.0, 0.0):
            ao = 1.0
            light = 0.0
        else:
            directions = ao_samples(normal, samples, seed=idx)
            ao = 1.0 - occlusion(mesh, position, normal, directions)
            light = shade_by_normal(normal)
        # AO scales the incoming light; clamp so the bake never overshoots 1.
        value = _clamp01(ao * light)
        colors.append((value, value, value))
    return colors


def smooth_colors(colors, indices, iterations) -> list:
    """Smooth ``colors`` across triangle adjacency without moving boundaries.

    One iteration replaces an interior vertex color with the average of itself
    and its one-ring neighbours; ``iterations`` iterations are applied in
    order.  Vertices on an open boundary (a mesh edge used by exactly one
    triangle) keep their incoming color for every iteration, so colors cannot
    bleed past the mesh border (averaging boundary vertices into the mesh is
    the bug to fix).  The input list is not mutated; return a new list of RGB
    triples in 0..1 with tolerance 1e-6.  ``iterations`` must be a
    non-negative integer, and a color/vertex count mismatch or an invalid
    index raises VColorError.  ``indices`` are triangles over the color array.
    """
    if (
        not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or iterations < 0
    ):
        raise VColorError(
            f"iterations must be a non-negative integer, got {iterations!r}"
        )
    count = len(colors)
    neighbours = [set() for _ in range(count)]
    edge_use = {}
    for face in indices:
        if not isinstance(face, (list, tuple)) or len(face) != 3:
            raise VColorError(f"indices entry is not a triangle: {face!r}")
        for i in face:
            if (
                not isinstance(i, int)
                or isinstance(i, bool)
                or i < 0
                or i >= count
            ):
                raise VColorError(
                    f"index {i!r} does not match the {count} provided colors"
                )
        i, j, k = face
        for u, v in ((i, j), (j, k), (k, i)):
            neighbours[u].add(v)
            neighbours[v].add(u)
            edge = (u, v) if u < v else (v, u)
            edge_use[edge] = edge_use.get(edge, 0) + 1

    # Vertices touched by an edge used by exactly one triangle are boundary
    # vertices: they keep their color so nothing bleeds past the border.
    boundary = set()
    for (u, v), uses in edge_use.items():
        if uses == 1:
            boundary.add(u)
            boundary.add(v)

    current = [tuple(float(c) for c in color) for color in colors]
    for _ in range(iterations):
        nxt = list(current)
        for v in range(count):
            if v in boundary or not neighbours[v]:
                continue
            ring = [current[v]] + [current[u] for u in neighbours[v]]
            n = len(ring)
            nxt[v] = (
                sum(c[0] for c in ring) / n,
                sum(c[1] for c in ring) / n,
                sum(c[2] for c in ring) / n,
            )
        current = nxt
    return [tuple(_clamp01(c) for c in color) for color in current]


def validate(mesh, colors) -> list:
    """Return human-readable problems found in ``mesh`` and ``colors``.

    Report (never silently clamp) every color channel outside 0..1, using
    tolerance 1e-6; report a color count that differs from the vertex count
    (silently shifting indices is a bug); report degenerate faces -- repeated
    indices, out-of-range indices or zero area; report missing mesh keys,
    non-numeric coordinates and color entries that are not three channels.
    Each problem is a short string that mentions the affected field, e.g.
    "color".  Return an empty list when the input is valid.  The mesh uses
    positions/normals/indices; colors are RGB triples in 0..1; tolerance 1e-6.
    """
    problems = []

    if not isinstance(mesh, dict):
        return ["mesh is not a dict with positions/normals/indices"]
    for key in _MESH_KEYS:
        if key not in mesh:
            problems.append(f"mesh is missing key {key!r}")
    if problems:
        return problems

    positions = mesh["positions"]
    indices = mesh["indices"]
    if not isinstance(positions, (list, tuple)):
        problems.append("positions must be a sequence of 3-vectors")
        positions = []
    if not isinstance(indices, (list, tuple)):
        problems.append("indices must be a sequence of triangles")
        indices = []

    for vi, p in enumerate(positions):
        if not isinstance(p, (list, tuple)) or len(p) != 3:
            problems.append(f"positions[{vi}] is not a 3-vector")
        elif not all(_is_number(c) for c in p):
            problems.append(f"positions[{vi}] has non-numeric coordinates")

    n = len(positions)
    for fi, face in enumerate(indices):
        if not isinstance(face, (list, tuple)) or len(face) != 3:
            problems.append(f"indices[{fi}] is not a triangle")
            continue
        if any(not isinstance(i, int) or isinstance(i, bool) for i in face):
            problems.append(f"indices[{fi}] contains a non-integer index")
            continue
        if any(i < 0 or i >= n for i in face):
            problems.append(f"indices[{fi}] references an out-of-range vertex")
            continue
        if len(set(face)) != 3:
            problems.append(f"indices[{fi}] repeats a vertex index")
            continue
        a, b, c = (positions[i] for i in face)
        if all(
            isinstance(p, (list, tuple)) and len(p) == 3 for p in (a, b, c)
        ) and _length(_cross(_sub(b, a), _sub(c, a))) <= TOL:
            problems.append(f"indices[{fi}] is a zero-area degenerate face")

    if len(colors) != n:
        problems.append(
            f"color count {len(colors)} does not match vertex count {n}"
        )
    for ci, color in enumerate(colors):
        if not isinstance(color, (list, tuple)) or len(color) != 3:
            problems.append(f"color[{ci}] is not a three-channel RGB triple")
            continue
        for channel in color:
            if not _is_number(channel):
                problems.append(f"color[{ci}] has a non-numeric channel")
                break
            if channel < 0.0 - TOL or channel > 1.0 + TOL:
                problems.append(
                    f"color[{ci}] channel {channel} is outside the 0..1 range"
                )
    return problems


def bake_vcolor_batch(meshes, config) -> dict:
    """Bake a list of meshes as one all-or-nothing transaction.

    ``config`` contains a positive integer ``samples`` (samples == 0 raises
    VColorError) and an integer ``smooth_iterations`` >= 0.  Validate the
    config and every mesh (including structural emptiness and colors) before
    producing any output; one invalid mesh raises VColorError and leaves no
    partial result behind.  On success return
    ``{"meshes": [...], "colors": [...]}`` where both lists follow the input
    ``meshes`` order, each colors entry is that mesh's own list of RGB triples
    in 0..1, and no color list is shared or aliased between meshes (a shared
    buffer lets one model contaminate another).  An empty ``meshes`` list
    raises VColorError.  Colors and AO values are in 0..1 with tolerance 1e-6.
    """
    if not isinstance(config, dict):
        raise VColorError("config must be a dict")
    samples = config.get("samples")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise VColorError(f"config samples must be a positive integer, got {samples!r}")
    smooth_iterations = config.get("smooth_iterations", 0)
    if (
        not isinstance(smooth_iterations, int)
        or isinstance(smooth_iterations, bool)
        or smooth_iterations < 0
    ):
        raise VColorError(
            "config smooth_iterations must be a non-negative integer, "
            f"got {smooth_iterations!r}"
        )
    if not isinstance(meshes, (list, tuple)) or len(meshes) == 0:
        raise VColorError("meshes must be a non-empty list")

    # Validate everything up front: one bad mesh aborts the whole batch.
    for mesh in meshes:
        _check_mesh(mesh)
        if len(mesh["positions"]) == 0:
            raise VColorError("mesh has no vertices")

    baked_meshes = []
    baked_colors = []
    for mesh in meshes:
        colors = bake_colors(mesh, samples)
        colors = smooth_colors(colors, mesh["indices"], smooth_iterations)
        problems = validate(mesh, colors)
        if problems:
            raise VColorError("invalid bake result: " + "; ".join(problems))
        baked_meshes.append(mesh)
        # Each mesh gets its own freshly built list -- nothing is shared.
        baked_colors.append([tuple(color) for color in colors])
    return {"meshes": baked_meshes, "colors": baked_colors}

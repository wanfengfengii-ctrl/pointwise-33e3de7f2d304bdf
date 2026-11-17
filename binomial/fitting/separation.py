"""Exact boundary-face identification for binomial fits.

The extended maximum-likelihood fit either fixes a row probability at 0 or
1 (boundary) or leaves it strictly inside (0, 1) (internal).  A row is
internal exactly when every recession direction of the log-likelihood
vanishes on it.  The recession cone is the set of column-space directions
d with d_i >= 0 on all-success rows, d_i <= 0 on all-failure rows and
d_i = 0 on rows with mixed counts.  The cone is computed exactly once per
design as its lineality space plus its extreme rays (enumerated over
active-constraint subsets in exact rational arithmetic), and each row
membership question reduces to dot products against those directions.
No truncation, penalty or observation deletion is involved.
"""
from fractions import Fraction
from itertools import combinations


def _dot(a, b):
    return sum((u * v for u, v in zip(a, b)), Fraction())


def _canonical(row):
    first = next((value for value in row if value), None)
    if first is None:
        return None
    scale = abs(first)
    return tuple(value / scale for value in row)


def _nullspace_basis(matrix, width):
    """Exact Fraction basis of {v : matrix v == 0} by Gauss-Jordan."""
    rows = [list(map(Fraction, row)) for row in matrix if any(row)]
    pivots = []
    current = 0
    for col in range(width):
        pivot = next((i for i in range(current, len(rows)) if rows[i][col]), None)
        if pivot is None:
            continue
        rows[current], rows[pivot] = rows[pivot], rows[current]
        scale = rows[current][col]
        rows[current] = [v / scale for v in rows[current]]
        for i in range(len(rows)):
            if i != current and rows[i][col]:
                scale = rows[i][col]
                rows[i] = [u - scale * v for u, v in zip(rows[i], rows[current])]
        pivots.append(col)
        current += 1
        if current == len(rows):
            break
    basis = []
    for col in range(width):
        if col in pivots:
            continue
        vector = [Fraction()] * width
        vector[col] = Fraction(1)
        for i, pivot_col in enumerate(pivots):
            vector[pivot_col] = -rows[i][col]
        basis.append(tuple(vector))
    return tuple(basis)


def _cone_directions(a_rows, width):
    """Return (lineality, rays, examined) of the cone {beta: a_rows beta >= 0}.

    Every direction of a polyhedral cone is a lineality vector plus a
    nonnegative combination of extreme rays, so a sign pattern is
    achievable exactly when it is achievable by these directions.
    """
    lineality = _nullspace_basis(a_rows, width)
    rank = width - len(lineality)
    if rank == 0:
        return lineality, (), 0
    rays = []
    seen = set()
    examined = 0
    for subset in combinations(range(len(a_rows)), rank - 1):
        examined += 1
        null = _nullspace_basis([a_rows[i] for i in subset], width)
        if len(null) != width - rank + 1:
            continue
        for vector in null:
            if all(not _dot(row, vector) for row in a_rows):
                continue
            for candidate in (vector, tuple(-v for v in vector)):
                if all(_dot(row, candidate) >= 0 for row in a_rows):
                    key = _canonical(candidate)
                    if key is not None and key not in seen:
                        seen.add(key)
                        rays.append(key)
            break
    return lineality, tuple(rays), examined


def identify_boundary(design):
    """Return (boundary flags, examined count) for a validated frozen design.

    The boundary tuple holds None for internal rows and the determined 0 or
    1 for rows whose probability is fixed at every feasible maximizer.  The
    count records how many exact cone directions were examined.
    """
    width = len(design.column_ids)
    a_rows = []
    for x, n, k in zip(design.x, design.n, design.k):
        row = tuple(Fraction(v) for v in x)
        if k == n:
            a_rows.append(row)
        elif k == 0:
            a_rows.append(tuple(-v for v in row))
        else:
            a_rows.append(row)
            a_rows.append(tuple(-v for v in row))
    lineality, rays, examined = _cone_directions(a_rows, width)
    boundary = []
    for x, n, k in zip(design.x, design.n, design.k):
        if 0 < k < n:
            boundary.append(None)
            continue
        test = tuple((1 if k == n else -1) * Fraction(v) for v in x)
        separable = (any(_dot(test, u) for u in lineality)
                     or any(_dot(test, r) > 0 for r in rays))
        boundary.append((1 if k == n else 0) if separable else None)
    return tuple(boundary), examined

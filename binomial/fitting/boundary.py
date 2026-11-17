import math
from dataclasses import dataclass
from fractions import Fraction
from binomial.design.compiler import FrozenDesign, validate_frozen_design
from binomial.fitting.subspace import row_space
from binomial.fitting.finite import fit_finite, NumericalFitError, solve_positive_system
from binomial.fitting.likelihood import binomial_log_likelihood, score_and_information
from binomial.fitting.separation import identify_boundary


@dataclass(frozen=True)
class FitArtifact:
    frozen: FrozenDesign
    probabilities: tuple
    boundary: tuple
    internal_row_ids: tuple
    basis: tuple
    theta: tuple
    information: tuple
    diagnostics: tuple

    def to_value(self):
        return {"frozen": self.frozen.to_value(), "probabilities": list(self.probabilities),
                "boundary": list(self.boundary), "internal_row_ids": list(self.internal_row_ids),
                "basis": [list(row) for row in self.basis], "theta": list(self.theta),
                "information": [list(row) for row in self.information],
                "diagnostics": dict(self.diagnostics)}

    @classmethod
    def from_value(cls, value):
        result = cls(FrozenDesign.from_value(value["frozen"]), tuple(value["probabilities"]),
                     tuple(value["boundary"]), tuple(value["internal_row_ids"]),
                     tuple(tuple(row) for row in value["basis"]), tuple(value["theta"]),
                     tuple(tuple(row) for row in value["information"]),
                     tuple(sorted(value["diagnostics"].items())))
        validate_fit_artifact(result)
        return result



def fit_extended(design, finite_solver=fit_finite):
    validate_frozen_design(design)
    boundary, examined = identify_boundary(design)
    internal = tuple(i for i, flag in enumerate(boundary) if flag is None)
    rows = tuple(design.x[i] for i in internal)
    space = row_space(rows, len(design.column_ids))
    a = tuple(space.coordinates(row) for row in rows)
    finite = finite_solver(a, tuple(design.n[i] for i in internal),
                           tuple(design.k[i] for i in internal), len(space.basis))
    positions = {row_index: position for position, row_index in enumerate(internal)}
    probabilities = tuple(finite.probabilities[positions[i]] if flag is None else flag
                          for i, flag in enumerate(boundary))
    if len(internal) == len(design.x):
        status = 'finite'
    elif not internal:
        status = 'complete'
    else:
        status = 'quasi'
    diagnostics = {'status': status, 'design_rank': len(row_space(design.x).basis),
                   'finite_rank': len(space.basis), 'iterations': finite.iterations,
                   'score_norm': finite.score_norm, 'vertices_examined': examined,
                   'log_likelihood': binomial_log_likelihood(design.n, design.k, probabilities)}
    result = FitArtifact(design, probabilities, boundary,
                         tuple(design.row_ids[i] for i in internal), space.basis,
                         finite.theta, finite.information, tuple(sorted(diagnostics.items())))
    return validate_fit_artifact(result)


def validate_fit_artifact(fit):
    design = validate_frozen_design(fit.frozen)
    m = len(design.x)
    if len(fit.boundary) != m or any(v is not None and type(v) is not int for v in fit.boundary):
        raise ValueError('invalid boundary declaration')
    if any(v not in (None, 0, 1) for v in fit.boundary):
        raise ValueError('invalid boundary value')
    indices = tuple(i for i, flag in enumerate(fit.boundary) if flag is None)
    if fit.internal_row_ids != tuple(design.row_ids[i] for i in indices):
        raise ValueError('fit row identity mismatch')
    space = row_space([design.x[i] for i in indices], len(design.column_ids))
    if fit.basis != space.basis or any(type(v) is not Fraction for row in fit.basis for v in row):
        raise ValueError('invalid exact finite space')
    d = len(space.basis)
    if len(fit.theta) != d or any(not math.isfinite(v) for v in fit.theta):
        raise NumericalFitError('invalid finite coordinates')
    a = tuple(space.coordinates(design.x[i]) for i in indices)
    p, score, information = score_and_information(
        a, tuple(design.n[i] for i in indices), tuple(design.k[i] for i in indices), fit.theta)
    if len(fit.probabilities) != m or len(fit.information) != d:
        raise ValueError('fit dimension mismatch')
    for i, flag in enumerate(fit.boundary):
        actual = fit.probabilities[i]
        if type(actual) not in (float, int) or not math.isfinite(actual):
            raise ValueError('invalid probability')
        if flag is not None and actual != flag:
            raise ValueError('fixed probability mismatch')
        if flag is None and not 0 < actual < 1:
            raise ValueError('internal probability mismatch')
    if any(abs(fit.probabilities[i] - expected) > 2e-12 for i, expected in zip(indices, p)):
        raise ValueError('probability coordinate mismatch')
    if any(len(row) != d for row in fit.information):
        raise ValueError('information dimension mismatch')
    for row, expected in zip(fit.information, information):
        if any(not math.isfinite(v) or abs(v - e) > 2e-12 * max(1, abs(e))
               for v, e in zip(row, expected)):
            raise ValueError('information source mismatch')
    solve_positive_system(fit.information, (0.0,) * d)
    diag = dict(fit.diagnostics)
    status = 'finite' if len(indices) == m else ('complete' if not indices else 'quasi')
    if (diag.get('status') != status or diag.get('finite_rank') != d
            or diag.get('design_rank') != len(row_space(design.x).basis)
            or type(diag.get('iterations')) is not int or diag['iterations'] < 0
            or type(diag.get('vertices_examined')) is not int or diag['vertices_examined'] < 0
            or abs(diag.get('score_norm', math.inf) - max(map(abs, score), default=0)) > 1e-10
            or abs(diag.get('log_likelihood', math.inf) -
                   binomial_log_likelihood(design.n, design.k, fit.probabilities)) > 1e-10):
        raise ValueError('invalid fit diagnostics')
    return fit

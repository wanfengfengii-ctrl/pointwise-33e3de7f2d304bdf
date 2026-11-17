import math
from dataclasses import dataclass
from fractions import Fraction
from binomial.fitting.boundary import validate_fit_artifact
from binomial.fitting.subspace import row_space
from binomial.fitting.finite import solve_positive_system
from binomial.reporting.covariance import effect_variance


@dataclass(frozen=True)
class EffectResult:
    name: str
    ordinal: int
    coefficients: tuple
    status: str
    estimate: float | None
    variance: float | None

    def to_value(self, column_ids):
        return {"name": self.name, "ordinal": self.ordinal,
                "coefficients": {c: v for c, v in zip(column_ids, self.coefficients) if v},
                "status": self.status, "estimate": self.estimate, "variance": self.variance}


@dataclass(frozen=True)
class EffectReport:
    fit_id: str
    frozen_digest: str
    revision_id: str
    design_digest: str
    column_ids: tuple
    items: tuple

    def to_value(self):
        return {"fit_id": self.fit_id, "frozen_digest": self.frozen_digest,
                "revision_id": self.revision_id, "design_digest": self.design_digest,
                "column_ids": list(self.column_ids),
                "items": [item.to_value(self.column_ids) for item in self.items]}

    @classmethod
    def from_value(cls, value):
        columns = tuple(value["column_ids"])
        items = tuple(EffectResult(
            item["name"], item["ordinal"], tuple(item["coefficients"].get(c, Fraction()) for c in columns),
            item["status"], item["estimate"], item["variance"]) for item in value["items"])
        result = cls(value["fit_id"], value["frozen_digest"], value["revision_id"],
                     value["design_digest"], columns, items)
        if result.to_value() != value:
            raise ValueError("noncanonical report")
        return result


def _report_coordinates(fit):
    space = row_space(fit.frozen.x, len(fit.frozen.column_ids))
    a = tuple(space.coordinates(row) for row in fit.frozen.x)
    d = len(space.basis)
    probabilities = tuple(max(1e-8, min(1 - 1e-8, p)) for p in fit.probabilities)
    logits = tuple(math.log(p) - math.log1p(-p) for p in probabilities)
    gram = tuple(tuple(math.fsum(float(row[j]) * float(row[l]) for row in a)
                       for l in range(d)) for j in range(d))
    rhs = tuple(math.fsum(float(row[j]) * z for row, z in zip(a, logits)) for j in range(d))
    theta = solve_positive_system(gram, rhs)
    information = tuple(tuple(math.fsum(float(row[j]) * float(row[l]) * n * p * (1 - p)
                                        for row, n, p in zip(a, fit.frozen.n, probabilities))
                              for l in range(d)) for j in range(d))
    return space, theta, information


def evaluate_effects(fit, request):
    validate_fit_artifact(fit)
    if request.column_ids != fit.frozen.column_ids:
        raise ValueError("effect column binding mismatch")
    space, theta, information = _report_coordinates(fit)
    items = []
    for index, item in enumerate(request.effects):
        coordinates = space.coordinates(item.coefficients)
        if coordinates is None:
            result = EffectResult(item.name, index, item.coefficients, "not_estimable", None, None)
        else:
            estimate = math.fsum(float(w) * value for w, value in zip(coordinates, theta))
            variance = effect_variance(information, coordinates)
            if not math.isfinite(estimate):
                raise ArithmeticError("nonfinite effect estimate")
            result = EffectResult(item.name, index, item.coefficients, "estimable", estimate, variance)
        items.append(result)
    return EffectReport(request.fit_id, fit.frozen.digest, fit.frozen.revision_id,
                        fit.frozen.design_digest, fit.frozen.column_ids, tuple(items))


def validate_effect_report(report, fit):
    from binomial.reporting.requests import validate_report_request
    request = validate_report_request(report.fit_id, report.column_ids,
        [{"name": item.name, "coefficients": item.to_value(report.column_ids)["coefficients"]}
         for item in report.items])
    expected = evaluate_effects(fit, request)
    if expected.to_value() != report.to_value():
        raise ValueError("report differs from frozen fit")
    return report

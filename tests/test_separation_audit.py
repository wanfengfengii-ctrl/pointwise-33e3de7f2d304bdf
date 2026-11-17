"""Independent reconciliation audits for the boundary fit and effect chain.

Every check derives its expectations independently of the production fit:
exact separation certificates, moment conditions, plain-float recomputation
of probabilities and information, and high-precision Decimal reference
solvers.  Covers the incident, complete and quasi separation, pure counts,
rank deficiency, zero effects, column transport and an exhaustive sweep of
small one-column designs.
"""
import math
import tempfile
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
from pathlib import Path

from binomial.archive.fit_records import load_fit, read_result
from binomial.archive.store import ArchiveStore
from binomial.design.compiler import FrozenDesign
from binomial.fitting.boundary import fit_extended
from binomial.reporting.service import report
from tests.support import MemoryCase, observation, effect, setup_analysis


def make_design(rows, columns):
    ordered = sorted(rows, key=lambda row: row[0])
    return FrozenDesign("audit-cohort", "audit-revision", "audit-design", "0" * 64,
                        tuple(row[0] for row in ordered), tuple(columns),
                        tuple(tuple(row[1]) for row in ordered),
                        tuple(row[2] for row in ordered),
                        tuple(row[3] for row in ordered))


def decimal_mle_1d(a, ns, ks):
    """Independent one-dimensional reference fit by Decimal bisection."""
    with localcontext() as context:
        context.prec = 40
        a = [Decimal(x.numerator) / Decimal(x.denominator) if isinstance(x, Fraction)
             else Decimal(x) for x in a]

        def score(theta):
            return sum(Decimal(x) * (Decimal(k) - Decimal(n) / (1 + (-Decimal(x) * theta).exp()))
                       for x, n, k in zip(a, ns, ks))

        low, high = Decimal(-40), Decimal(40)
        for _ in range(100):
            mid = (low + high) / 2
            if score(mid) > 0:
                low = mid
            else:
                high = mid
        theta = (low + high) / 2
        probabilities = [1 / (1 + (-Decimal(x) * theta).exp()) for x in a]
        information = sum(Decimal(n) * Decimal(x * x) * p * (1 - p)
                          for x, n, p in zip(a, ns, probabilities))
        return theta, probabilities, information


def decimal_mle_2d(rows, ns, ks):
    """Independent two-dimensional reference fit by Decimal Newton iteration."""
    with localcontext() as context:
        context.prec = 50
        theta = [Decimal(0), Decimal(0)]
        for _ in range(200):
            p = [1 / (1 + (-(Decimal(r[0]) * theta[0] + Decimal(r[1]) * theta[1])).exp())
                 for r in rows]
            score = [sum(Decimal(r[j]) * (Decimal(k) - Decimal(n) * pi)
                         for r, n, k, pi in zip(rows, ns, ks, p)) for j in (0, 1)]
            info = [[sum(Decimal(r[j]) * Decimal(r[l]) * Decimal(n) * pi * (1 - pi)
                         for r, n, pi in zip(rows, ns, p)) for l in (0, 1)] for j in (0, 1)]
            det = info[0][0] * info[1][1] - info[0][1] * info[1][0]
            step = [(info[1][1] * score[0] - info[0][1] * score[1]) / det,
                    (info[0][0] * score[1] - info[1][0] * score[0]) / det]
            theta = [theta[0] + step[0], theta[1] + step[1]]
            if max(abs(step[0]), abs(step[1])) < Decimal(10) ** (-35):
                break
        p = [1 / (1 + (-(Decimal(r[0]) * theta[0] + Decimal(r[1]) * theta[1])).exp())
             for r in rows]
        info = [[sum(Decimal(r[j]) * Decimal(r[l]) * Decimal(n) * pi * (1 - pi)
                     for r, n, pi in zip(rows, ns, p)) for l in (0, 1)] for j in (0, 1)]
        return theta, p, info


def assert_certificate(case, design, boundary, beta):
    """Exact check that beta separates boundary rows from internal rows."""
    direction = [sum(Fraction(x) * Fraction(b) for x, b in zip(row, beta))
                 for row in design.x]
    for flag, d in zip(boundary, direction):
        if flag == 1:
            case.assertGreater(d, 0)
        elif flag == 0:
            case.assertLess(d, 0)
        else:
            case.assertEqual(d, 0)


def assert_moment_conditions(case, design, probabilities, delta=1e-8):
    for j in range(len(design.column_ids)):
        residual = math.fsum(x[j] * (k - n * p)
                             for x, n, k, p in zip(design.x, design.n, design.k, probabilities))
        case.assertAlmostEqual(residual, 0, delta=delta)


class TestIncidentAudit(MemoryCase):
    def test_incident_reconciled(self):
        _, fit_id = self.analyze([observation("a", {"u": 1, "v": 0}, 1, 1),
                                  observation("b", {"u": 0, "v": 1}, 2, 1),
                                  observation("c", {"u": 0, "v": -1}, 2, 1)], ["u", "v"])
        artifact = load_fit(self.store, fit_id)
        self.assertEqual(artifact.probabilities, (1, 0.5, 0.5))
        self.assertEqual(artifact.boundary, (1, None, None))
        self.assertEqual(artifact.internal_row_ids, ("b", "c"))
        self.assertEqual(artifact.basis, ((Fraction(0), Fraction(1)),))
        self.assertEqual(artifact.theta, (0.0,))
        self.assertEqual(artifact.information, ((1.0,),))
        diagnostics = dict(artifact.diagnostics)
        self.assertEqual((diagnostics["status"], diagnostics["design_rank"],
                          diagnostics["finite_rank"]), ("quasi", 2, 1))
        assert_certificate(self, artifact.frozen, artifact.boundary, (1, 0))
        assert_moment_conditions(self, artifact.frozen, artifact.probabilities)
        first, second, zero = self.results(
            fit_id, [effect("u", u=1), effect("v", v=1), effect("zero")])
        self.assertEqual((first["status"], first["estimate"], first["variance"]),
                         ("not_estimable", None, None))
        # independent recomputation from the frozen artifact alone
        self.assertEqual(second["estimate"], artifact.theta[0])
        self.assertEqual(second["variance"], 1 / artifact.information[0][0])
        self.assertEqual((zero["estimate"], zero["variance"]), (0, 0))

    def test_reopen_preserves_boundary_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.sqlite"
            with ArchiveStore(path) as store:
                _, fit_id = setup_analysis(store, [observation("a", {"u": 1, "v": 0}, 1, 1),
                                                   observation("b", {"u": 0, "v": 1}, 2, 1),
                                                   observation("c", {"u": 0, "v": -1}, 2, 1)], ["u", "v"])
                report_id = report(store, "audit", fit_id,
                                   [effect("u", u=1), effect("v", v=1), effect("zero")])
                frozen_values = (read_result(store, fit_id), read_result(store, report_id))
            with ArchiveStore(path) as reopened:
                self.assertEqual((read_result(reopened, fit_id), read_result(reopened, report_id)),
                                 frozen_values)
                self.assertEqual(read_result(reopened, fit_id)["probabilities"], [1, 0.5, 0.5])


class TestBoundaryFaces(MemoryCase):
    def test_complete_separation_audit(self):
        _, fit_id = self.analyze([observation("a", {"x": 1}, 2, 2),
                                  observation("b", {"x": -1}, 3, 0)], ["x"])
        artifact = load_fit(self.store, fit_id)
        self.assertEqual(artifact.probabilities, (1, 0))
        self.assertEqual(artifact.boundary, (1, 0))
        self.assertEqual(artifact.internal_row_ids, ())
        self.assertEqual((artifact.basis, artifact.theta, artifact.information), ((), (), ()))
        diagnostics = dict(artifact.diagnostics)
        self.assertEqual((diagnostics["status"], diagnostics["design_rank"],
                          diagnostics["finite_rank"]), ("complete", 1, 0))
        self.assertEqual(diagnostics["log_likelihood"], 0.0)
        assert_certificate(self, artifact.frozen, artifact.boundary, (1,))
        slope, zero = self.results(fit_id, [effect("x", x=1), effect("zero")])
        self.assertEqual((slope["status"], slope["estimate"], slope["variance"]),
                         ("not_estimable", None, None))
        self.assertEqual((zero["status"], zero["estimate"], zero["variance"]),
                         ("estimable", 0, 0))

    def test_quasi_with_finite_interior(self):
        _, fit_id = self.analyze([observation("a", {"u": 1, "v": 1}, 1, 1),
                                  observation("b", {"u": 0, "v": 1}, 3, 2),
                                  observation("c", {"u": 0, "v": -1}, 3, 1)], ["u", "v"])
        artifact = load_fit(self.store, fit_id)
        self.assertEqual(artifact.boundary, (1, None, None))
        self.assertEqual(artifact.internal_row_ids, ("b", "c"))
        # independent one-dimensional reference on the internal rows
        theta, probabilities, information = decimal_mle_1d([1, -1], [3, 3], [2, 1])
        self.assertAlmostEqual(artifact.theta[0], float(theta), delta=1e-8)
        self.assertAlmostEqual(artifact.probabilities[1], float(probabilities[0]), delta=1e-8)
        self.assertAlmostEqual(artifact.probabilities[2], float(probabilities[1]), delta=1e-8)
        self.assertAlmostEqual(artifact.information[0][0], float(information), delta=1e-8)
        self.assertAlmostEqual(artifact.probabilities[1], 2 / 3, delta=1e-8)
        assert_certificate(self, artifact.frozen, artifact.boundary, (1, 0))
        assert_moment_conditions(self, artifact.frozen, artifact.probabilities)
        u_item, v_item, sum_item = self.results(
            fit_id, [effect("u", u=1), effect("v", v=1), effect("sum", u=1, v=1)])
        self.assertEqual(u_item["status"], "not_estimable")
        self.assertEqual(sum_item["status"], "not_estimable")
        self.assertAlmostEqual(v_item["estimate"], math.log(2), delta=1e-8)
        self.assertAlmostEqual(v_item["variance"], 0.75, delta=1e-8)

    def test_pure_counts_and_aggregation(self):
        _, pair = self.analyze([observation("a", {"x": 1}, 2, 2),
                                observation("b", {"x": 1}, 2, 0)], ["x"])
        artifact = load_fit(self.store, pair)
        self.assertEqual(artifact.probabilities, (0.5, 0.5))
        self.assertEqual(artifact.boundary, (None, None))
        self.assertEqual(self.results(pair, [effect("x", x=1)])[0]["variance"], 1.0)
        _, success = self.analyze([observation("a", {"x": 1}, 2, 2)], ["x"],
                                  cohort="s", design="s", request="s")
        _, failure = self.analyze([observation("a", {"x": 1}, 2, 0)], ["x"],
                                  cohort="f", design="f", request="f")
        self.assertEqual(load_fit(self.store, success).probabilities, (1,))
        self.assertEqual(load_fit(self.store, failure).probabilities, (0,))
        self.assertEqual(dict(load_fit(self.store, success).diagnostics)["status"], "complete")

    def test_rank_deficiency_is_not_separation(self):
        _, fit_id = self.analyze([observation("a", {"u": 1, "v": 2}, 4, 2),
                                  observation("b", {"u": 2, "v": 4}, 4, 2)], ["u", "v"])
        artifact = load_fit(self.store, fit_id)
        diagnostics = dict(artifact.diagnostics)
        self.assertEqual((diagnostics["status"], diagnostics["design_rank"],
                          diagnostics["finite_rank"]), ("finite", 1, 1))
        self.assertEqual(artifact.probabilities, (0.5, 0.5))
        both, doubled, u_only = self.results(
            fit_id, [effect("both", u=1, v=2), effect("doubled", u=2, v=4), effect("u", u=1)])
        self.assertEqual(both["status"], "estimable")
        self.assertAlmostEqual(both["variance"], 0.2, delta=1e-8)
        self.assertAlmostEqual(doubled["variance"], 0.8, delta=1e-8)
        self.assertEqual(u_only["status"], "not_estimable")
        # rank deficiency combined with quasi separation: duplicate columns,
        # an all-success row and a zero-design interior row
        _, quasi = self.analyze([observation("a", {"u": 1, "v": 2}, 1, 1),
                                 observation("b", {"u": 0, "v": 0}, 2, 1)], ["u", "v"],
                                cohort="q", design="q", request="q")
        artifact = load_fit(self.store, quasi)
        diagnostics = dict(artifact.diagnostics)
        self.assertEqual((diagnostics["status"], diagnostics["design_rank"],
                          diagnostics["finite_rank"]), ("quasi", 1, 0))
        self.assertEqual(artifact.probabilities, (1, 0.5))
        assert_certificate(self, artifact.frozen, artifact.boundary, (1, 0))
        for item in self.results(quasi, [effect("u", u=1), effect("both", u=1, v=2),
                                         effect("zero")], request="q"):
            if item["name"] == "zero":
                self.assertEqual((item["status"], item["estimate"], item["variance"]),
                                 ("estimable", 0, 0))
            else:
                self.assertEqual(item["status"], "not_estimable")

    def test_zero_design_rows(self):
        _, fit_id = self.analyze([observation("a", {"x": 0}, 2, 2),
                                  observation("b", {"x": 0}, 3, 0)], ["x"])
        artifact = load_fit(self.store, fit_id)
        self.assertEqual(artifact.probabilities, (0.5, 0.5))
        self.assertEqual(artifact.boundary, (None, None))
        slope, zero = self.results(fit_id, [effect("x", x=1), effect("zero")])
        self.assertEqual(slope["status"], "not_estimable")
        self.assertEqual((zero["estimate"], zero["variance"]), (0, 0))


class TestColumnTransport(MemoryCase):
    def test_boundary_transport(self):
        rows = [observation("a", {"u": 1, "v": 0}, 1, 1),
                observation("b", {"u": 0, "v": 1}, 2, 1),
                observation("c", {"u": 0, "v": -1}, 2, 1)]
        _, original = self.analyze(rows, ["u", "v"])
        _, moved = self.analyze([observation("a", {"s": 1, "t": 2}, 1, 1),
                                 observation("b", {"s": 0, "t": 1}, 2, 1),
                                 observation("c", {"s": 0, "t": -1}, 2, 1)],
                                ["s", "t"], cohort="t", design="t", request="t")
        self.assertEqual(load_fit(self.store, original).probabilities,
                         load_fit(self.store, moved).probabilities)
        # c' = T^T c for T = ((1, 2), (0, 1)): u -> (1, 2), v -> (0, 1)
        original_items = self.results(original, [effect("u", u=1), effect("v", v=1),
                                                 effect("zero")])
        moved_items = self.results(moved, [effect("u", s=1, t=2), effect("v", t=1),
                                           effect("zero")], request="t")
        self.assertEqual([(i["status"], i["estimate"], i["variance"]) for i in original_items],
                         [(i["status"], i["estimate"], i["variance"]) for i in moved_items])

    def test_finite_transport(self):
        _, original = self.analyze([observation("a", {"u": 1, "v": 0}, 4, 1),
                                    observation("b", {"u": 0, "v": 1}, 4, 3),
                                    observation("c", {"u": 1, "v": 1}, 4, 2)], ["u", "v"])
        artifact = load_fit(self.store, original)
        theta, probabilities, info = decimal_mle_2d([(1, 0), (0, 1), (1, 1)], [4, 4, 4], [1, 3, 2])
        for j in range(2):
            self.assertAlmostEqual(artifact.theta[j], float(theta[j]), delta=1e-8)
        for actual, expected in zip(artifact.probabilities, probabilities):
            self.assertAlmostEqual(actual, float(expected), delta=1e-8)
        det = info[0][0] * info[1][1] - info[0][1] * info[1][0]

        def variance(c0, c1):
            c0, c1 = Decimal(str(c0)), Decimal(str(c1))
            return float((c0 * c0 * info[1][1] - 2 * c0 * c1 * info[0][1]
                          + c1 * c1 * info[0][0]) / det)

        _, moved = self.analyze([observation("a", {"s": 1, "t": 2}, 4, 1),
                                 observation("b", {"s": 0, "t": 1}, 4, 3),
                                 observation("c", {"s": 1, "t": 3}, 4, 2)],
                                ["s", "t"], cohort="t", design="t", request="t")
        for actual, expected in zip(load_fit(self.store, moved).probabilities, probabilities):
            self.assertAlmostEqual(actual, float(expected), delta=1e-8)
        # effects and T^T-transported effects: u->(1,2), v->(0,1), u+v->(1,3), (2,-1)->(2,3)
        pairs = [((1, 0), (1, 2)), ((0, 1), (0, 1)), ((1, 1), (1, 3)), ((2, -1), (2, 3))]
        for index, (original_c, moved_c) in enumerate(pairs):
            original_item = self.results(original, [effect("e", u=original_c[0], v=original_c[1])],
                                         request="o%d" % index)[0]
            moved_item = self.results(moved, [effect("e", s=moved_c[0], t=moved_c[1])],
                                      request="m%d" % index)[0]
            self.assertEqual(original_item["status"], moved_item["status"])
            self.assertAlmostEqual(original_item["estimate"], moved_item["estimate"], delta=1e-8)
            self.assertAlmostEqual(original_item["variance"], moved_item["variance"], delta=1e-8)
            self.assertAlmostEqual(original_item["estimate"],
                                   float(theta[0]) * original_c[0] + float(theta[1]) * original_c[1],
                                   delta=1e-8)
            self.assertAlmostEqual(original_item["variance"], variance(*original_c), delta=1e-8)


class TestExhaustiveOneColumn(unittest.TestCase):
    def audit_artifact(self, design, artifact):
        xs = [row[0] for row in design.x]
        internal = [i for i, flag in enumerate(artifact.boundary) if flag is None]
        # boundary flags agree with probabilities and counts
        for flag, p, n, k in zip(artifact.boundary, artifact.probabilities, design.n, design.k):
            if flag is None:
                self.assertTrue(0 < p < 1)
            else:
                self.assertEqual((p, k), (flag, n if flag else 0))
        self.assertEqual(artifact.internal_row_ids,
                         tuple(design.row_ids[i] for i in internal))
        # moment conditions characterize the maximizer
        assert_moment_conditions(self, design, artifact.probabilities)
        # exact separation certificate: the single column itself, either sign
        if any(flag is not None for flag in artifact.boundary):
            self.assertTrue(any(xs))
            certified = False
            for beta in ((1,), (-1,)):
                try:
                    assert_certificate(self, design, artifact.boundary, beta)
                    certified = True
                except AssertionError:
                    pass
            self.assertTrue(certified)
        # independent finite reference for the internal rows
        nonzero = [i for i in internal if xs[i]]
        if not nonzero:
            self.assertEqual((artifact.basis, artifact.theta, artifact.information), ((), (), ()))
            for i in internal:
                self.assertEqual(artifact.probabilities[i], 0.5)
            return
        first = nonzero[0]
        self.assertEqual(artifact.basis, ((Fraction(xs[first]),),))
        a = [Fraction(xs[i], xs[first]) for i in internal]
        theta, probabilities, information = decimal_mle_1d(
            a, [design.n[i] for i in internal], [design.k[i] for i in internal])
        self.assertAlmostEqual(artifact.theta[0], float(theta), delta=1e-8)
        self.assertAlmostEqual(artifact.information[0][0], float(information), delta=1e-8)
        for position, i in enumerate(internal):
            self.assertAlmostEqual(artifact.probabilities[i], float(probabilities[position]),
                                   delta=1e-8)
        # diagnostics reconcile with an independent log-likelihood
        diagnostics = dict(artifact.diagnostics)
        expected_ll = math.fsum(
            (k * math.log(p) if k else 0) + ((n - k) * math.log1p(-p) if n != k else 0)
            for n, k, p in zip(design.n, design.k, artifact.probabilities))
        self.assertAlmostEqual(diagnostics["log_likelihood"], expected_ll, delta=1e-8)
        self.assertEqual(diagnostics["finite_rank"], 1)

    def test_all_small_designs(self):
        row_types = [(x, n, k) for x in (-1, 0, 1) for n in (1, 2) for k in range(n + 1)]
        nonzero_types = [row for row in row_types if row[0]]
        combos = ([(row,) for row in row_types]
                  + list(product(row_types, repeat=2))
                  + list(product(nonzero_types, repeat=3)))
        self.assertEqual(len(combos), 15 + 225 + 1000)
        for combo in combos:
            with self.subTest(combo=combo):
                design = make_design([("r%d" % i, (x,), n, k)
                                      for i, (x, n, k) in enumerate(combo)], ("x",))
                self.audit_artifact(design, fit_extended(design))


if __name__ == "__main__":
    unittest.main()

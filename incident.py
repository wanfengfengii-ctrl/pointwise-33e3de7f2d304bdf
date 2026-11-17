import argparse
from binomial.archive.store import ArchiveStore
from binomial.archive.schema import encode_value
from binomial.archive.fit_records import read_result
from binomial.cohort.batches import apply_cohort
from binomial.design.columns import ColumnDefinition
from binomial.design.specification import register_design
from binomial.fitting.service import fit
from binomial.reporting.service import report
from samples import boundary_rows, boundary_effects


def observe(path=':memory:'):
    with ArchiveStore(path) as store:
        receipt = apply_cohort(store, 'incident', 'input', boundary_rows())
        register_design(store, 'incident-uv', ['u', 'v'], [
            ColumnDefinition('u', source='u'), ColumnDefinition('v', source='v')])
        fit_id = fit(store, 'incident-fit', ('incident', receipt.revision_id), 'incident-uv')
        report_id = report(store, 'incident-report', fit_id, boundary_effects())
        fitted = read_result(store, fit_id)
        effects = read_result(store, report_id)['items']
        return {'probabilities': fitted['probabilities'], 'boundary': fitted['boundary'],
                'internal_row_ids': fitted['internal_row_ids'],
                'effects': [{'name': e['name'], 'status': e['status'],
                             'estimate': e['estimate'], 'variance': e['variance']} for e in effects]}


def deviations(actual, case='all'):
    failures = []
    if case in ('all', 'probabilities'):
        if actual['probabilities'] != [1, 0.5, 0.5]:
            failures.append('probabilities must be [1, 0.5, 0.5]')
        if actual['boundary'] != [1, None, None] or actual['internal_row_ids'] != ['b', 'c']:
            failures.append('boundary and internal row identities disagree with the extended fit')
    if case in ('all', 'effects'):
        first, second, zero = actual['effects']
        if (first['status'], first['estimate'], first['variance']) != ('not_estimable', None, None):
            failures.append('first effect must be not_estimable without numbers')
        for item, variance in ((second, 1), (zero, 0)):
            if (item['status'] != 'estimable' or item['estimate'] is None
                    or item['variance'] is None or abs(item['estimate']) > 1e-8
                    or abs(item['variance'] - variance) > 1e-8):
                failures.append(item['name'] + ' has an incorrect finite estimate or variance')
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=':memory:')
    parser.add_argument('--case', choices=('all', 'probabilities', 'effects'), default='all')
    args = parser.parse_args()
    actual = observe(args.db)
    failures = deviations(actual, args.case)
    print(encode_value({'actual': actual, 'failures': failures}))
    if failures:
        print('BOUNDARY_EFFECT_MISREPORT')
        return 1
    print('boundary incident reconciled')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

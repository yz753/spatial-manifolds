"""
Check linear mixed effects model with added param: re_formula='1'.
In theory, only adding this allows different intercepts for each mouse. This code aims to check this.
"""
import argparse
from pathlib import Path
import warnings

import pandas as pd
import statsmodels
import statsmodels.formula.api as smf

from test.PATHS import repo_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path,
                        default=Path(repo_root))
    args = parser.parse_args()
    folder = args.repo.expanduser() / 'data' / 'population_state'
    keys = ['mouse', 'day', 'cluster_id']
    units = pd.read_csv(folder / 'unit_table.csv')
    scores = pd.read_csv(folder / 'pc1_by_region.csv')
    data = units.merge(scores[keys + ['excess']], on=keys,
                       validate='one_to_one')
    
    # match the selection and group coding in fig2_single_units.py
    data = data[data.in_population & data.r.notna()
                & data.identity.isin(['grid', 'putative interneuron'])] # only these two are included in the manuscript
    data = data.dropna(subset=['excess']).copy()
    data['g'] = data.identity.eq('grid').astype(int)
    data['sess'] = data.mouse.astype(str) + 'D' + data.day.astype(str)

    # test
    print('\nTest linear mixed effects model with added param: re_formula="1"')
    print(f'Data: {len(data)} cells, {data.mouse.nunique()} mice, '
          f'{data.sess.nunique()} sessions')
    print(data.identity.value_counts().to_string())
    print('\nOutcome: |r| minus mean |shifted r|.')
    print('Group coefficient: grid minus putative interneuron.')

    results = []
    mice = sorted(data.mouse.unique())

    print('\nPer-mouse intercepts:')
    for name, extra in [
        ('Current', {}),
        ('Add mouse intercept', {'re_formula': '1'}),
    ]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = smf.mixedlm(
                'excess ~ g', data,
                groups=data.mouse,
                vc_formula={'sess': '0 + C(sess)'},
                **extra,
            )
            fit = model.fit(reml=True, method='lbfgs')

        intercepts = [
            fit.fe_params['Intercept']
            + (fit.random_effects[m].iloc[0] if model.k_re else 0)
            for m in mice
        ]
        print(f"{name}: {[round(x, 6) for x in intercepts]}")

        low, high = fit.conf_int().loc['g']

        results.append(dict(
            model=name,
            difference=fit.params['g'],
            p_unadjusted=fit.pvalues['g'],
            CI_lower=low,
            CI_upper=high,
            mouse_variance=(
                fit.cov_re.iloc[0, 0] if model.k_re else float('nan')
            ),
            session_variance=fit.vcomp[0],
            mouse_var=model.k_re, # k_re
            session_var=model.k_vc, # k_vc
        ))

    print('\n' + pd.DataFrame(results).set_index('model').T.to_string())


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""게이트 2 — 비대칭 지수가 PD와 대조군을 가르는가?
메타분석 보고값(진폭 비대칭 SMD 0.84)이 재현되어야 파이프라인이 유효하다."""
import os, pathlib, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
warnings.filterwarnings('ignore')

H = pathlib.Path(__file__).parent
RAW = pathlib.Path(os.environ['WEARGAIT_RAW'])
d = pd.read_csv(H/'armswing_SPmT.csv')

pdc = pd.read_csv(RAW/'PD - Demographic+Clinical - datasetV1.csv', header=1).rename(
    columns={'Subject ID':'pid','Age (years)':'age','Height (in)':'height','Sex':'sex'})[['pid','age','height','sex']]
ctc = pd.read_csv(RAW/'CONTROLS - Demographic+Clinical - datasetV1.csv', header=1).rename(
    columns={'Subject ID':'pid','Age':'age','Height (in)':'height','Sex':'sex'})[['pid','age','height','sex']]
clin = pd.concat([pdc, ctc]); clin['sex01'] = (clin.sex=='Male').astype(int)
for c in ('age','height'): clin[c] = pd.to_numeric(clin[c], errors='coerce')
d = d.merge(clin, on='pid', how='left')
print(f"병합 후 {len(d)}명, 공변량 결측 {d[['age','sex01','height']].isna().any(axis=1).sum()}명\n")

def smd(a, b):
    a, b = np.asarray(a), np.asarray(b)
    s = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    return (a.mean()-b.mean())/s

print("="*82)
print("게이트 2 — PD vs 대조군 (SelfPace_matTURN 직선보행 구간)")
print("="*82)
METRICS = [
    ('asa_p2p_deg',  '비대칭 지수 ASA (진폭, %)',      '↑'),
    ('asa_rms',      '비대칭 지수 ASA (각속도, %)',    '↑'),
    ('min_p2p',      '덜 흔들리는 팔 진폭 (deg)',      '↓'),
    ('mean_p2p',     '양팔 평균 진폭 (deg)',           '↓'),
]
d['min_p2p'] = d[['L_p2p_deg','R_p2p_deg']].min(1)
d['mean_p2p'] = d[['L_p2p_deg','R_p2p_deg']].mean(1)
rows=[]
for col, lab, exp in METRICS:
    s = d.dropna(subset=[col])
    P, C = s[s.grp=='PD'][col], s[s.grp=='Control'][col]
    e = smd(P, C); u = stats.mannwhitneyu(P, C).pvalue
    m = s.dropna(subset=['age','sex01','height']).copy()
    X = sm.add_constant(m[['age','sex01','height']].astype(float))
    fit = sm.OLS(m[col], X.assign(PD=(m.grp=='PD').astype(int))).fit()
    rows.append(dict(metric=lab, pd_mean=P.mean(), pd_sd=P.std(), ct_mean=C.mean(), ct_sd=C.std(),
                     SMD=e, p_mwu=u, beta_adj=fit.params['PD'], p_adj=fit.pvalues['PD']))
    print(f"\n{lab}  (기대 방향 {exp})")
    print(f"  PD    {P.mean():7.2f} ± {P.std():5.2f}  (n={len(P)})")
    print(f"  대조군 {C.mean():7.2f} ± {C.std():5.2f}  (n={len(C)})")
    print(f"  SMD = {e:+.3f}   Mann-Whitney P = {u:.4f}")
    print(f"  연령·성별·신장 보정: β = {fit.params['PD']:+.3f}, P = {fit.pvalues['PD']:.4f}")
pd.DataFrame(rows).to_csv(H/'G2_validity.csv', index=False)

print("\n" + "="*82)
print("판정")
print("="*82)
a = d.dropna(subset=['asa_p2p_deg'])
e = smd(a[a.grp=='PD'].asa_p2p_deg, a[a.grp=='Control'].asa_p2p_deg)
print(f"  문헌 메타분석 진폭 비대칭 SMD = 0.84 (14편 1,130명)")
print(f"  본 자료 재현값             SMD = {e:.2f}")
print(f"  → {'통과: 방향·크기 모두 재현' if e>0.4 else ('경계: 방향은 맞으나 크기 미달' if e>0.15 else '실패: 재현 안 됨')}")

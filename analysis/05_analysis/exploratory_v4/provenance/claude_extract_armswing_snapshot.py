#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""게이트 1 — 양측 손목 IMU에서 팔 흔들림 진폭 추출.

설계 근거:
 - GeneralEvent == 'Walk' 구간만 사용(회전·기립 배제)
 - 0.3–3.0 Hz 대역통과로 PD 진전(4–7 Hz)과 드리프트를 배제.
   NLS035 우측처럼 진전이 스윙보다 큰 사례가 실재하므로 필수.
 - 각속도 '크기'는 정류 효과로 주파수가 2배로 접히므로 사용하지 않고,
   3축 PCA 제1주성분에 투영한 부호 있는 1차원 신호를 사용.
 - 좌우 각각 자체 주축을 쓰므로 센서 부착 방향에 불변.
"""
import os, pathlib, re, sys
import numpy as np, pandas as pd
from scipy import signal
import warnings; warnings.filterwarnings('ignore')

FS = 100.0
LO, HI = 0.3, 3.0            # 팔 흔들림 대역
TLO, THI = 4.0, 7.0          # 진전 대역(참고용)
MIN_WALK_S = 5.0
RAW = pathlib.Path(os.environ['WEARGAIT_RAW'])
OUT = pathlib.Path(__file__).parent

bp = signal.butter(4, [LO/(FS/2), HI/(FS/2)], btype='band', output='sos')
tp = signal.butter(4, [TLO/(FS/2), THI/(FS/2)], btype='band', output='sos')


def arm_metrics(g3):
    """g3: (n,3) 각속도. 대역통과 → PCA 제1축 투영 → 진폭 지표."""
    x = signal.sosfiltfilt(bp, g3, axis=0)
    x = x - x.mean(0)
    # PCA 제1주성분
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    w = x @ vt[0]                                   # 부호 있는 스윙 각속도 (rad/s)
    rms = float(np.sqrt(np.mean(w**2)))
    # 사이클별 peak-to-peak: 영교차로 반주기 분할
    ang = signal.detrend(np.cumsum(w) / FS)          # 적분 → 각도(rad)
    pk, _ = signal.find_peaks(ang, distance=int(0.4*FS))
    tr, _ = signal.find_peaks(-ang, distance=int(0.4*FS))
    p2p = np.nan
    if len(pk) >= 2 and len(tr) >= 2:
        m = min(len(pk), len(tr))
        p2p = float(np.median(np.abs(ang[pk[:m]] - ang[tr[:m]])) * 180/np.pi)  # deg
    var_explained = float(s[0]**2 / (s**2).sum())
    trem = float(np.sqrt(np.mean(signal.sosfiltfilt(tp, g3, axis=0)**2)))
    return dict(rms=rms, p2p_deg=p2p, pc1_var=var_explained, tremor_rms=trem, n_cycles=min(len(pk), len(tr)))


def process(path, task):
    cols = ['GeneralEvent'] + [f'{s}_Wrist_Gyr_{a}' for s in ('L', 'R') for a in ('X', 'Y', 'Z')]
    try:
        d = pd.read_csv(path, usecols=cols)
    except ValueError:
        return None
    w = d[d.GeneralEvent.astype(str) == 'Walk']
    if len(w) < MIN_WALK_S * FS:
        return None
    out = {'walk_s': len(w)/FS}
    for s in ('L', 'R'):
        g3 = w[[f'{s}_Wrist_Gyr_{a}' for a in ('X', 'Y', 'Z')]].values.astype(float)
        if np.isnan(g3).any():
            g3 = pd.DataFrame(g3).interpolate().bfill().ffill().values
        for k, v in arm_metrics(g3).items():
            out[f'{s}_{k}'] = v
    return out


def main():
    rows = []
    SRC = [('full', 'PD', r'(.+?)_SelfPace_matTURN\.csv$'),
           ('control', 'Control', r'(.+?)_SelfPace_matTURN\.csv$')]
    for folder, grp, pat in SRC:
        for f in sorted((RAW/folder).glob('*_SelfPace_matTURN.csv')):
            m = re.match(pat, f.name)
            if not m: continue
            pid = m.group(1).replace(' (control)', '').strip()
            r = process(f, 'SelfPace_matTURN')
            if r is None:
                print(f'  [제외] {f.name}: Walk 구간 부족'); continue
            rows.append(dict(pid=pid, grp=grp, **r))
    df = pd.DataFrame(rows)
    # 비대칭 지표 (부호: 음수 = 우측이 덜 흔들림 = 우측 더 영향)
    for m in ('rms', 'p2p_deg'):
        df[f'logratio_{m}'] = np.log(df[f'R_{m}'] / df[f'L_{m}'])
        df[f'asa_{m}'] = 100*np.abs(df[f'R_{m}']-df[f'L_{m}'])/np.maximum(df[f'R_{m}'], df[f'L_{m}'])
    df.to_csv(OUT/'armswing_SPmT.csv', index=False)
    print(f"\n추출 완료: {len(df)}명 (PD {(df.grp=='PD').sum()}, 대조군 {(df.grp=='Control').sum()})")
    print(f"Walk 구간 길이 중앙값 {df.walk_s.median():.1f}초, 사이클 수 중앙값 L{df.L_n_cycles.median():.0f}/R{df.R_n_cycles.median():.0f}")
    print(f"PC1 설명분산 중앙값 L{df.L_pc1_var.median():.2f} / R{df.R_pc1_var.median():.2f}")
    print(f"\n팔 흔들림 진폭(p2p, deg) 중앙값:")
    print(df.groupby('grp')[['L_p2p_deg','R_p2p_deg','L_rms','R_rms']].median().round(2).to_string())


if __name__ == '__main__':
    main()

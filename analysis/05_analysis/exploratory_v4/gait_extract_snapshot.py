#!/usr/bin/env python
"""
WearGait-PD 워크웨이 보행지표 추출 파이프라인 (프로토타입)

목적: 공개된 PKMAS 요약파일은 SelfPace/HurriedPace에만 존재한다. 나머지 task
      (SelfPace_matTURN, TUG 등)와 V2 Longitudinal 전체에는 요약지표가 없으므로
      raw 워크웨이 데이터에서 직접 산출해야 한다.

전략: SP/HP에는 PKMAS라는 정답지가 67명분 있다. 먼저 이 구간에서 자체 산출값이
      PKMAS와 일치함을 보인 뒤, 동일 파이프라인을 turning/TUG/V2로 확장한다.

품질규칙은 FDA QC 툴킷(syn70082610)에서 도출 — `02_데이터인벤토리/06_QC툴킷_분석_20260903.md` 참조.
"""
import numpy as np
import pandas as pd

FS = 100.0                    # Hz. 전 모달리티 공통 샘플링 주파수
CM_PER_SENSOR = 1.27          # 워크웨이 센서 한 칸 = 0.5인치

# --- QC 툴킷 유래 임계값 (QcAllData.m) ---
THRESH_LARGE_GAP = 10         # 프레임. 이 이상 내부 결측이면 해당 구간 배제
THRESH_SMALL_GAP = 7          # 프레임. 플래그만
THRESH_MISSING_COL_PCT = 50   # % 초과 결측 컬럼은 사용 불가

BASE_COLS = ['Time', 'GeneralEvent', 'ClinicalEvent',
             'L Foot Contact', 'R Foot Contact',
             'L Foot Pressure', 'R Foot Pressure',
             'Walkway_X', 'Walkway_Y', 'WalkwayFoot', 'WalkwayPressureLevel']


def load_walkway(path, cols=BASE_COLS):
    """워크웨이 관련 컬럼만 읽는다. 파일당 300여 컬럼이라 전체 로드는 낭비."""
    avail = set(pd.read_csv(path, nrows=0).columns)
    use = [c for c in cols if c in avail]
    d = pd.read_csv(path, usecols=use, low_memory=False)
    for c in ('L Foot Contact', 'R Foot Contact'):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0).astype(int)
    return d


def contact_episodes(binary):
    """연속 1구간 = 개별 footfall. (시작idx, 끝idx) 목록 반환."""
    x = np.asarray(binary).astype(int)
    if x.size == 0:
        return []
    d = np.diff(np.concatenate([[0], x, [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def build_footfalls(d):
    """좌우 접촉 에피소드를 시간순으로 합쳐 하나의 footfall 시퀀스로 만든다."""
    rows = []
    for side in ('L', 'R'):
        col = f'{side} Foot Contact'
        if col not in d:
            continue
        for a, b in contact_episodes(d[col]):
            rows.append(dict(side=side, i0=a, i1=b,
                             t_on=a / FS, t_off=(b + 1) / FS,
                             stance=(b + 1 - a) / FS))
    ff = pd.DataFrame(rows).sort_values('t_on').reset_index(drop=True)
    return ff


def bout_ids(ev):
    """
    GeneralEvent의 연속 동일라벨 구간마다 고유 bout 번호를 부여.

    QC 규칙 P7: SelfPace는 '4 walking bouts broken up by turns'로 구성된다
    (MC1 육안검수 기준 원문). SP/HP는 매트 밖에서 회전하므로, 서로 다른 bout에
    걸친 두 footfall을 하나의 stride로 묶으면 회전 시간이 통째로 stride_time에
    섞여 들어간다. bout 경계를 넘는 stride를 배제하기 위해 필요하다.
    """
    ev = pd.Series(ev).astype(str)
    return (ev != ev.shift()).cumsum().values


def annotate_phase(ff, d):
    """각 footfall에 GeneralEvent 국면과 bout 번호를 최빈값으로 배정."""
    if 'GeneralEvent' not in d or ff.empty:
        ff['phase'] = np.nan
        ff['bout'] = np.nan
        return ff
    ev = d['GeneralEvent'].astype(str).values
    bid = bout_ids(ev)
    ff['phase'] = [pd.Series(ev[a:b + 1]).mode().iloc[0] if b >= a else np.nan
                   for a, b in zip(ff.i0, ff.i1)]
    ff['bout'] = [pd.Series(bid[a:b + 1]).mode().iloc[0] if b >= a else np.nan
                  for a, b in zip(ff.i0, ff.i1)]
    return ff


def stride_metrics(ff, phases=None):
    """
    동측 연속 footfall 쌍에서 stride 지표를 산출.

    QC 규칙 P5: 매트를 완전히 밟지 않은 스텝은 접촉 스트림에 아예 나타나지 않는다
    (MC3 육안검수 기준 원문). 따라서 동측 두 footfall 사이에 반대측 footfall이
    정확히 1개 있을 때만 유효한 stride로 인정한다. 0개(스텝 누락) 또는 2개 이상이면
    배제한다.
    """
    out = []
    for side in ('L', 'R'):
        s = ff[ff.side == side].reset_index(drop=True)
        other = ff[ff.side != side]
        for k in range(len(s) - 1):
            a, b = s.iloc[k], s.iloc[k + 1]
            if phases is not None and a.phase not in phases:
                continue
            # 두 동측 접촉 사이의 반대측 접촉 수
            n_between = ((other.t_on > a.t_on) & (other.t_on < b.t_on)).sum()
            # 규칙 P5: 반대측 접촉이 정확히 1개일 때만 유효
            # 규칙 P7: 두 footfall이 같은 bout에 속해야 함 (bout 경계 = 회전/정지)
            same_bout = (a.bout == b.bout)
            valid = (n_between == 1) and bool(same_bout)
            out.append(dict(side=side, phase=a.phase, bout=a.bout,
                            stride_time=b.t_on - a.t_on,
                            stance_time=a.stance,
                            swing_time=(b.t_on - a.t_on) - a.stance,
                            n_between=int(n_between), same_bout=bool(same_bout),
                            valid=valid))
    return pd.DataFrame(out)


def summarize(strides, footfalls=None, phases=None):
    """
    PKMAS와 대조 가능한 요약통계.

    ★ 산출 수준이 지표마다 다르다.
      - stance_time 은 **footfall 단위** 지표다. PKMAS의 Stance Time #Samples가
        해당 국면의 footfall 총수와 일치하는 것으로 검증됨. stride를 시작하는
        footfall만 쓰면 각 bout의 마지막 footfall이 빠져 계통오차가 생긴다
        (검증 결과 평균절대오차 1.72% -> 0.07%).
      - stride_time / swing_time 은 **stride 단위** 지표다.
    """
    s = strides[strides.valid] if len(strides) else strides
    if phases is not None and len(s):
        s = s[s.phase.isin(phases)]
    res = {}

    def stats(v, name):
        if v is None or len(v) == 0:
            return
        res[f'{name}_mean'] = v.mean()
        res[f'{name}_sd'] = v.std()
        res[f'{name}_cv'] = v.std() / v.mean() * 100 if v.mean() else np.nan

    def asi(df, col, name, sidecol='side'):
        if df is None or len(df) == 0:
            return
        L, R = df.loc[df[sidecol] == 'L', col].mean(), df.loc[df[sidecol] == 'R', col].mean()
        m = df[col].mean()
        res[f'{name}_asi'] = (L - R) / m * 100 if m else np.nan

    # --- stride 단위 ---
    if len(s):
        res['n_strides'] = len(s)
        for m in ('stride_time', 'swing_time'):
            stats(s[m], m)
            asi(s, m, m)
        res['cadence'] = 2 / s.stride_time.mean() * 60 if s.stride_time.mean() else np.nan

    # --- footfall 단위 ---
    if footfalls is not None and len(footfalls):
        fw = footfalls
        if phases is not None:
            fw = fw[fw.phase.isin(phases)]
        if len(fw):
            res['n_footfalls_phase'] = len(fw)
            stats(fw['stance'], 'stance_time')
            asi(fw, 'stance', 'stance_time')
    return res


def process_file(path, phases=('Walk',)):
    d = load_walkway(path)
    ff = annotate_phase(build_footfalls(d), d)
    st = stride_metrics(ff, phases=phases)
    out = summarize(st, footfalls=ff, phases=phases)
    out['n_footfalls'] = len(ff)
    out['n_strides_raw'] = len(st)
    out['n_strides_excluded'] = int((~st.valid).sum()) if len(st) else 0
    out['duration_s'] = len(d) / FS
    return out, d, ff, st


if __name__ == '__main__':
    import sys, pathlib
    for p in sys.argv[1:]:
        res, *_ = process_file(p)
        print(f"\n=== {pathlib.Path(p).name} ===")
        for k, v in res.items():
            print(f"  {k:24s} {v:10.4f}" if isinstance(v, float) else f"  {k:24s} {v:>10}")


# ============================================================
# 공간 지표 — Walkway_X / Walkway_Y / WalkwayFoot 기반
# ============================================================

def footfall_centroids(d, ff):
    """
    각 footfall의 워크웨이 위치(무게중심)를 센서 단위로 산출한 뒤 cm로 변환.

    Walkway_X / Walkway_Y 는 프레임별로 '활성 센서 좌표'가 '|'로 구분된 문자열이며,
    WalkwayFoot 의 동일 위치 원소가 그 센서를 어느 발로 판정했는지를 L/R/other 로 담는다.
    해당 footfall의 side 라벨과 일치하는 센서만 골라 압력 가중 없이 평균한다.
    """
    if not {'Walkway_X', 'Walkway_Y', 'WalkwayFoot'} <= set(d.columns):
        ff['cx'] = np.nan; ff['cy'] = np.nan
        return ff
    X = d['Walkway_X'].astype(str).values
    Y = d['Walkway_Y'].astype(str).values
    F = d['WalkwayFoot'].astype(str).values
    cx, cy = [], []
    for a, b, side in zip(ff.i0, ff.i1, ff.side):
        xs, ys = [], []
        for i in range(int(a), int(b) + 1):
            if X[i] in ('nan', ''):
                continue
            xi, yi, fi = X[i].split('|'), Y[i].split('|'), F[i].split('|')
            if not (len(xi) == len(yi) == len(fi)):
                continue
            for x_, y_, f_ in zip(xi, yi, fi):
                if f_ == side:
                    try:
                        xs.append(float(x_)); ys.append(float(y_))
                    except ValueError:
                        pass
        cx.append(np.mean(xs) * CM_PER_SENSOR if xs else np.nan)
        cy.append(np.mean(ys) * CM_PER_SENSOR if ys else np.nan)
    ff['cx'], ff['cy'] = cx, cy
    return ff


def spatial_metrics(ff, phases=('Walk',)):
    """
    step / stride 공간 지표.
      step  : 연속한 반대측 footfall 사이 (진행축 X 거리)
      stride: 연속한 동측 footfall 사이
      width : 반대측 footfall 간 진행축 수직(Y) 거리 절대값
    bout 경계를 넘는 쌍은 배제한다(규칙 P7).
    """
    f = ff[ff.phase.isin(phases)].sort_values('t_on').reset_index(drop=True) if phases else ff
    steps, strides = [], []
    for k in range(len(f) - 1):
        a, b = f.iloc[k], f.iloc[k + 1]
        if a.bout != b.bout or a.side == b.side:
            continue
        if np.isnan(a.cx) or np.isnan(b.cx):
            continue
        steps.append(dict(idx=k, side=b.side, bout=a.bout,
                          step_length=abs(b.cx - a.cx),
                          step_time=b.t_on - a.t_on))
    for side in ('L', 'R'):
        s = f[f.side == side].reset_index(drop=True)
        for k in range(len(s) - 1):
            a, b = s.iloc[k], s.iloc[k + 1]
            if a.bout != b.bout or np.isnan(a.cx) or np.isnan(b.cx):
                continue
            dt = b.t_on - a.t_on
            strides.append(dict(side=side, bout=a.bout,
                                stride_length=abs(b.cx - a.cx),
                                stride_velocity=abs(b.cx - a.cx) / dt if dt else np.nan))
    # --- stride width: 진행선까지의 수직거리 ---
    # PKMAS/GAITRite 관례: 한 발의 접지점에서, 반대발의 직전·직후 접지점을 잇는
    # 진행선(line of progression)까지의 수직거리. 단순 |Δy|는 진행 방향이 워크웨이
    # 축과 어긋날 때 과대추정된다(검증 결과 +10.6%).
    widths = []
    for k in range(1, len(f) - 1):
        prev, cur, nxt = f.iloc[k - 1], f.iloc[k], f.iloc[k + 1]
        if not (prev.side == nxt.side != cur.side):
            continue
        if prev.bout != cur.bout or cur.bout != nxt.bout:
            continue
        if any(np.isnan(v) for v in (prev.cx, prev.cy, cur.cx, cur.cy, nxt.cx, nxt.cy)):
            continue
        # 점 cur 에서 직선 prev->nxt 까지의 수직거리
        num = abs((nxt.cx - prev.cx) * (prev.cy - cur.cy) - (prev.cx - cur.cx) * (nxt.cy - prev.cy))
        den = np.hypot(nxt.cx - prev.cx, nxt.cy - prev.cy)
        if den:
            widths.append(dict(side=cur.side, bout=cur.bout, stride_width=num / den))

    return pd.DataFrame(steps), pd.DataFrame(strides), pd.DataFrame(widths)


# ============================================================
# 압력가중 COP — 자체 정의
# ============================================================
# PKMAS의 COP/CISP/DOP 지표는 조작적 정의가 공개되어 있지 않아 방어가 불가능하다.
# (`04_산출물/08_주제_포트폴리오_및_착수계획...md` 참조)
# 따라서 raw 워크웨이에서 우리가 정의한 대로 직접 산출한다.
#   COP(t) = Σ(좌표 · 압력) / Σ(압력)
# 필요한 재료는 모두 공개되어 있다: Walkway_X / Walkway_Y / WalkwayPressureLevel / WalkwayFoot

COP_COLS = ['Walkway_X', 'Walkway_Y', 'WalkwayPressureLevel', 'WalkwayFoot']


def _parse_frame(xs, ys, ps, fs):
    """한 프레임의 '|' 구분 문자열들을 (x, y, pressure, foot) 배열로 변환."""
    if xs in ('nan', '', None):
        return None
    x, y, p, f = xs.split('|'), ys.split('|'), ps.split('|'), fs.split('|')
    n = len(x)
    if not (len(y) == len(p) == len(f) == n):
        return None
    try:
        return (np.array(x, float), np.array(y, float), np.array(p, float), np.array(f))
    except ValueError:
        return None


def cop_series(d, side=None):
    """
    프레임별 압력가중 COP 궤적(cm)을 산출.
    side가 'L'/'R'이면 해당 발로 판정된 센서만, None이면 활성 센서 전체(전신 COP).
    """
    if not set(COP_COLS) <= set(d.columns):
        return None
    X = d['Walkway_X'].astype(str).values
    Y = d['Walkway_Y'].astype(str).values
    P = d['WalkwayPressureLevel'].astype(str).values
    F = d['WalkwayFoot'].astype(str).values
    n = len(d)
    cx = np.full(n, np.nan); cy = np.full(n, np.nan); tot = np.zeros(n)
    for i in range(n):
        r = _parse_frame(X[i], Y[i], P[i], F[i])
        if r is None:
            continue
        x, y, p, f = r
        if side is not None:
            m = (f == side)
            x, y, p = x[m], y[m], p[m]
        s = p.sum()
        if s <= 0:
            continue
        cx[i] = (x * p).sum() / s * CM_PER_SENSOR
        cy[i] = (y * p).sum() / s * CM_PER_SENSOR
        tot[i] = s
    return pd.DataFrame({'cop_x': cx, 'cop_y': cy, 'total_pressure': tot})


def cop_metrics(cop, i0, i1):
    """
    한 구간 [i0, i1] 의 COP 지표.
      path_length     : 실제 이동 경로 길이 (cm)
      displacement    : 시작점-끝점 직선거리 (cm)
      path_efficiency : displacement / path_length  (1에 가까울수록 직선적)
      ml_range/ap_range : 좌우(Y)·전후(X) 이동 범위 (cm)
      cop_velocity    : path_length / 소요시간 (cm/s)
    """
    seg = cop.iloc[int(i0):int(i1) + 1].dropna(subset=['cop_x', 'cop_y'])
    if len(seg) < 3:
        return {}
    dx = np.diff(seg.cop_x.values); dy = np.diff(seg.cop_y.values)
    path = float(np.hypot(dx, dy).sum())
    disp = float(np.hypot(seg.cop_x.iloc[-1] - seg.cop_x.iloc[0],
                          seg.cop_y.iloc[-1] - seg.cop_y.iloc[0]))
    dur = len(seg) / FS
    return dict(cop_path_length=path,
                cop_displacement=disp,
                cop_path_efficiency=disp / path if path else np.nan,
                cop_ap_range=float(seg.cop_x.max() - seg.cop_x.min()),
                cop_ml_range=float(seg.cop_y.max() - seg.cop_y.min()),
                cop_velocity=path / dur if dur else np.nan)


# ============================================================
# 회전(Turn) 지표 — SelfPace_matTURN / TUG
# ============================================================

def turn_metrics(d, ff, cop=None):
    """
    `GeneralEvent == 'Turn'` 으로 주석된 각 회전 구간의 지표.
    회전 구간 검출은 전문가 주석(2인 + 신경과 전문의 검토)을 그대로 사용한다.
    """
    if 'GeneralEvent' not in d:
        return pd.DataFrame()
    ev = d['GeneralEvent'].astype(str).values
    segs = contact_episodes((ev == 'Turn').astype(int))
    rows = []
    for k, (a, b) in enumerate(segs):
        dur = (b - a + 1) / FS
        inseg = ff[(ff.i0 >= a) & (ff.i1 <= b)]
        n_ff = len(inseg)
        r = dict(turn_idx=k, t_start=a / FS, duration=dur,
                 n_footfalls=n_ff,
                 steps_per_min=n_ff / dur * 60 if dur else np.nan,
                 mean_stance=inseg.stance.mean() if n_ff else np.nan,
                 contact_pct=float(((d['L Foot Contact'].values[a:b + 1] > 0) |
                                    (d['R Foot Contact'].values[a:b + 1] > 0)).mean() * 100))
        if cop is not None:
            r.update(cop_metrics(cop, a, b))
        rows.append(r)
    return pd.DataFrame(rows)

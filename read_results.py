import csv, statistics
from pathlib import Path

rows = list(csv.DictReader(open('experiments/results/time_varying/scenario_summary.csv')))

scenarios = list(dict.fromkeys(r['scenario'] for r in rows))
print("="*100)
print("RAW RESULTS BY SCENARIO AND SEED")
print("="*100)

for sc in scenarios:
    sc_rows = [r for r in rows if r['scenario'] == sc]
    print(f"\n--- {sc} ---")
    hdr = f"{'Seed':>5} {'TgtTot':>6} {'TV':>4} {'Det%':>6} {'TVDet':>6} {'Succ%':>7} {'TVComp':>7} {'AvgDly':>8} {'TVTrk':>7} {'Loss':>5} {'Reacq':>6} {'HO':>4} {'FPS':>5}"
    print(hdr)
    for r in sc_rows:
        tv_n = int(r['n_time_varying'])
        tv_d = int(r['tv_detected'])
        tv_c = int(r['tv_completed'])
        print(f"  {r['seed']:>5} {r['total_targets']:>6} {tv_n:>4} "
              f"{float(r['detection_rate'])*100:>5.1f}% "
              f"{tv_d:>3}/{tv_n:<2} "
              f"{float(r['search_success_rate'])*100:>6.1f}% "
              f"{tv_c:>7} "
              f"{float(r['avg_det_delay_s']):>8.1f} "
              f"{float(r['tv_avg_tracking_dur']):>7.1f} "
              f"{r['total_loss_events']:>5} "
              f"{r['total_reacq_events']:>6} "
              f"{r['total_handovers']:>4} "
              f"{float(r['sim_fps']):>5.0f}")

print("\n\n" + "="*100)
print("AGGREGATED SUMMARY (mean ± std over 5 seeds)")
print("="*100)

keys = [
    ('detection_rate',      'Det Rate',          '%'),
    ('tv_detection_rate',   'TV Det Rate',        '%'),
    ('search_success_rate', 'Success Rate',       '%'),
    ('tv_completed',        'TV Completed',       ''),
    ('avg_det_delay_s',     'Avg Det Delay(s)',   ''),
    ('avg_tracking_dur_s',  'Avg Track Dur(s)',   ''),
    ('tv_avg_tracking_dur', 'TV Track Dur(s)',    ''),
    ('total_loss_events',   'Loss Events',        ''),
    ('total_reacq_events',  'Reacq Events',       ''),
    ('reacq_rate',          'Reacq Rate',         ''),
    ('total_handovers',     'Handovers',          ''),
    ('sim_fps',             'FPS',                ''),
    ('wall_time_s',         'Wall(s)',            ''),
]

print(f"\n{'Scenario':<28}", end='')
for _, label, _ in keys:
    print(f"  {label:>16}", end='')
print()
print("-"*28 + ("  " + "-"*16)*len(keys))

for sc in scenarios:
    sc_rows = [r for r in rows if r['scenario'] == sc]
    print(f"{sc:<28}", end='')
    for key, _, unit in keys:
        vals = [float(r[key]) for r in sc_rows]
        mu = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals)>1 else 0.0
        if unit == '%':
            s = f"{mu*100:.1f}±{sd*100:.1f}%"
        elif key in ('tv_completed',):
            s = f"{mu:.1f}±{sd:.1f}"
        elif key in ('sim_fps', 'total_loss_events', 'total_reacq_events', 'total_handovers'):
            s = f"{mu:.1f}±{sd:.1f}"
        else:
            s = f"{mu:.2f}±{sd:.2f}"
        print(f"  {s:>16}", end='')
    print()

print("\n" + "="*100)

# Per-target detail for TIME-VARYING ONLY scenario
print("\nPER-TARGET DETAIL: Time-Varying Only (all 5 seeds)")
print("-"*100)
pt_rows = list(csv.DictReader(open('experiments/results/time_varying/per_target_detail.csv')))
tv_pt = [r for r in pt_rows if r['scenario']=='Time-Varying Only' and r['target_type']=='time_varying']
print(f"{'Seed':>5} {'TID':>4} {'Type':>14} {'FinalSts':>12} {'DetTime':>9} {'TrkDur':>8} {'Loss':>5} {'Reacq':>6} {'HO':>4} {'TrkAcc':>8} {'StateChg':>9}")
for r in tv_pt:
    print(f"  {r['seed']:>5} {r['target_id']:>4} {r['target_type']:>14} {r['final_status']:>12} "
          f"{r['detection_time'] or 'N/A':>9} "
          f"{float(r['tracking_duration']):>8.1f} "
          f"{r['loss_events']:>5} "
          f"{r['reacq_events']:>6} "
          f"{r['handover_events']:>4} "
          f"{float(r['tracking_accuracy'])*100:>7.1f}% "
          f"{r['state_changes']:>9}")

print("\nConfiguration:")
print(f"  World: 60x60m,  UAVs: 10,  dt: 0.1s,  Search phase: 80s")
print(f"  Tracking required: 8s,  Loss timeout: 6s,  Detection radius: 4.5m")
print(f"  Speed range: 0.4-0.9 m/s,  Handover ratio: 0.55,  Max recovery: 4")

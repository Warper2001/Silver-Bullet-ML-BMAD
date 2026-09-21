VERDICT: UNDETERMINED   (rule: UNDERPOWERED if simulated power < 0.80 at the ceiling d with N_avail_high, DEFF 1.0)

window: 998 sessions with 09:31+16:00; 124 in seen months (2023/24 Sep-Nov) excluded -> 874 unseen; usable 844 after 14-session warm-up and 16 roll sessions
expected trades: N_low=350 (rate 0.414) .. N_high=430 (rate 0.509)
2025 dev shape: mean +14.11 bp, SD 122.8 bp, skew 5.54, kurtosis 47.5, n=114

anchor                          d      mean_bp | N_req normal(DEFF1) (DEFF1.5) | N_req simulated | power @N_low  @N_high | years@0.509/d
ceiling (2025 dev, in-sample)  0.1149   14.11 |       469       703 |             395 | 0.75 0.83 | 3.1
x0.75                          0.0861   10.58 |       833      1250 |             737 | 0.48 0.57 | 5.7
x0.50                          0.0574    7.05 |      1874      2812 |            1737 | 0.24 0.29 | 13.5
x0.25                          0.0287    3.53 |      7498     11247 |            7202 | 0.09 0.10 | 56.2
third-party report (+2.6 bp)   0.0212    2.60 |     13794     20690 |           13363 | 0.06 0.07 | 104.2
live ledger (+0.31 bp)         0.0041    0.50 |    370620    555929 |          >20000 | 0.03 0.03 | n/a

MDE at the expected N (normal approx., 80% power, one-sided 5%):
  N_low: N=349  d=0.133 (16.3 bp) = 1.16x the 2025 in-sample effect
  N_high: N=429  d=0.120 (14.7 bp) = 1.04x the 2025 in-sample effect

VERDICT: UNDERPOWERED   (rule: POWERED if <=2.0y at both rates; POWER_UNDETERMINED if <=2.0y at r_high only; UNDERPOWERED if >2.0y at both — at the PF-1.3 anchor, DEFF 1.0)

corrected-bar bullish trades: N=19 mean $27.18 SD $707.68 PF 1.105; gross loss/trade $259.58
PF-1.3 bar implies theta = $77.87/trade  d = 0.1100   (observed d = 0.0384)
rates: backtest 0.314/wk (16.3/yr) | shadow 1.000/wk (52.0/yr, 4 trades in 28 days)

anchor                                   theta$   SDx    d     | N(DEFF1)  yrs@low  yrs@high | N(DEFF1.5) yrs@low  yrs@high
PF-1.3 bar (PRIMARY)                       77.9  0.75  0.1467 |      287     17.6      5.5 |       431     26.4      8.3
PF-1.3 bar (PRIMARY)                       77.9  1.00  0.1100 |      511     31.3      9.8 |       766     47.0     14.7
PF-1.3 bar (PRIMARY)                       77.9  1.25  0.0880 |      798     48.9     15.3 |      1197     73.4     23.0
0.5 x bar                                  38.9  0.75  0.0734 |     1149     70.4     22.1 |      1723    105.6     33.1
0.5 x bar                                  38.9  1.00  0.0550 |     2042    125.2     39.3 |      3063    187.8     58.9
0.5 x bar                                  38.9  1.25  0.0440 |     3191    195.6     61.4 |      4787    293.5     92.1
1.5 x bar                                 116.8  0.75  0.2201 |      128      7.8      2.5 |       191     11.7      3.7
1.5 x bar                                 116.8  1.00  0.1651 |      227     13.9      4.4 |       340     20.9      6.5
1.5 x bar                                 116.8  1.25  0.1320 |      355     21.7      6.8 |       532     32.6     10.2
derivation point est. (+$27.18, PF 1.105)   27.2  0.75  0.0512 |     2357    144.5     45.3 |      3535    216.7     68.0
derivation point est. (+$27.18, PF 1.105)   27.2  1.00  0.0384 |     4190    256.9     80.6 |      6285    385.3    120.9
derivation point est. (+$27.18, PF 1.105)   27.2  1.25  0.0307 |     6547    401.4    125.9 |      9820    602.0    188.9

Inside a 2y leash (80% power, one-sided 5%):
  rate low: N = 33   MDE = $308/trade = 3.96x the PF-1.3 effect
  rate high: N = 104   MDE = $173/trade = 2.22x the PF-1.3 effect

simulated (pooled bullish+bearish shape, n=60): size at d=0, N=511: 0.043; power at the PF-1.3 effect: N=511 0.82 | N_leash_low 0.10 | N_leash_high 0.27

rate uncertainty (exact Poisson 95% on 4 shadow trades/28d): 0.27 .. 2.56 per week
  years to N_stop at the UPPER rate: PF-1.3 bar, SD x1.0: 3.8 | PF-1.3 bar, SD x0.75: 2.2 | 1.5 x bar, SD x0.75: 1.0
  inside a 2y leash at the upper rate: N = 266, MDE = $108/trade = 1.38x the PF-1.3 effect

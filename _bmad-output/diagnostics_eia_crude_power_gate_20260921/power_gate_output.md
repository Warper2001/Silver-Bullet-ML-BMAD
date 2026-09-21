VERDICT: UNDERPOWERED   (rule: UNDERPOWERED if accrual > 2.0y at the paper's own point estimate and 52 events/yr)

paper: mean 1.64 bp/trade, SD 21.0 bp, d=0.0782, t reproduced 1.90 (paper 1.88); events/yr paper 44.1

effect                          d      theta_bp | N_req(DEFF1)  yrs@paper  yrs@52 | N_req(DEFF1.5)  yrs@paper  yrs@52
ceiling (paper point est.)     0.0782    1.64  |      1010      22.9     19.4 |        1515      34.3     29.1
x0.75                          0.0587    1.23  |      1796      40.7     34.5 |        2694      61.0     51.8
x0.50                          0.0391    0.82  |      4041      91.6     77.7 |        6061     137.3    116.6
x0.33                          0.0261    0.55  |      9092     206.0    174.8 |       13638     309.0    262.3
lower 1-SE (d - 1/sqrt(N))     0.0371    0.78  |      4493     101.8     86.4 |        6739     152.7    129.6

MDE inside a 2y leash (80% power, one-sided 5%):
  paper_observed: N=88  MDE d=0.265  = 3.4x the paper's effect
  calendar_max: N=104  MDE d=0.244  = 3.1x the paper's effect

COST BREAK-EVEN (gross edge per contract vs an ASSUMED one-tick $1.00 spread floor; price grid assumed):
  price  notional  gross_edge_usd  edge/one_tick
  $ 50  $ 5,000  $0.82        0.82
  $ 60  $ 6,000  $0.99        0.99
  $ 70  $ 7,000  $1.15        1.15
  $ 80  $ 8,000  $1.31        1.31
  $ 90  $ 9,000  $1.48        1.48

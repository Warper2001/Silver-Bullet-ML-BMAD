VERDICT: UNDERPOWERED   (rule: UNDERPOWERED if net_d<=0 or years>2.0 at tradable ceiling (1-min retention) with cost card)

days with both legs: 992 over 3.96y; basket 2 MNQ vs 3 MES; paper d = 0.2571 (Sharpe 4.078/sqrt(252) = 0.2569)
SIGNAL-FREE: daily SD of fixed-direction basket open->close P&L = $257.64 (robust $237.40); notional ratio MNQ/MES median 0.91 (p05-p95 0.79-0.95)
overnight dispersion (each leg alone): SD MNQ 81.6 bp, MES 61.3 bp, spread 30.3 bp (robust 24.6 bp; roll days contaminate the plain SD)

[cost card: comm 1.04 + 1 tick/side (MNQ RT 2.04)] cost $14.70/day = 0.057 d | paper as printed (0 delay, reference only): gross d 0.2571 ($66.25/day) net d +0.2001 | no-cost yrs 0.4 | net yrs DEFF1 0.6 DEFF1.5 0.9
[cost card: comm 1.04 + 1 tick/side (MNQ RT 2.04)] cost $14.70/day = 0.057 d | tradable ceiling (1-min retention 0.306): gross d 0.0786 ($20.24/day) net d +0.0215 | no-cost yrs 4.0 | net yrs DEFF1 53.0 DEFF1.5 79.5
[cost card: comm 1.04 + 1 tick/side (MNQ RT 2.04)] cost $14.70/day = 0.057 d | 15-min retention 0.111: gross d 0.0286 ($7.36/day) net d -0.0285 | no-cost yrs 30.1 | net yrs DEFF1 inf DEFF1.5 inf
[operator MNQ RT 2.24, MES scaled as above] cost $15.10/day = 0.059 d | paper as printed (0 delay, reference only): gross d 0.2571 ($66.25/day) net d +0.1985 | no-cost yrs 0.4 | net yrs DEFF1 0.6 DEFF1.5 0.9
[operator MNQ RT 2.24, MES scaled as above] cost $15.10/day = 0.059 d | tradable ceiling (1-min retention 0.306): gross d 0.0786 ($20.24/day) net d +0.0200 | no-cost yrs 4.0 | net yrs DEFF1 61.6 DEFF1.5 92.4
[operator MNQ RT 2.24, MES scaled as above] cost $15.10/day = 0.059 d | 15-min retention 0.111: gross d 0.0286 ($7.36/day) net d -0.0300 | no-cost yrs 30.1 | net yrs DEFF1 inf DEFF1.5 inf

MDE inside 2y leash (N=504): d = 0.111 = 0.43x the paper's d
Max total round-trip cost that leaves positive net at the tradable ceiling: $20.24/day (vs cost card $14.70)
Retention of the paper's gross effect needed for POWERED inside the leash (cost card): 0.653 (paper's own 1-min stock retention = 0.306)

# MIM profit-giveback definitions

- **Decision mark:** an existing completed cash-session half-hour mark from 10:00 through 15:30 America/New_York.
- **MFE:** maximum favorable gross dollar excursion observed while the position was held, including the entry fill and completed bars through the decision mark.
- **Giveback ratio:** `(MFE - current gross P&L) / MFE`, defined only when MFE is positive.
- **Primary execution:** arm A, one contract, a one-full-minute-later open (`delay=2` in the established end-labelled engine), and $1.12 per executed side.
- **Power firewall:** candidate triggers and recipient outcomes are never paired from the same session; identity pairings are rejected before a statistic exists.
- **Power effect:** PF 1.40 translated with 90% of baseline winning dollars retained; this conservative power-only translation is distinct from the 90%-of-baseline-net success gate.
- **Success:** prospective PF at least 1.40, net profit at least 90% of the paired baseline, and a one-sided 95% confidence lower bound for paired candidate-minus-baseline daily net above zero.
- **Horizon:** 500 eligible future sessions or 30 calendar months after freeze, whichever occurs first.

All historical data through 2026-08-27 is exposed development evidence. No artifact authorizes deployment.

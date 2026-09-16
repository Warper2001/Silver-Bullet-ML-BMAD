# Mechanics

Bars are right-labelled five-minute OHLC bars from exactly 390 RTH close-stamped
minutes. Signal labels range from 09:35 to 15:50. The reference entry/exit are
the next bar's open/close, respectively, so adjacent signal intervals do not
overlap. A shifted session supplies the same next-bar slot for each actual
signal. The actual next-bar prices are never requested or retained as outcomes.

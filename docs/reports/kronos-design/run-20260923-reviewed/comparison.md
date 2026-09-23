# Kronos prospective evaluation design

strategy_test_permitted=false; trading_authorized=false

PARK_PENDING_EVIDENCE. All session counts are conditional, never measured or admitted. Dated calendar is an assertion, not admission. Warmup uses 128 completed same-contract 15-minute bars and leaves four horizon bars; gaps reset context conservatively. Calendar sessions have 26 bars, early closes 14. Deductions are disjoint with all causes retained. Calibration allocation is hypothetical, not powered calibration; zero allocation requires calibration separately sourced before this horizon.

|Months|Scenario|Weekday ceiling|Conditional evaluation N|Detectable effect (SE inflation 1 / 1.5 / 2)|
|---:|---|---:|---:|---|
|3|proxy-no-roll-capacity-ceiling|65|59|0.4220 / 0.6330 / 0.8440|
|3|assumed-quarterly-resets|65|54|0.4411 / 0.6617 / 0.8822|
|3|assumed-resets-losses-and-20-calibration|65|15|0.8370 / 1.2554 / 1.6739|
|6|proxy-no-roll-capacity-ceiling|129|119|0.2971 / 0.4457 / 0.5943|
|6|assumed-quarterly-resets|129|109|0.3105 / 0.4657 / 0.6210|
|6|assumed-resets-losses-and-20-calibration|129|46|0.4779 / 0.7169 / 0.9559|
|12|proxy-no-roll-capacity-ceiling|261|246|0.2067 / 0.3100 / 0.4133|
|12|assumed-quarterly-resets|261|226|0.2156 / 0.3234 / 0.4312|
|12|assumed-resets-losses-and-20-calibration|261|119|0.2971 / 0.4457 / 0.5943|

See scenarios.json for separate deductions, assumptions and calendar coverage; protocol.json for proposed recovery, concealment and operational requirements. No horizon is sufficient on evidence currently assessed.

# Data handling

Reject malformed timestamps, non-finite OHLC, invalid high/low ordering, and
duplicate retained timestamps. Exclude (with reason) RTH dates missing any of
the 390 expected minute stamps or containing more than one contract label.
Expected local timestamps are generated per date, preserving DST boundaries.
Records at or after the UTC cutoff are never retained. No sealed-holdout path
is opened.

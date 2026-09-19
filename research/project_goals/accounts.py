"""Provisional cash-path model, not account instructions or execution."""

from copy import deepcopy
import math

SOURCES = {
    "api": "https://help.topstep.com/en/articles/11187768-topstepx-api-access",
    "payout": "https://help.topstep.com/en/articles/8284233-topstep-payout-policy",
    "floor": "https://help.topstep.com/en/articles/8284204-what-is-the-maximum-loss-limit",
    "live": "https://help.topstep.com/en/articles/10657969-live-funded-account-parameters",
    "combine_consistency": "https://help.topstep.com/en/articles/8284208-consistency-at-topstep",
    "pricing": "https://help.topstep.com/en/articles/14289835-topstep-pricing-and-payment-questions",
}
DEFAULT = dict(
    status="PROVISIONAL_VERIFY_ACCOUNT_AGREEMENT",
    verified_date="2026-09-19",
    sources=SOURCES,
    route="combine_to_xfa_standard",
    initial_balance=50000.0,
    loss_limit=2000.0,
    profit_target=3000.0,
    combine_consistency=0.55,
    direct_capital=10000.0,
    direct_monthly_cost=0.0,
    monthly_fee=49.0,
    activation_fee=149.0,
    reset_fee=49.0,
    api_monthly=29.0,
    other_monthly=0.0,
    payout_minimum=125.0,
    payout_cap=None,
    payout_cap_standard=2000.0,
    payout_cap_consistency=3000.0,
    payout_balance_fraction=0.5,
    operator_split=0.9,
    winning_day=150.0,
    winning_days=5,
    consistency_days=3,
    consistency_fraction=0.4,
    automated_live_available=False,
)


def model_path(sessions, config=None, starting_phase="combine"):
    c = deepcopy(DEFAULT)
    c.update(config or {})
    if c.get("automated_live_available"):
        raise ValueError(
            "automated Live route unavailable without written confirmation and model revision"
        )
    if starting_phase not in (
        "combine",
        "xfa_standard",
        "xfa_consistency",
        "self_funded",
    ):
        raise ValueError("unsupported phase")
    for key, value in c.items():
        if isinstance(value, (float, int)) and (not math.isfinite(value) or value < 0):
            raise ValueError("invalid model parameter: " + key)
    for key in (
        "combine_consistency",
        "consistency_fraction",
        "payout_balance_fraction",
        "operator_split",
    ):
        if not 0 < c[key] <= 1:
            raise ValueError("invalid fraction: " + key)
    phase = starting_phase
    balance = (
        c["direct_capital"]
        if phase == "self_funded"
        else (0.0 if phase.startswith("xfa") else c["initial_balance"])
    )
    initial = balance
    peak = balance
    floor = None if phase == "self_funded" else balance - c["loss_limit"]
    cap = 0.0 if phase.startswith("xfa") else c["initial_balance"]
    paid, operating, win_days, since, first = 0.0, 0.0, 0, [], True
    months = set()
    rows = []
    complete = True
    terminal_unknown = False
    traded_days = 0
    previous = None
    for session in sessions:
        day = session["session"]
        if previous is not None and day <= previous:
            raise ValueError("sessions must be unique and chronological")
        previous = day
        month = day[:7]
        if month not in months:
            operating += (
                c["direct_monthly_cost"]
                if phase == "self_funded"
                else c["api_monthly"]
                + c["other_monthly"]
                + (c["monthly_fee"] if phase == "combine" else 0.0)
            )
            months.add(month)
        pnl = session.get("net_pnl")
        trough = session.get("intraday_min_pnl")
        if pnl is None or trough is None:
            complete = False
            terminal_unknown = True
            rows.append(
                dict(
                    session=day,
                    status="UNKNOWN_MISSING_INTRADAY_PATH",
                    terminal_balance=None,
                    last_verified_balance=balance,
                )
            )
            break
        pnl, trough = float(pnl), float(trough)
        if not math.isfinite(pnl) or not math.isfinite(trough):
            raise ValueError("nonfinite path")
        if phase == "self_funded" and balance + trough <= 0:
            rows.append(
                dict(
                    session=day,
                    status="INSOLVENT",
                    terminal_balance=None,
                    last_verified_balance=balance,
                )
            )
            terminal_unknown = True
            complete = False
            break
        if trough > min(0.0, pnl):
            raise ValueError("intraday minimum incompatible with session P&L")
        if floor is not None and balance + trough <= floor:
            rows.append(
                dict(
                    session=day,
                    phase=phase,
                    status="BREACHED",
                    floor=floor,
                    minimum_balance=balance + trough,
                    withdrawal=0.0,
                    terminal_balance=None,
                    last_verified_balance=balance,
                )
            )
            complete = False
            terminal_unknown = True
            break
        balance += pnl
        peak = max(peak, balance)
        if floor is not None:
            floor = max(floor, min(cap, peak - c["loss_limit"]))
        traded = session.get("traded") is True
        traded_days += traded
        win_days += traded and pnl >= c["winning_day"]
        since.append(pnl)
        withdrawal = 0.0
        denied = None
        transition = None
        requested = float(session.get("requested_payout", 0.0))
        if not math.isfinite(requested) or requested < 0:
            raise ValueError("negative payout request")
        if requested:
            if not phase.startswith("xfa"):
                denied = "payout rules modeled only for XFA"
            else:
                standard = win_days >= c["winning_days"] and (first or sum(since) > 0)
                consistency = (
                    traded_days >= c["consistency_days"]
                    and sum(since) > 0
                    and max(since) <= c["consistency_fraction"] * sum(since)
                )
                eligible = standard if phase == "xfa_standard" else consistency
                cap_limit = (
                    c["payout_cap"]
                    if c["payout_cap"] is not None
                    else c[
                        (
                            "payout_cap_standard"
                            if phase == "xfa_standard"
                            else "payout_cap_consistency"
                        )
                    ]
                )
                maximum = min(cap_limit, balance * c["payout_balance_fraction"])
                if (
                    not eligible
                    or requested < c["payout_minimum"]
                    or requested > maximum
                ):
                    denied = "eligibility, minimum, cap or balance fraction unmet"
                elif balance - requested <= max(floor, 0.0):
                    denied = "payout would touch post-payout floor"
                else:
                    balance -= requested
                    withdrawal = requested * c["operator_split"]
                    paid += withdrawal
                    floor = max(floor, 0.0)
                    win_days = 0
                    traded_days = 0
                    since = []
                    first = False
        if (
            phase == "combine"
            and traded_days >= 2
            and balance - initial
            >= max(c["profit_target"], max(since) / c["combine_consistency"])
        ):
            transition = "combine_pass_model_only"
            phase = (
                "xfa_standard" if c["route"].endswith("standard") else "xfa_consistency"
            )
            operating += c["activation_fee"]
            balance = 0.0
            initial = 0.0
            peak = 0.0
            floor = -c["loss_limit"]
            cap = 0.0
            win_days = 0
            traded_days = 0
            since = []
            first = True
        if session.get("live_callup"):
            transition = "AUTOMATED_LIVE_ROUTE_UNAVAILABLE"
            complete = False
        rows.append(
            dict(
                session=day,
                phase=phase,
                status="MODELED",
                balance=balance,
                floor=floor,
                cushion=None if floor is None else balance - floor,
                operator_withdrawal=withdrawal,
                request_denied=denied,
                transition=transition,
                qualifying_winning_days=win_days,
                traded_days_since_reset=traded_days,
            )
        )
        if session.get("live_callup"):
            break
    return dict(
        status="PROVISIONAL_PATH_ONLY" if complete else "INCOMPLETE_OR_STOPPED",
        starting_phase=starting_phase,
        ending_phase=phase,
        rows=rows,
        operator_withdrawals=paid,
        operating_cost=operating,
        withdrawals_less_operating=paid - operating,
        end_balance=None if terminal_unknown else balance,
        last_verified_balance=balance,
        automated_live_available=False,
        configuration=c,
        limitations=[
            "intraday minimum must include unrealized P&L",
            "capital adequacy/margin requirements unknown unless independently supplied",
            "all fees are modeled assumptions, not verified invoices",
            "taxes, grandfathering, promotions, resets and execution capacity require account-specific inputs",
            "monthly income aspiration is not extrapolated from this path",
        ],
    )


def sensitivity(sessions, config=None):
    scenarios = {}
    for phase in ("combine", "xfa_standard", "xfa_consistency", "self_funded"):
        scenarios[phase] = model_path(sessions, config, phase)
    for capital in (5000.0, 10000.0, 20000.0):
        scenarios["direct_capital_" + str(capital)] = model_path(
            sessions, dict(config or {}, direct_capital=capital), "self_funded"
        )
    # Capital stress changes direct capital only, not strategy quantities or cash P&L.
    for monthly in (0.0, 29.0, 67.0):
        c = dict(config or {}, api_monthly=monthly)
        scenarios["operating_" + str(monthly)] = model_path(sessions, c)
    return scenarios

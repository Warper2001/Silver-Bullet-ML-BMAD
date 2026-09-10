"""Execute selected frozen live methods against an in-memory broker, without importing production.

The original AST methods execute unchanged. Their database import resolves only
an in-memory stub. No credentials, live adapters, loggers or production state are
created. This is an independent controlled-event oracle for the research model.
"""
import ast
import asyncio
import builtins
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo
import numpy as np


class Sink:
    def __init__(self): self.rows = []
    def append(self, row): self.rows.append(row)
    def __getattr__(self, _): return lambda *args, **kwargs: None


def oracle():
    src = Path(__file__).with_name('mim_nb_live.py').read_text()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == 'MimNbLive')
    wanted = {'on_bar', '_enter', '_exit', '_record_trade', '_cancel_cat_stop', '_flatten', 'prev_ref_price'}
    methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in wanted]
    def restricted_import(name, *args, **kwargs):
        if name == 'time': return builtins.__import__(name, *args, **kwargs)
        if name != 'src.monitoring.trade_db': raise ImportError(name)
        return SimpleNamespace(TradeDatabase=lambda: SimpleNamespace(log_trade=lambda **kwargs: None))
    marks = {f'{h:02d}:{m}' for h in range(10, 16) for m in ('00', '30')}
    scope = {'__builtins__': dict(vars(builtins), __import__=restricted_import), 'np': np,
        'datetime': datetime, 'timezone': timezone, 'ET': ZoneInfo('America/New_York'),
        'LOOKBACK_DAYS': 14, 'CAT_STOP_PTS': 250., 'DLL_GUARD_USD': -1000.,
        'CONTRACTS': 1, 'PT_VAL': 2., 'COMBINE_START_BALANCE': 50000.,
        '_SIDE_BUY': 0, '_SIDE_SELL': 1, '_TYPE_MARKET': 2, '_TYPE_STOP': 4,
        'ENTRY_MARKS': marks, 'CHECK_MARKS': marks | {'16:00'}, 'RTH_LAST': '16:00',
        'EARLY_CLOSE_DATES': set(), 'logger': Sink(), 'bars_log': Sink(),
        'decisions_log': Sink(), 'trades_log': Sink(), 'orders_log': Sink()}
    exec(compile(ast.Module(body=methods, type_ignores=[]), '<frozen live methods>', 'exec'), scope)
    return type('FrozenLiveOracle', (), {name: scope[name] for name in wanted}), scope


async def fixture(close, position=0, anchor=100., realized=0., rejected=False, eod=False, stop_filled=False, reject_exits_only=False):
    cls, scope = oracle(); obj = cls()
    async def noop(*args, **kwargs): pass
    calls = [0]
    async def order(*args):
        calls[0] += 1
        return None if rejected or (reject_exits_only and calls[0] <= 2) else 'fixture-order'
    async def is_open(*args): return not stop_filled
    obj.px = SimpleNamespace(is_order_open=is_open, cancel_order=noop, cancel_orders=noop, close_position_at_market=noop)
    obj.day = datetime(2026, 9, 10).date(); obj.open_d = 100.
    obj.prev_close = 100.; obj.cum_pv = 0.; obj.cum_v = 0.; obj.today_moves = {}
    obj.sigma_hist = {m: [.01] * 14 for m in scope['CHECK_MARKS']}
    obj.position = position; obj.entry_px = anchor; obj.entry_t = '10:00'
    obj.day_pnl = realized; obj.day_deactivated = realized <= -1000
    obj._realized_pnl = realized; obj._mll_eod_hwm = 50000.; obj._buffer_source = 'fixture'
    obj.cat_stop_id = 'fixture-stop' if position else None
    obj.account_id = 'fixture'; obj._ts_sim_mirror = None
    obj._save_state = lambda: None; obj._remaining_mll_buffer = lambda: 10000.
    obj._close_out_session = lambda c: None; obj._order = order
    obj._bar_et = lambda b: datetime.fromisoformat(b['TimeStamp']).astimezone(scope['ET'])
    hour = 20 if eod else 15
    await obj.on_bar({'TimeStamp': f'2026-09-10T{hour}:00:00+00:00', 'Open': 100., 'High': max(100., close), 'Low': min(100., close), 'Close': close, 'TotalVolume': 10})
    return {'close': close, 'initial_position': position, 'initial_anchor': anchor,
        'initial_realized': realized, 'rejected': rejected, 'eod': eod, 'stop_filled': stop_filled, 'reject_exits_only': reject_exits_only,
        'position': obj.position, 'realized': obj.day_pnl, 'deactivated': obj.day_deactivated,
        'action': scope['decisions_log'].rows[-1]['action'], 'trades': scope['trades_log'].rows}


async def run():
    cases = [dict(close=102.), dict(close=98.), dict(close=101.),
        dict(close=100., position=1), dict(close=98., position=1),
        dict(close=98., position=1, anchor=598.),
        dict(close=98., position=1, anchor=597.995),
        dict(close=102., rejected=True), dict(close=98., position=1, reject_exits_only=True), dict(close=102., position=1, eod=True),
        dict(close=102., position=1, anchor=102., stop_filled=True)]
    return [await fixture(**case) for case in cases]


if __name__ == '__main__': print(json.dumps(asyncio.run(run()), indent=2))

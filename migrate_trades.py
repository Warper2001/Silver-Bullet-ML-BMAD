
import pandas as pd
from src.monitoring.trade_db import TradeDatabase
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("migration")

db = TradeDatabase()

LOG_CONFIGS = {
    'logs/tier2_trade_log.csv': ('trader-yank', {
        'timestamp_entry': 'timestamp', 
        'direction': 'direction', 
        'entry_price': 'entry_price', 
        'exit_price': 'exit_price', 
        'pnl_usd': 'pnl', 
        'exit_reason': 'exit_reason',
        'contracts': 'contracts'
    }),
    'logs/s26_soft_fvg_trade_log.csv': ('trader-s26', {
        'entry_time': 'timestamp', 
        'direction': 'direction', 
        'entry_price': 'entry_price', 
        'exit_price': 'exit_price', 
        'pnl': 'pnl', 
        'reason': 'exit_reason',
        'ml_proba': 'ml_proba'
    }),
    'logs/s27_squeeze_trade_log.csv': ('trader-s27', {
        'entry_time': 'timestamp', 
        'direction': 'direction', 
        'entry_price': 'entry_price', 
        'exit_price': 'exit_price', 
        'pnl': 'pnl', 
        'reason': 'exit_reason',
        'ml_proba': 'ml_proba'
    }),
    'data/mim_nb/trades.csv': ('trader-mim-nb', {
        'entry_t': 'timestamp', 
        'dir': 'direction', 
        'entry_px': 'entry_price', 
        'exit_px': 'exit_price', 
        'reason': 'exit_reason', 
        'pnl_usd': 'pnl'
    }),
    'logs/carry_positions.csv': ('trader-btc-carry', {
        'event_time': 'timestamp', 
        'pnl_usd': 'pnl'
    }),
}

def migrate():
    for path_str, (trader_id, col_map) in LOG_CONFIGS.items():
        path = Path(path_str)
        if not path.exists():
            logger.warning(f"File {path} not found, skipping.")
            continue
        
        logger.info(f"Migrating {path} for {trader_id}...")
        try:
            df = pd.read_csv(path, on_bad_lines='skip')
            
            if 'event_type' in df.columns:
                df = df[df['event_type'] == 'EXIT']

            count = 0
            for _, row in df.iterrows():
                data = {db_col: row.get(leg_col) for leg_col, db_col in col_map.items()}
                
                if pd.isna(data.get('timestamp')) or pd.isna(data.get('pnl')):
                    continue
                
                db.log_trade(
                    trader_id=trader_id,
                    timestamp=str(data['timestamp']),
                    pnl=float(data['pnl']),
                    direction=data.get('direction'),
                    entry_price=data.get('entry_price'),
                    exit_price=data.get('exit_price'),
                    exit_reason=data.get('exit_reason'),
                    ml_proba=data.get('ml_proba'),
                    metadata={"legacy_source": path_str}
                )
                count += 1
            logger.info(f"Successfully migrated {count} trades from {path}")
        except Exception as e:
            logger.error(f"Failed to migrate {path}: {e}")

if __name__ == "__main__":
    migrate()

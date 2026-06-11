import re
import os

def update_file(path):
    if not os.path.exists(path): return
    with open(path, 'r') as f:
        content = f.read()

    # 1. Fix naive datetime.now() -> _now()
    # First ensure we import _now
    if "from src.execution.models import" in content:
        content = content.replace("from src.execution.models import", "from src.execution.models import _now, NY_TZ,")
    
    content = content.replace("datetime.now()", "_now()")
    content = content.replace("datetime.utcnow()", "_now()")

    # 2. Fix exit reason strings
    reasons = {
        "time_stop": "Time stop (10-min max)",
        "take_profit": "Take profit",
        "stop_loss": "Stop loss",
        "hybrid_partial": "Hybrid partial (1.5R)",
        "hybrid_trail": "Hybrid trail (2R)"
    }
    for old, new in reasons.items():
        content = content.replace(f'"{old}"', f'"{new}"')
        content = content.replace(f"'{old}'", f'"{new}"')

    # 3. Fix calculate_rr_achieved signature in tests
    # find: calculate_rr_achieved(entry, price, sl)
    # replace with: calculate_rr_achieved(entry, price, sl, "long") (default to long for tests)
    content = re.sub(r'calculate_rr_achieved\(([^,]+),\s*([^,]+),\s*([^,)]+)\)', 
                     r'calculate_rr_achieved(\1, \2, \3, "long")', content)

    with open(path, 'w') as f:
        f.write(content)

test_files = [
    "tests/unit/test_exit_logic_models.py",
    "tests/unit/test_time_based_exit.py",
    "tests/unit/test_risk_reward_exit.py",
    "tests/unit/test_hybrid_exit.py",
    "tests/integration/test_exit_integration.py"
]

for f in test_files:
    update_file(f)

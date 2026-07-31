
import os

log_file = "logs/s26_combine_bot.log"
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        # Grep for trade execution lines
        lines = [line for line in f if "TRADE" in line or "FILLED" in line or "ORDER" in line]
        print("".join(lines[-15:]))
else:
    print(f"Log file {log_file} not found.")

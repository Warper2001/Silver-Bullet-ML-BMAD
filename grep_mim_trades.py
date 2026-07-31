
import os
log_file = "logs/mim_nb_live.log"
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        found = [line for line in f if "TRADE" in line or "ORDER" in line or "FILLED" in line]
        print("".join(found[-10:]))
else:
    print("Log file not found.")


import os
log_file = "logs/mim_nb_live.log" # Assuming this follows the pattern
if not os.path.exists(log_file):
    # Try finding the actual log file if it's named differently
    import glob
    possible_logs = glob.glob("logs/mim*.log")
    if possible_logs:
        log_file = possible_logs[0]
    else:
        print("Log file not found.")
        exit()

with open(log_file, "r") as f:
    lines = f.readlines()[-20:]
    print("".join(lines))

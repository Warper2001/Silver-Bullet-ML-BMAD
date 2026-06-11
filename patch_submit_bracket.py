import re

def patch_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    # We want to replace the parsing loop in submit_bracket_order
    # It looks like this:
    old_code = """
            for order in orders:
                oid = order.get("OrderID")
                order_type = order.get("OrderType", "")
                msg = order.get("Message", "")
                tif = (order.get("TimeInForce") or {}).get("Duration", "")
                # Primary: OrderType + TimeInForce distinguish entry (Limit/DAY) from TP (Limit/GTC)
                if order_type == "StopMarket" or "Stop Market" in msg:
                    sl_id = oid
                elif order_type == "Limit" and tif == "GTC":
                    tp_id = oid
                elif order_type == "Limit":
                    entry_id = oid  # DAY or unknown duration
                else:
                    # Fallback: positional — entry first, then TP
                    if entry_id is None:
                        entry_id = oid
                    else:
                        tp_id = oid
"""
    
    new_code = """
            for order in orders:
                oid = order.get("OrderID")
                msg = order.get("Message", "")
                
                # TradeStation API often returns only Message and OrderID for OCOs
                # Message format: "Sent order: Buy 1 MNQM26 @ 1000.00 Limit"
                if "Stop Market" in msg:
                    sl_id = oid
                elif exit_action.capitalize() in msg and "Limit" in msg:
                    tp_id = oid
                elif entry_action.capitalize() in msg and "Limit" in msg:
                    entry_id = oid
                else:
                    # Fallback if messages don't match expected pattern
                    if entry_id is None:
                        entry_id = oid
                    elif tp_id is None:
                        tp_id = oid
                    else:
                        sl_id = oid
"""

    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Patched {filepath}")
    else:
        print(f"Could not find old code in {filepath}")

patch_file("src/research/tier2_streaming_working.py")
patch_file("src/research/yank_streaming_working.py")

from datetime import datetime, timezone, timedelta

def now_utc_plus_one():
    return datetime.now(timezone(timedelta(hours=1)))
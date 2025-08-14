# core/state_registry.py
import time

_state = {}  # client_key -> {"mode": "idle|training|rest", "ts": float}

def client_key_from_conn(conn):
    # 在不同工程里可能有不同的标识，这里做降级：
    for attr in ("client_id", "uid", "id", "device_id"):
        v = getattr(conn, attr, None)
        if v: return str(v)
    return str(id(conn))  # 退而求其次：对象地址

def set_state(conn, mode: str):
    if mode not in ("idle", "training", "rest"): return
    key = client_key_from_conn(conn)
    _state[key] = {"mode": mode, "ts": time.time()}

def get_state(conn, default="idle"):
    key = client_key_from_conn(conn)
    return _state.get(key, {}).get("mode", default)

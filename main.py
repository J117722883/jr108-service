import os
import re
import requests
from datetime import datetime, timezone, date
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from dateutil.parser import isoparse

app = FastAPI(title="JR-108 Tool API", version="1.0.0")

# ====== REQUIRED ENV VARS (you will set these in Render) ======
JR108_ACTION_BEARER_TOKEN = os.getenv("JR108_ACTION_BEARER_TOKEN", "")
ALPACA_KEY_ID = os.getenv("ALPACA_KEY_ID", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

ALPACA_DATA_BASE = "https://data.alpaca.markets"
ALPACA_HEADERS = {
    "APCA-API-KEY-ID": ALPACA_KEY_ID,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "accept": "application/json",
}

# ====== Input schema from ChatGPT action ======
class JR108Request(BaseModel):
    run_id: str
    ticker: str
    as_of_date: str
    catalysts: List[Dict[str, Any]] = []


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_auth(authorization: Optional[str]) -> None:
    """
    ChatGPT Actions will send the token in Authorization header.
    """
    if not JR108_ACTION_BEARER_TOKEN:
        raise HTTPException(status_code=500, detail="Server missing JR108_ACTION_BEARER_TOKEN")
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    parts = authorization.split()
    token = parts[-1] if len(parts) >= 2 else authorization
    if token != JR108_ACTION_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")


def require_alpaca_keys() -> None:
    if not ALPACA_KEY_ID or not ALPACA_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Server missing Alpaca keys (ALPACA_KEY_ID / ALPACA_SECRET_KEY)")


def citation(url: str, title: str, snippet_obj: Any) -> Dict[str, Any]:
    # Keep snippet short; validator-friendly.
    snippet = str(snippet_obj)
    if len(snippet) > 700:
        snippet = snippet[:700] + "…"
    return {
        "url": url,
        "title": title,
        "published_date": None,
        "retrieved_at": utc_now_iso(),
        "supporting_snippet": snippet,
    }


def parse_catalyst_window_to_date(window: Optional[str], fallback_iso: str) -> Tuple[Optional[date], List[str]]:
    """
    Tries to turn a catalyst window into a concrete date.
    Supports:
      - YYYY-MM-DD
      - 1H 2026, 2H 2026
      - Q1 2026, Q2 2026, etc.
    If it can't parse, returns None + an assumption message.
    """
    assumptions = []
    if not window:
        assumptions.append("No catalyst window provided; will choose expiry nearest after as_of_date + 30 days.")
        return None, assumptions

    w = window.strip()

    # direct date
    try:
        dt = isoparse(w).date()
        return dt, assumptions
    except Exception:
        pass

    # 1H/2H
    m = re.match(r"^(1H|2H)\s+(\d{4})$", w, re.IGNORECASE)
    if m:
        half = m.group(1).upper()
        year = int(m.group(2))
        if half == "1H":
            return date(year, 6, 30), assumptions
        else:
            return date(year, 12, 31), assumptions

    # Quarter Q1-Q4
    m = re.match(r"^Q([1-4])\s+(\d{4})$", w, re.IGNORECASE)
    if m:
        q = int(m.group(1))
        year = int(m.group(2))
        end_month = {1: 3, 2: 6, 3: 9, 4: 12}[q]
        end_day = {3: 31, 6: 30, 9: 30, 12: 31}[end_month]
        return date(year, end_month, end_day), assumptions

    assumptions.append(f"Could not parse catalyst window '{window}'; will choose expiry nearest after as_of_date + 30 days.")
    return None, assumptions


def get_stock_snapshot(ticker: str) -> Tuple[Optional[float], Optional[str], Dict[str, Any]]:
    """
    Pulls stock snapshot. We'll use latestTrade price as spot if present.
    Alpaca endpoint: /v2/stocks/snapshots
    """
    url = f"{ALPACA_DATA_BASE}/v2/stocks/snapshots"
    params = {"symbols": ticker}
    r = requests.get(url, headers=ALPACA_HEADERS, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    spot = None
    snap_time = None
    try:
        snap = data["snapshots"][ticker]
        spot = snap.get("latestTrade", {}).get("p")
        snap_time = snap.get("latestTrade", {}).get("t")
    except Exception:
        pass

    return spot, snap_time, {"url": f"{url}?symbols={ticker}", "raw": data}


def get_option_chain_snapshot(underlying: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Alpaca option chain snapshot endpoint:
      /v1beta1/options/snapshots/{underlying_symbol}
    """
    url = f"{ALPACA_DATA_BASE}/v1beta1/options/snapshots/{underlying}"
    r = requests.get(url, headers=ALPACA_HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data, {"url": url, "raw": data}


def extract_contracts(chain_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Alpaca option chain snapshot typically returns a dict keyed by contract symbols.
    We'll normalize into a list with fields we need.
    """
    contracts = []
    if not isinstance(chain_json, dict):
        return contracts

    # Common shape: {"snapshots": { "O:TSLA....": {...}, ... }}
    snapshots = chain_json.get("snapshots")
    if isinstance(snapshots, dict):
        items = snapshots.items()
    else:
        # Sometimes it might already be a dict of contracts
        items = chain_json.items()

    for sym, obj in items:
        if not isinstance(obj, dict):
            continue
        c = obj.get("contract", {}) if isinstance(obj.get("contract"), dict) else {}
        # Try to find expiry/strike/type in contract
        expiry = c.get("expiration_date") or c.get("expiration") or c.get("expiry")
        strike = c.get("strike_price") or c.get("strike")
        opt_type = c.get("type") or c.get("option_type")

        # Fallback: try to parse OCC-style symbol if present (best-effort)
        contracts.append({
            "symbol": sym,
            "expiry": expiry,
            "strike": strike,
            "type": opt_type,
            "raw": obj
        })

    return contracts


def parse_date_safe(x: Any) -> Optional[date]:
    if not x:
        return None
    try:
        return isoparse(str(x)).date()
    except Exception:
        return None


def pick_expiries(contracts: List[Dict[str, Any]], catalyst_date: Optional[date], as_of: date) -> Tuple[Optional[date], Optional[date]]:
    expiries = sorted({parse_date_safe(c.get("expiry")) for c in contracts if parse_date_safe(c.get("expiry"))}, key=lambda d: d)
    if not expiries:
        return None, None

    target = catalyst_date
    if not target:
        # fallback: target is as_of + 30 days
        target = as_of.toordinal() + 30
        target = date.fromordinal(target)

    before = None
    after = None
    for d in expiries:
        if d < target:
            before = d
        if d >= target and after is None:
            after = d
    return before, after


def find_atm_call_put(contracts: List[Dict[str, Any]], expiry: date, spot: float) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Find call and put for given expiry with strike closest to spot.
    """
    same_exp = []
    for c in contracts:
        ed = parse_date_safe(c.get("expiry"))
        if ed == expiry:
            try:
                strike = float(c.get("strike"))
            except Exception:
                continue
            opt_type = (c.get("type") or "").lower()
            same_exp.append((c, strike, opt_type))

    if not same_exp:
        return None, None

    # Find strike closest to spot
    same_exp.sort(key=lambda t: abs(t[1] - spot))
    atm_strike = same_exp[0][1]

    # Pick nearest call and put at that strike
    call = None
    put = None
    for c, strike, opt_type in same_exp:
        if strike != atm_strike:
            continue
        if "call" in opt_type or opt_type == "c":
            call = c
        if "put" in opt_type or opt_type == "p":
            put = c
        if call and put:
            break

    return call, put


def get_option_snapshots(symbols: List[str]) -> Dict[str, Any]:
    """
    Option snapshots endpoint (for specific contracts):
      /v1beta1/options/snapshots?symbols=...
    """
    url = f"{ALPACA_DATA_BASE}/v1beta1/options/snapshots"
    params = {"symbols": ",".join(symbols)}
    r = requests.get(url, headers=ALPACA_HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return {"url": f"{url}?symbols={params['symbols']}", "raw": r.json()}


def mid_from_quote(snap: Dict[str, Any]) -> Optional[float]:
    """
    Given a contract snapshot object, try to compute mid from latestQuote bid/ask.
    """
    q = snap.get("latestQuote") or snap.get("latest_quote") or {}
    bid = q.get("bp") or q.get("bid_price") or q.get("bid")
    ask = q.get("ap") or q.get("ask_price") or q.get("ask")
    try:
        bid = float(bid)
        ask = float(ask)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
    except Exception:
        return None
    return None


@app.get("/health")
def health():
    return {"ok": True, "time_utc": utc_now_iso()}


@app.post("/jr108/run")
def jr108_run(req: JR108Request, authorization: Optional[str] = Header(default=None)):
    # 1) Basic auth (token from GPT Action)
    require_auth(authorization)
    require_alpaca_keys()

    ticker = req.ticker.upper()

    # Global arrays required by your schema
    citations_global: List[Dict[str, Any]] = []
    sourced_facts: List[Dict[str, Any]] = []
    assumptions: List[str] = []
    unknowns: List[str] = []
    conflicts: List[Dict[str, Any]] = []

    # 2) Choose catalyst (first)
    catalyst_ref = "catalyst_1"
    catalyst_window = None
    if req.catalysts:
        catalyst_ref = str(req.catalysts[0].get("id") or req.catalysts[0].get("event") or "catalyst_1")
        catalyst_window = str(req.catalysts[0].get("window") or req.catalysts[0].get("timing") or "")
    else:
        unknowns.append("No catalysts provided to JR-108 input; expiry selection will fall back to as_of_date + 30 days.")

    # parse as_of_date
    try:
        as_of_date = isoparse(req.as_of_date).date()
    except Exception:
        as_of_date = datetime.now(timezone.utc).date()
        assumptions.append("as_of_date could not be parsed; used current UTC date as fallback.")

    catalyst_date, parse_assumptions = parse_catalyst_window_to_date(catalyst_window, req.as_of_date)
    assumptions.extend(parse_assumptions)

    # 3) Underlying snapshot
    spot = None
    snap_time = None
    try:
        spot, snap_time, snap_meta = get_stock_snapshot(ticker)
        c = citation(snap_meta["url"], "Alpaca Market Data API — Stock Snapshots", {"spot": spot, "snapshot_time_utc": snap_time})
        citations_global.append(c)
        sourced_facts.append({"value": {"spot_price": spot, "snapshot_time_utc": snap_time}, "confidence": "high" if spot else "low", "citations": [c]})
    except Exception as e:
        unknowns.append(f"Failed to fetch stock snapshot from Alpaca: {str(e)}")

    # If we don't have a spot price, we can't compute ATM
    if not spot:
        payload = {
            "payload": {
                "underlying": {"spot_price": None, "snapshot_time_utc": snap_time, "citations": []},
                "event_context": {"catalyst_ref": catalyst_ref, "catalyst_window": catalyst_window or "", "chosen_expiry": "", "rationale": "", "citations": []},
                "atm_straddle": {"strike": None, "call_mid": None, "put_mid": None, "straddle_mid": None, "implied_move_pct": None, "citations": []},
                "volatility": {"iv_if_available": None, "skew_notes": None, "term_structure_notes": None, "citations": []},
                "liquidity": {"call_oi": None, "put_oi": None, "call_volume": None, "put_volume": None, "citations": []},
                "limitations": {"notes": "Could not compute options implied move because spot price was unavailable."},
            },
            "sourced_facts": sourced_facts,
            "assumptions": assumptions,
            "unknowns": unknowns,
            "conflicts": conflicts,
            "citations": citations_global,
        }
        return payload

    # 4) Options chain snapshot
    contracts: List[Dict[str, Any]] = []
    chain_meta = None
    try:
        chain_json, chain_meta = get_option_chain_snapshot(ticker)
        contracts = extract_contracts(chain_json)
        c = citation(chain_meta["url"], "Alpaca Market Data API — Option Chain Snapshots", {"contracts_seen": len(contracts)})
        citations_global.append(c)
    except Exception as e:
        unknowns.append(f"Failed to fetch options chain from Alpaca: {str(e)}")

    if not contracts:
        unknowns.append("No options contracts found in Alpaca chain snapshot for this ticker (or response shape not recognized).")

    # 5) Choose expiries
    before_exp, after_exp = (None, None)
    if contracts:
        before_exp, after_exp = pick_expiries(contracts, catalyst_date, as_of_date)

    chosen_exp = after_exp or before_exp
    if not chosen_exp:
        unknowns.append("Could not identify any expiries from options chain; cannot compute implied move.")
        chosen_exp_str = ""
    else:
        chosen_exp_str = chosen_exp.isoformat()

    rationale = ""
    if chosen_exp and after_exp:
        rationale = "Chosen expiry is the nearest expiry on/after the catalyst target date (primary focus)."
    elif chosen_exp and before_exp:
        rationale = "No expiry found after the catalyst target date; used nearest expiry before catalyst date."
        assumptions.append("Used nearest expiry before catalyst because no later expiry was available in chain snapshot.")
    else:
        rationale = "No suitable expiry could be selected."

    # 6) Find ATM call/put
    call_c = put_c = None
    atm_strike = None
    if chosen_exp and contracts:
        call_c, put_c = find_atm_call_put(contracts, chosen_exp, spot)
        if call_c and call_c.get("strike"):
            try:
                atm_strike = float(call_c["strike"])
            except Exception:
                pass
        if (atm_strike is None) and put_c and put_c.get("strike"):
            try:
                atm_strike = float(put_c["strike"])
            except Exception:
                pass

    if not call_c or not put_c:
        unknowns.append("Could not identify both an ATM call and ATM put for the chosen expiry.")
        assumptions.append("Option contract parsing may be incomplete if chain response does not include strike/type fields in expected format.")

    # 7) Fetch snapshots for those contracts (to get quotes/greeks)
    call_mid = put_mid = None
    iv_atm = None
    skew_notes = None
    term_notes = None
    opt_citations = []

    if call_c and put_c:
        try:
            symbols = [call_c["symbol"], put_c["symbol"]]
            snaps_meta = get_option_snapshots(symbols)
            raw = snaps_meta["raw"]
            # Common shape: {"snapshots": {"CONTRACT": {...}}}
            snaps = raw.get("snapshots", raw)

            call_snap = snaps.get(call_c["symbol"], {})
            put_snap = snaps.get(put_c["symbol"], {})

            call_mid = mid_from_quote(call_snap)
            put_mid = mid_from_quote(put_snap)

            # IV if available in greeks
            call_g = call_snap.get("greeks") or {}
            put_g = put_snap.get("greeks") or {}
            call_iv = call_g.get("iv")
            put_iv = put_g.get("iv")
            try:
                if call_iv is not None:
                    call_iv = float(call_iv)
                if put_iv is not None:
                    put_iv = float(put_iv)
            except Exception:
                call_iv = None
                put_iv = None

            if call_iv is not None and put_iv is not None:
                iv_atm = (call_iv + put_iv) / 2.0
                if put_iv > call_iv:
                    skew_notes = "ATM put IV > ATM call IV (put skew)."
                elif call_iv > put_iv:
                    skew_notes = "ATM call IV > ATM put IV (call skew)."
                else:
                    skew_notes = "ATM call IV ≈ ATM put IV."

            c = citation(snaps_meta["url"], "Alpaca Market Data API — Option Snapshots", {
                "call_mid": call_mid, "put_mid": put_mid, "call_iv": call_iv, "put_iv": put_iv
            })
            citations_global.append(c)
            opt_citations.append(c)

        except Exception as e:
            unknowns.append(f"Failed to fetch option snapshots/quotes/greeks: {str(e)}")

    # 8) Compute straddle + implied move
    straddle_mid = None
    implied_move_pct = None
    if call_mid is not None and put_mid is not None:
        straddle_mid = call_mid + put_mid
        implied_move_pct = straddle_mid / spot if spot else None
    else:
        unknowns.append("Could not compute straddle mid because call_mid or put_mid was missing (illiquid or no quote).")

    # 9) Time-to-expiry (days) — include as a sourced fact? (derived)
    dte = None
    if chosen_exp:
        dte = (chosen_exp - as_of_date).days
        # derived value: still include with same citations as expiry selection
        sourced_facts.append({
            "value": {"chosen_expiry": chosen_exp_str, "days_to_expiry": dte},
            "confidence": "medium",
            "citations": opt_citations[:] if opt_citations else []
        })

    # 10) Liquidity (OI/volume) — not always available from snapshots; we’ll set null and note limitations
    call_oi = put_oi = None
    call_vol = put_vol = None
    unknowns.append("Open interest and volume retrieval not implemented in this minimal service version; fields set to null.")
    # (You can add OI/volume later without changing the GPT Action.)

    # Build payload with required schema
    underlying_citations = [c for c in citations_global if c.get("title") == "Alpaca Market Data API — Stock Snapshots"]
    chain_citations = [c for c in citations_global if c.get("title") == "Alpaca Market Data API — Option Chain Snapshots"]

    payload_obj = {
        "payload": {
            "underlying": {
                "spot_price": spot,
                "snapshot_time_utc": snap_time,
                "citations": underlying_citations,
            },
            "event_context": {
                "catalyst_ref": catalyst_ref,
                "catalyst_window": catalyst_window or "",
                "chosen_expiry": chosen_exp_str,
                "rationale": rationale,
                "citations": chain_citations,
            },
            "atm_straddle": {
                "strike": atm_strike,
                "call_mid": call_mid,
                "put_mid": put_mid,
                "straddle_mid": straddle_mid,
                "implied_move_pct": implied_move_pct,
                "citations": opt_citations,
            },
            "volatility": {
                "iv_if_available": iv_atm,
                "skew_notes": skew_notes,
                "term_structure_notes": term_notes,
                "citations": opt_citations,
            },
            "liquidity": {
                "call_oi": call_oi,
                "put_oi": put_oi,
                "call_volume": call_vol,
                "put_volume": put_vol,
                "citations": [],
            },
            "limitations": {
                "notes": "If options are missing/illiquid, mids and IV may be null. OI/volume not implemented in this minimal version."
            }
        },
        "sourced_facts": sourced_facts,
        "assumptions": assumptions,
        "unknowns": unknowns,
        "conflicts": conflicts,
        "citations": citations_global,
    }

    return payload_obj

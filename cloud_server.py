"""
Cloud-side service for MayaCardio Pipeline.

Changes vs. original:
- Heart-rate logic replaced with shared section3.fetch_heart_rate_from_db.
- /ingest accepts losses and offline results in addition to raw samples;
  loss inserts are idempotent (UNIQUE constraint).
- /inventory returns the highest confirmed sample index per sensor per session
  so the device can reconcile after outages.
- cloud DB uses synchronous=FULL (was NORMAL).
- Window processing is deferred until a full window is available (not on
  every partial batch), avoiding publishing incomplete windows.
"""

import gzip
import json
import math
import os
import sqlite3
import sys
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

app = FastAPI(title="MayaCardio Cloud Ingestion Service")

CLOUD_DB = os.environ.get("CLOUD_DB", "cloud_storage.db")


def get_cloud_db():
    conn = sqlite3.connect(CLOUD_DB, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=FULL;")   # was NORMAL
    return conn


def init_cloud_db():
    conn = get_cloud_db()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS cloud_samples (
            session_id   TEXT,
            sensor       INTEGER,
            sample_index INTEGER,
            ch1  INTEGER,
            ch2  INTEGER,
            ch3  INTEGER,
            PRIMARY KEY (session_id, sensor, sample_index)
        );
    """)

    # Idempotent loss inserts via UNIQUE constraint
    c.execute("""
        CREATE TABLE IF NOT EXISTS cloud_loss_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            sensor     INTEGER,
            from_index INTEGER,
            to_index   INTEGER,
            reason     TEXT,
            UNIQUE (session_id, sensor, from_index, to_index, reason)
        );
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS cloud_results (
            session_id  TEXT,
            window_id   INTEGER,
            mode        TEXT DEFAULT 'online',
            missing_1   INTEGER,
            missing_2   INTEGER,
            complete    BOOLEAN,
            s1_ch1_mean REAL, s1_ch1_min REAL, s1_ch1_max REAL,
            s1_ch2_mean REAL, s1_ch2_min REAL, s1_ch2_max REAL,
            s1_ch3_mean REAL, s1_ch3_min REAL, s1_ch3_max REAL,
            s2_ch1_mean REAL, s2_ch1_min REAL, s2_ch1_max REAL,
            s2_ch2_mean REAL, s2_ch2_min REAL, s2_ch2_max REAL,
            magnitude   REAL,
            heart_rate  REAL,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (session_id, window_id, mode)
        );
    """)

    conn.commit()
    conn.close()


init_cloud_db()


# ── analytics helpers ─────────────────────────────────────────────────────────

def _stats(values):
    if not values:
        return None, None, None
    return round(sum(values) / len(values), 1), round(min(values), 1), round(max(values), 1)


def _process_cloud_window(conn, session_id, window_w):
    """Compute and persist an online window result. Only called on closed windows."""
    from section3 import fetch_heart_rate_from_db

    c = conn.cursor()
    start_sample = 400 * window_w
    end_sample   = start_sample + 399
    expected     = set(range(start_sample, end_sample + 1))

    c.execute("""
        SELECT sample_index, ch1, ch2, ch3 FROM cloud_samples
        WHERE session_id = ? AND sensor = 1
          AND sample_index BETWEEN ? AND ?
        ORDER BY sample_index ASC
    """, (session_id, start_sample, end_sample))
    s1_rows   = c.fetchall()
    missing_1 = len(expected - {r["sample_index"] for r in s1_rows})

    c.execute("""
        SELECT sample_index, ch1, ch2 FROM cloud_samples
        WHERE session_id = ? AND sensor = 2
          AND sample_index BETWEEN ? AND ?
        ORDER BY sample_index ASC
    """, (session_id, start_sample, end_sample))
    s2_rows   = c.fetchall()
    missing_2 = len(expected - {r["sample_index"] for r in s2_rows})

    complete = missing_1 == 0 and missing_2 == 0

    # Only publish a closed window
    if not complete:
        return

    s1_ch1_m, s1_ch1_mn, s1_ch1_mx = _stats([r["ch1"] for r in s1_rows if r["ch1"] is not None])
    s1_ch2_m, s1_ch2_mn, s1_ch2_mx = _stats([r["ch2"] for r in s1_rows if r["ch2"] is not None])
    s1_ch3_m, s1_ch3_mn, s1_ch3_mx = _stats([r["ch3"] for r in s1_rows if r["ch3"] is not None])
    s2_ch1_m, s2_ch1_mn, s2_ch1_mx = _stats([r["ch1"] for r in s2_rows if r["ch1"] is not None])
    s2_ch2_m, s2_ch2_mn, s2_ch2_mx = _stats([r["ch2"] for r in s2_rows if r["ch2"] is not None])

    s2_mags   = [math.sqrt(r["ch1"]**2 + r["ch2"]**2)
                 for r in s2_rows if r["ch1"] is not None and r["ch2"] is not None]
    magnitude = round(sum(s2_mags) / len(s2_mags), 1) if s2_mags else None

    # Use shared heart-rate logic
    heart_rate = fetch_heart_rate_from_db(c, session_id, window_w, table="cloud_samples")

    c.execute("""
        INSERT OR REPLACE INTO cloud_results (
            session_id, window_id, mode, missing_1, missing_2, complete,
            s1_ch1_mean, s1_ch1_min, s1_ch1_max,
            s1_ch2_mean, s1_ch2_min, s1_ch2_max,
            s1_ch3_mean, s1_ch3_min, s1_ch3_max,
            s2_ch1_mean, s2_ch1_min, s2_ch1_max,
            s2_ch2_mean, s2_ch2_min, s2_ch2_max,
            magnitude, heart_rate
        ) VALUES (?, ?, 'online', ?, ?, ?,
                  ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?, ?,
                  ?, ?)
    """, (
        session_id, window_w, missing_1, missing_2, 1 if complete else 0,
        s1_ch1_m, s1_ch1_mn, s1_ch1_mx,
        s1_ch2_m, s1_ch2_mn, s1_ch2_mx,
        s1_ch3_m, s1_ch3_mn, s1_ch3_mx,
        s2_ch1_m, s2_ch1_mn, s2_ch1_mx,
        s2_ch2_m, s2_ch2_mn, s2_ch2_mx,
        magnitude, heart_rate,
    ))


# ── endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "mayacardio-cloud"}


@app.post("/ingest")
async def ingest(request: Request):
    content_encoding = request.headers.get("content-encoding", "")
    raw_body = await request.body()
    payload_text = gzip.decompress(raw_body).decode() if "gzip" in content_encoding \
                   else raw_body.decode()

    data    = json.loads(payload_text)
    samples = data.get("samples", [])
    losses  = data.get("losses", [])
    results = data.get("results", [])   # offline results from device

    conn = get_cloud_db()
    c    = conn.cursor()

    # ── raw samples ───────────────────────────────────────────────────────────
    affected_windows = set()
    if samples:
        rows = []
        for s in samples:
            vals = s["values"]
            ch1  = vals[0] if len(vals) > 0 else None
            ch2  = vals[1] if len(vals) > 1 else None
            ch3  = vals[2] if len(vals) > 2 else None
            rows.append((s["session_id"], s["sensor"], s["index"], ch1, ch2, ch3))
            affected_windows.add((s["session_id"], s["index"] // 400))
        c.executemany("""
            INSERT OR IGNORE INTO cloud_samples
                (session_id, sensor, sample_index, ch1, ch2, ch3)
            VALUES (?, ?, ?, ?, ?, ?)
        """, rows)

    # ── losses (idempotent) ───────────────────────────────────────────────────
    for loss in losses:
        c.execute("""
            INSERT OR IGNORE INTO cloud_loss_log
                (session_id, sensor, from_index, to_index, reason)
            VALUES (?, ?, ?, ?, ?)
        """, (loss["session_id"], loss["sensor"],
              loss["from_index"], loss.get("to_index"), loss.get("reason", "unknown")))

    # ── offline results from device ───────────────────────────────────────────
    for res in results:
        c.execute("""
            INSERT OR REPLACE INTO cloud_results (
                session_id, window_id, mode, missing_1, missing_2, complete,
                s1_ch1_mean, s1_ch1_min, s1_ch1_max,
                s1_ch2_mean, s1_ch2_min, s1_ch2_max,
                s1_ch3_mean, s1_ch3_min, s1_ch3_max,
                s2_ch1_mean, s2_ch1_min, s2_ch1_max,
                s2_ch2_mean, s2_ch2_min, s2_ch2_max,
                magnitude, heart_rate
            ) VALUES (?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?,
                      ?, ?)
        """, (
            res["session_id"], res["window_id"], res["mode"],
            res["missing_1"], res["missing_2"], 1 if res["complete"] else 0,
            res.get("s1_ch1_mean"), res.get("s1_ch1_min"), res.get("s1_ch1_max"),
            res.get("s1_ch2_mean"), res.get("s1_ch2_min"), res.get("s1_ch2_max"),
            res.get("s1_ch3_mean"), res.get("s1_ch3_min"), res.get("s1_ch3_max"),
            res.get("s2_ch1_mean"), res.get("s2_ch1_min"), res.get("s2_ch1_max"),
            res.get("s2_ch2_mean"), res.get("s2_ch2_min"), res.get("s2_ch2_max"),
            res.get("magnitude"),   res.get("heart_rate"),
        ))

    conn.commit()

    # ── online window processing for closed windows only ──────────────────────
    for sess_id, win_id in affected_windows:
        _process_cloud_window(conn, sess_id, win_id)

    conn.commit()
    conn.close()

    return {"status": "accepted", "inserted": len(samples)}


@app.get("/inventory")
def inventory():
    """
    Return the highest confirmed sample_index per sensor per session.
    Used by the device to reconcile after outages.
    Response: { session_id: { "1": max_idx, "2": max_idx } }
    """
    conn = get_cloud_db()
    c    = conn.cursor()
    c.execute("""
        SELECT session_id, sensor, MAX(sample_index) AS max_idx
        FROM cloud_samples
        GROUP BY session_id, sensor
    """)
    out = {}
    for row in c.fetchall():
        sid = row["session_id"]
        if sid not in out:
            out[sid] = {}
        out[sid][str(row["sensor"])] = row["max_idx"]
    conn.close()
    return out


@app.get("/status")
def status():
    conn = get_cloud_db()
    c    = conn.cursor()
    c.execute("SELECT COUNT(*) FROM cloud_samples")
    n_samples = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM cloud_results")
    n_results = c.fetchone()[0]
    conn.close()
    return {"status": "online", "cloud_samples": n_samples, "windows_processed": n_results}


@app.get("/export", response_class=PlainTextResponse)
def export(session: str):
    conn = get_cloud_db()
    c    = conn.cursor()
    lines = []

    c.execute("""
        SELECT sensor, sample_index, ch1, ch2, ch3 FROM cloud_samples
        WHERE session_id = ? ORDER BY sensor, sample_index ASC
    """, (session,))
    for r in c.fetchall():
        vals = [r["ch1"], r["ch2"]]
        if r["ch3"] is not None:
            vals.append(r["ch3"])
        lines.append(json.dumps({"sensor": r["sensor"],
                                 "index":  r["sample_index"],
                                 "values": vals}))

    c.execute("""
        SELECT sensor, from_index, to_index, reason FROM cloud_loss_log
        WHERE session_id = ? ORDER BY sensor, from_index ASC
    """, (session,))
    for r in c.fetchall():
        lines.append(json.dumps({"sensor":     r["sensor"],
                                 "from_index": r["from_index"],
                                 "to_index":   r["to_index"],
                                 "reason":     r["reason"]}))

    conn.close()
    return "\n".join(lines)


@app.get("/results", response_class=PlainTextResponse)
def results(session: str):
    conn = get_cloud_db()
    c    = conn.cursor()
    c.execute("""
        SELECT * FROM cloud_results WHERE session_id = ?
        ORDER BY window_id, mode ASC
    """, (session,))
    lines = []
    for row in c.fetchall():
        lines.append(json.dumps({
            "session_id": row["session_id"],
            "window_id":  row["window_id"],
            "mode":       row["mode"],
            "missing_1":  row["missing_1"],
            "missing_2":  row["missing_2"],
            "complete":   bool(row["complete"]),
            "s1_ch1":  {"mean": row["s1_ch1_mean"],
                        "min":  row["s1_ch1_min"],
                        "max":  row["s1_ch1_max"]},
            "s1_ch2":  {"mean": row["s1_ch2_mean"],
                        "min":  row["s1_ch2_min"],
                        "max":  row["s1_ch2_max"]},
            "s1_ch3":  {"mean": row["s1_ch3_mean"],
                        "min":  row["s1_ch3_min"],
                        "max":  row["s1_ch3_max"]},
            "s2_ch1":  {"mean": row["s2_ch1_mean"],
                        "min":  row["s2_ch1_min"],
                        "max":  row["s2_ch1_max"]},
            "s2_ch2":  {"mean": row["s2_ch2_mean"],
                        "min":  row["s2_ch2_min"],
                        "max":  row["s2_ch2_max"]},
            "magnitude":  row["magnitude"],
            "heart_rate": row["heart_rate"],
        }))
    conn.close()
    return "\n".join(lines)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"[*] Starting MayaCardio Cloud Service on port {port}…")
    uvicorn.run(app, host="0.0.0.0", port=port)

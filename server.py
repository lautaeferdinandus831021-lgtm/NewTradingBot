#!/usr/bin/env python3
"""
BG-BOT v5 Backend Server
Serves the PWA + handles API communication with Bitget.
Run: python server.py
Then open http://localhost:8080 in your phone browser.
"""

import json, os, time, hmac, hashlib, base64
from http.server import HTTPServer, SimpleHTTPRequestHandler
import pandas as pd

try:
    import requests
except ImportError:
    print("Install: pip install requests pandas")
    exit(1)


# ═══ Bitget API Client ═══════════════════════════════
class BitgetClient:
    BASE = "https://api.bitget.com"

    def __init__(self, key, secret, passphrase, demo=True):
        self.key = key
        self.secret = secret
        self.passphrase = passphrase
        self.demo = demo
        self.sess = requests.Session()

    def _sign(self, ts, method, path, body=""):
        msg = ts + method.upper() + path + body
        mac = hmac.new(
            self.secret.encode(), msg.encode(), hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode()

    def _headers(self, method, path, body=""):
        ts = str(int(time.time()))
        h = {
            "ACCESS-KEY": self.key,
            "ACCESS-SIGN": self._sign(ts, method, path, body),
            "ACCESS-TIMESTAMP": ts,
            "ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }
        if self.demo:
            h["paptrading"] = "1"
        return h

    def _req(self, method, path, params=None, data=None):
        url = self.BASE + path
        body = json.dumps(data) if data else ""
        h = self._headers(method, path, body)
        try:
            r = self.sess.request(
                method, url, headers=h,
                params=params, data=body or None,
                timeout=10
            )
            return r.json()
        except Exception as e:
            return {"code": "99999", "msg": str(e)}

    def get_klines(self, symbol, gran, market="spot", limit=200):
        if market == "spot":
            path = "/api/v2/spot/market/candles"
            params = {"symbol": symbol, "granularity": gran,
                      "limit": str(limit)}
        else:
            path = "/api/v2/mix/market/candles"
            params = {"productType": "USDT-FUTURES",
                      "symbol": symbol, "granularity": gran,
                      "limit": str(limit)}
        result = self._req("GET", path, params)
        if not result or result.get("code") != "00000":
            return None
        df = pd.DataFrame(result["data"], columns=[
            "timestamp", "open", "high", "low",
            "close", "volume", "quote_volume"
        ])
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c])
        return df

    def get_balance(self, market="spot"):
        if market == "spot":
            r = self._req("GET", "/api/v2/spot/account/assets",
                          {"coin": "USDT"})
            if r and r.get("data") and r["data"]:
                return float(r["data"][0].get("available", 0))
        else:
            r = self._req("GET", "/api/v2/account/get-account-balance",
                          {"productType": "USDT-FUTURES"})
            if r and r.get("data"):
                return float(r["data"][0].get("available", 0))
        return 0

    def place_spot_order(self, symbol, side, size, otype="market",
                         price=None):
        data = {"symbol": symbol, "side": side,
                "orderType": otype, "force": "gtc", "size": str(size)}
        if price and otype == "limit":
            data["price"] = str(price)
        return self._req("POST", "/api/v2/spot/trade/place-order",
                         data=data)

    def place_perp_order(self, symbol, side, size, otype="market",
                         price=None, tp=None, sl=None):
        data = {
            "productType": "USDT-FUTURES", "symbol": symbol,
            "marginMode": "crossed", "marginCoin": "USDT",
            "size": str(size), "side": side, "orderType": otype,
        }
        if price and otype == "limit":
            data["price"] = str(price)
        if tp:
            data["presetStopSurplusPrice"] = str(tp)
        if sl:
            data["presetStopLossPrice"] = str(sl)
        return self._req("POST", "/api/v2/mix/order/place-order",
                         data=data)


# ═══ Indicator Engine ════════════════════════════════
class Indicators:
    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = [k for k, v in cfg.items()
                        if v.get("enabled")]

    def evaluate(self, df):
        signals = {}
        for name in self.enabled:
            try:
                fn = getattr(self, f"calc_{name}")
                signals[name] = fn(df, self.cfg[name])
            except:
                signals[name] = "NEUTRAL"
        return signals

    def calc_rsi(self, df, c):
        p = c.get("period", 14)
        delta = df["close"].diff()
        g = delta.where(delta > 0, 0).rolling(p).mean()
        l = (-delta.where(delta < 0, 0)).rolling(p).mean()
        rsi = 100 - (100 / (1 + g / l))
        v = rsi.iloc[-1]
        if v < c.get("oversold", 30): return "LONG"
        if v > c.get("overbought", 70): return "SHORT"
        return "NEUTRAL"

    def calc_macd(self, df, c):
        ef = df["close"].ewm(span=c.get("fast", 12)).mean()
        es = df["close"].ewm(span=c.get("slow", 26)).mean()
        h = (ef - es) - (ef - es).ewm(span=c.get("signal", 9)).mean()
        if h.iloc[-1] > 0 and h.iloc[-2] <= 0: return "LONG"
        if h.iloc[-1] < 0 and h.iloc[-2] >= 0: return "SHORT"
        if h.iloc[-1] > 0: return "LONG"
        if h.iloc[-1] < 0: return "SHORT"
        return "NEUTRAL"

    def calc_bb(self, df, c):
        p = c.get("period", 20)
        s = c.get("std_dev", 2)
        m = df["close"].rolling(p).mean()
        std = df["close"].rolling(p).std()
        pr = df["close"].iloc[-1]
        if pr <= (m - std * s).iloc[-1]: return "LONG"
        if pr >= (m + std * s).iloc[-1]: return "SHORT"
        return "NEUTRAL"

    def calc_ema(self, df, c):
        ef = df["close"].ewm(span=c.get("fast", 9)).mean()
        es = df["close"].ewm(span=c.get("slow", 21)).mean()
        if ef.iloc[-1] > es.iloc[-1]: return "LONG"
        return "SHORT"

    def calc_stoch(self, df, c):
        kp = c.get("k_period", 14)
        lo = df["low"].rolling(kp).min()
        hi = df["high"].rolling(kp).max()
        k = (100 * (df["close"] - lo) / (hi - lo)).rolling(
            c.get("smooth", 3)).mean()
        d = k.rolling(c.get("d_period", 3)).mean()
        if k.iloc[-1] < 20 and d.iloc[-1] < 20: return "LONG"
        if k.iloc[-1] > 80 and d.iloc[-1] > 80: return "SHORT"
        return "NEUTRAL"

    def calc_atr(self, df, c):
        p = c.get("period", 14)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(p).mean()
        avg = atr.rolling(50).mean().iloc[-1]
        if atr.iloc[-1] > avg * c.get("multiplier", 1.5):
            return "LONG" if df["close"].iloc[-1] > \
                df["close"].iloc[-5] else "SHORT"
        return "NEUTRAL"


# ═══ HTTP Server ═════════════════════════════════════
class BotHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.path = "/index.html"
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def main():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), BotHandler)
    print(f"""
    ╔══════════════════════════════════════╗
    ║   BG-BOT v5 — Dual Timeframe        ║
    ║   Server running on port {port}        ║
    ║                                      ║
    ║   Open on phone:                     ║
    ║   http://YOUR_IP:{port}               ║
    ║                                      ║
    ║   Install: Chrome → ⋮ →              ║
    ║   "Add to Home Screen"               ║
    ╚══════════════════════════════════════╝
    """)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
CLI Launcher: Crypto Structural Trend Paper Trading Web Dashboard (Phase 22)
=============================================================================
Starts the institutional web dashboard (10,000 USDT ETH paper trading),
manages ngrok public tunnel forwarding to port 8088, and starts the
background morning 08:02 BJT scheduler with automated email dispatch to
568701293@qq.com.

Usage:
    python scripts/run_paper_dashboard.py
    python scripts/run_paper_dashboard.py --port 8088 --send-startup-email
    python scripts/run_paper_dashboard.py --no-ngrok
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from crypto_quant.paper.config import (
    CONFIG_HASH,
    ENABLE_REAL_ORDERS,
    EXPERIMENT_ID,
    PAPER_MODE,
    get_code_commit,
    print_startup_banner,
)
from crypto_quant.paper.web_server import (
    DEFAULT_RECIPIENT,
    PORT,
    send_email_report,
    start_server,
)


def get_existing_ngrok_url() -> str:
    """Checks if ngrok is already running on local port 4040 and returns public HTTPS URL."""
    try:
        req = urllib.request.Request("http://127.0.0.1:4040/api/tunnels", headers={"User-Agent": "PaperRunner/1.0"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode())
            tunnels = data.get("tunnels", [])
            for t in tunnels:
                url = t.get("public_url", "")
                if url.startswith("https://"):
                    return url
            if tunnels:
                return tunnels[0].get("public_url", "")
    except Exception:
        pass
    return ""


def start_ngrok_tunnel(port: int, max_retries: int = 10) -> tuple:
    """
    Spawns ngrok http <port> process and retrieves the assigned public URL.
    Returns (process_handle, public_url).
    """
    existing_url = get_existing_ngrok_url()
    if existing_url:
        print(f"[TUNNEL] Found existing ngrok tunnel: {existing_url}")
        return None, existing_url

    print(f"[TUNNEL] Starting ngrok tunnel for port {port}...")
    try:
        p = subprocess.Popen(
            ["ngrok", "http", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
    except FileNotFoundError:
        print("[TUNNEL WARNING] 'ngrok' command not found on PATH. Proceeding without public tunnel.")
        return None, ""
    except Exception as e:
        print(f"[TUNNEL ERROR] Failed to start ngrok: {e}")
        return None, ""

    # Poll ngrok local web inspection API
    for i in range(max_retries):
        time.sleep(1.0)
        pub_url = get_existing_ngrok_url()
        if pub_url:
            print(f"[TUNNEL SUCCESS] Live public URL established: {pub_url}")
            return p, pub_url

    print("[TUNNEL WARNING] Timed out waiting for ngrok tunnel URL.")
    return p, ""


def main():
    parser = argparse.ArgumentParser(description="Crypto Paper Dashboard & Monitoring Server")
    parser.add_argument("--port", type=int, default=PORT, help=f"Web server port (Default: {PORT})")
    parser.add_argument("--no-ngrok", action="store_true", help="Disable automatic ngrok tunnel")
    parser.add_argument("--send-startup-email", action="store_true", default=True, help="Send startup notification email to 568701293@qq.com")
    parser.add_argument("--recipient", type=str, default=DEFAULT_RECIPIENT, help=f"Recipient email (Default: {DEFAULT_RECIPIENT})")
    args = parser.parse_args()

    # Hard safety assertion before doing anything
    if ENABLE_REAL_ORDERS is not False or PAPER_MODE is not True:
        print("FATAL: Safety lock violation! Real orders forbidden.", file=sys.stderr)
        sys.exit(1)

    print_startup_banner()
    print("=" * 70)
    print(" CRYPTO STRUCTURAL TREND PAPER DASHBOARD LAUNCHER")
    print(f" Experiment:    {EXPERIMENT_ID}")
    print(f" Code Commit:   {get_code_commit()[:8]}")
    print(f" Config Hash:   {CONFIG_HASH[:16]}...")
    print(f" Port:          {args.port}")
    print(f" Recipient:     {args.recipient}")
    print("=" * 70)

    ngrok_proc = None
    public_url = ""

    if not args.no_ngrok:
        ngrok_proc, public_url = start_ngrok_tunnel(args.port)

    effective_url = public_url if public_url else f"http://127.0.0.1:{args.port}"
    print(f"\n[DASHBOARD READY] Accessible URL: {effective_url}")
    print(f"[LOCAL ACCESS]   Local URL:      http://127.0.0.1:{args.port}\n")

    # Send initial email notification confirming the service is online
    if args.send_startup_email:
        print(f"[EMAIL] Sending launch confirmation email to {args.recipient}...")
        try:
            send_email_report(to_email=args.recipient, is_manual=True)
        except Exception as e:
            print(f"[EMAIL WARNING] Startup email dispatch encountered an issue: {e}")

    # Register cleanup handler for ngrok process
    def cleanup(signum, frame):
        print("\n[SHUTDOWN] Terminating dashboard and tunnel...")
        if ngrok_proc:
            try:
                ngrok_proc.terminate()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        # Start web server (blocking call, runs Flask & scheduler)
        start_server(port=args.port, public_url=public_url)
    finally:
        if ngrok_proc:
            try:
                ngrok_proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()

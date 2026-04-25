#!/usr/bin/env python3
"""
rerun_task1.py — Re-upload fixed scripts and restart Task 1 execution.
Run this after fixing 01_install_deps.sh and 02_build_fastdeploy.sh.
"""
import paramiko
import time
import sys
import os

HOST = "140.207.205.81"
PORT = 32222
USER = "root+vm-1Fe2g2PVUjoRh4Zq"
PASSWORD = "[REDACTED]"

LOCAL_ROOT = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FILES = [
    ("task1-warmup/scripts/01_install_deps.sh",     "/root/paddle-mx/task1-warmup/scripts/01_install_deps.sh"),
    ("task1-warmup/scripts/02_build_fastdeploy.sh", "/root/paddle-mx/task1-warmup/scripts/02_build_fastdeploy.sh"),
    ("task1-warmup/scripts/03_verify_install.sh",   "/root/paddle-mx/task1-warmup/scripts/03_verify_install.sh"),
    ("task1-warmup/email_template.md",              "/root/paddle-mx/task1-warmup/email_template.md"),
]

def ssh_connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD,
                   timeout=30, auth_timeout=30)
    return client

def run_cmd(client, cmd, wall_timeout=120, print_output=True):
    chan = client.get_transport().open_session()
    chan.get_pty()
    chan.exec_command(cmd)
    out_lines = []
    deadline = time.time() + wall_timeout
    while True:
        if chan.exit_status_ready():
            break
        if time.time() > deadline:
            print(f"\n  [TIMEOUT {wall_timeout}s]: {cmd[:60]}", flush=True)
            chan.close()
            return "".join(out_lines), -1
        if chan.recv_ready():
            chunk = chan.recv(4096).decode("utf-8", errors="replace")
            out_lines.append(chunk)
            if print_output:
                print(chunk, end="", flush=True)
        else:
            time.sleep(0.1)
    while chan.recv_ready():
        chunk = chan.recv(4096).decode("utf-8", errors="replace")
        out_lines.append(chunk)
        if print_output:
            print(chunk, end="", flush=True)
    exit_code = chan.recv_exit_status()
    chan.close()
    return "".join(out_lines), exit_code

def sftp_upload(client, local_rel, remote_path):
    sftp = client.open_sftp()
    remote_dir = os.path.dirname(remote_path)
    dirs = []
    d = remote_dir
    while d and d != "/":
        dirs.append(d)
        d = os.path.dirname(d)
    for d in reversed(dirs):
        try:
            sftp.mkdir(d)
        except IOError:
            pass
    local_path = os.path.join(LOCAL_ROOT, local_rel)
    sftp.put(local_path, remote_path)
    if remote_path.endswith(".sh"):
        sftp.chmod(remote_path, 0o755)
    sftp.close()

def banner(msg):
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}\n", flush=True)

def main():
    banner("Connecting to remote server")
    client = ssh_connect()
    print("  Connected!")

    # ── Re-upload fixed scripts ───────────────────────────────────────────────
    banner("Step 1: Re-upload fixed scripts via SFTP")
    for local_rel, remote_path in UPLOAD_FILES:
        print(f"  {local_rel} → {remote_path}")
        sftp_upload(client, local_rel, remote_path)
    print("  All scripts uploaded.")

    # ── Kill any stale build ──────────────────────────────────────────────────
    banner("Step 2: Kill stale build + clean FastDeploy dir")
    run_cmd(client, "pkill -f 02_build_fastdeploy.sh 2>/dev/null || true; echo 'old build killed'", wall_timeout=10)
    run_cmd(client, "rm -rf /root/FastDeploy /root/fastdeploy_build.log /root/fastdeploy_build.pid; echo 'cleaned'", wall_timeout=15)

    # ── Install Python deps ───────────────────────────────────────────────────
    banner("Step 3: Install Python dependencies")
    out, rc = run_cmd(client,
        "bash /root/paddle-mx/task1-warmup/scripts/01_install_deps.sh 2>&1",
        wall_timeout=600)
    if rc == 0:
        print("\n  [OK] Dependencies installed.")
    else:
        print(f"\n  [WARN] Install script returned exit code {rc} — check output above.")

    # ── Launch build in background ────────────────────────────────────────────
    banner("Step 4: Launch FastDeploy build in background (Gitee clone)")
    out, rc = run_cmd(client,
        "nohup bash /root/paddle-mx/task1-warmup/scripts/02_build_fastdeploy.sh "
        "> /root/fastdeploy_build.log 2>&1 & echo $! > /root/fastdeploy_build.pid && "
        "echo \"Build started with PID $(cat /root/fastdeploy_build.pid)\"",
        wall_timeout=30)
    print(out)

    # Wait and tail log
    time.sleep(15)
    print("\n--- First 50 lines of build log ---")
    run_cmd(client, "head -50 /root/fastdeploy_build.log", wall_timeout=10)

    banner("Done — build running in background")
    print("  Monitor: tail -f /root/fastdeploy_build.log")
    print("  PID file: /root/fastdeploy_build.pid")
    print("  Run check_build.py to check progress and verify wheel")
    client.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)

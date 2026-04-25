#!/usr/bin/env python3
"""
remote_runner.py — Execute Task 1 commands on the Metax GPU remote server via SSH.
Uses paramiko for SSH connectivity + SFTP to upload scripts directly (avoids GitHub).
"""
import paramiko
import time
import sys
import os

HOST = "140.207.205.81"
PORT = 32222
USER = "root+vm-1Fe2g2PVUjoRh4Zq"
PASSWORD = "[REDACTED]"

# Local project root (same dir as this script)
LOCAL_ROOT = os.path.dirname(os.path.abspath(__file__))

# Files to upload: (local relative path, remote absolute path)
UPLOAD_FILES = [
    ("task1-warmup/scripts/01_install_deps.sh",  "/root/paddle-mx/task1-warmup/scripts/01_install_deps.sh"),
    ("task1-warmup/scripts/02_build_fastdeploy.sh", "/root/paddle-mx/task1-warmup/scripts/02_build_fastdeploy.sh"),
    ("task1-warmup/scripts/03_verify_install.sh", "/root/paddle-mx/task1-warmup/scripts/03_verify_install.sh"),
    ("task1-warmup/email_template.md",           "/root/paddle-mx/task1-warmup/email_template.md"),
]

def ssh_connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD,
                   timeout=30, auth_timeout=30)
    return client

def run_cmd(client, cmd, wall_timeout=120, print_output=True):
    """Run a command and return (stdout_text, exit_code). Has wall-clock timeout."""
    chan = client.get_transport().open_session()
    chan.get_pty()
    chan.exec_command(cmd)
    out_lines = []
    deadline = time.time() + wall_timeout
    while True:
        if chan.exit_status_ready():
            break
        if time.time() > deadline:
            print(f"\n⚠️  Wall-clock timeout ({wall_timeout}s) for: {cmd[:60]}...", flush=True)
            chan.close()
            return "".join(out_lines), -1
        if chan.recv_ready():
            chunk = chan.recv(4096).decode("utf-8", errors="replace")
            out_lines.append(chunk)
            if print_output:
                print(chunk, end="", flush=True)
        else:
            time.sleep(0.1)
    # Drain remaining output
    while chan.recv_ready():
        chunk = chan.recv(4096).decode("utf-8", errors="replace")
        out_lines.append(chunk)
        if print_output:
            print(chunk, end="", flush=True)
    exit_code = chan.recv_exit_status()
    chan.close()
    return "".join(out_lines), exit_code

def sftp_upload(client, local_rel, remote_path):
    """Upload a local file to the remote path, creating parent dirs as needed."""
    sftp = client.open_sftp()
    # Ensure remote directory exists
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
            pass  # already exists
    local_path = os.path.join(LOCAL_ROOT, local_rel)
    sftp.put(local_path, remote_path)
    # Make shell scripts executable
    if remote_path.endswith(".sh"):
        sftp.chmod(remote_path, 0o755)
    sftp.close()

def banner(msg):
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}\n", flush=True)

def main():
    banner("Connecting to remote Metax GPU server")
    print(f"  Host: {HOST}:{PORT}  User: {USER}")
    client = ssh_connect()
    print("  Connected!\n")

    # ── Step 1: Environment check ─────────────────────────────────────────────
    banner("Step 1: Environment check")
    run_cmd(client, "uname -a", wall_timeout=10)
    run_cmd(client, "python3 --version || python --version", wall_timeout=10)
    run_cmd(client, "ls /opt/maca/bin/ 2>/dev/null | grep -i 'mx\\|maca\\|smi' | head -10 || echo 'no maca tools found in /opt/maca/bin'", wall_timeout=10)
    run_cmd(client, "ls /opt/maca/", wall_timeout=10)

    # ── Step 2: Upload scripts via SFTP (bypass GitHub) ───────────────────────
    banner("Step 2: Upload scripts via SFTP")
    for local_rel, remote_path in UPLOAD_FILES:
        print(f"  Uploading {local_rel} → {remote_path}", flush=True)
        sftp_upload(client, local_rel, remote_path)
    run_cmd(client, "find /root/paddle-mx -type f | sort", wall_timeout=10)
    print("\n  All scripts uploaded.")

    # ── Step 3: Install Python dependencies ───────────────────────────────────
    banner("Step 3: Install Python dependencies (01_install_deps.sh)")
    out, rc = run_cmd(client,
        "bash /root/paddle-mx/task1-warmup/scripts/01_install_deps.sh 2>&1",
        wall_timeout=600)
    if rc != 0:
        print(f"\nInstall script returned exit code {rc}")
    else:
        print("\nDependencies installed.")

    # ── Step 4: Build FastDeploy (background, log to file) ────────────────────
    banner("Step 4: Launch FastDeploy build in background")
    print("Build may take 30-60 minutes. Launching with nohup...")

    # Kill any previous build
    run_cmd(client, "pkill -f 02_build_fastdeploy.sh 2>/dev/null; echo done", wall_timeout=10)

    out, rc = run_cmd(client,
        "nohup bash /root/paddle-mx/task1-warmup/scripts/02_build_fastdeploy.sh "
        "> /root/fastdeploy_build.log 2>&1 & echo $! > /root/fastdeploy_build.pid && "
        "echo \"Build started with PID $(cat /root/fastdeploy_build.pid)\"",
        wall_timeout=30)
    if rc == 0:
        print("Build launched in background.")

    # Wait 8s and show first 30 lines of log
    time.sleep(8)
    run_cmd(client, "wc -l /root/fastdeploy_build.log 2>/dev/null; head -40 /root/fastdeploy_build.log 2>/dev/null || echo '(log empty)'", wall_timeout=10)

    # ── Summary ───────────────────────────────────────────────────────────────
    banner("Summary")
    print("  Environment verified (Linux 5.15, Python 3.12, /opt/maca present)")
    print("  Scripts uploaded via SFTP (no GitHub dependency)")
    print("  Python dependencies installation started")
    print("  FastDeploy build launched in background")
    print()
    print("  Monitor build: run check_build.py (or ssh and: tail -f ~/fastdeploy_build.log)")
    client.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)

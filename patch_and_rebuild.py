#!/usr/bin/env python3
"""
patch_and_rebuild.py — Upload fixed scripts, install paddle-metax-gpu (best-effort),
then relaunch FastDeploy build with the python→python3 symlink fix.
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

def ssh_connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30, auth_timeout=30)
    return c

def run_cmd(client, cmd, wall_timeout=120, print_output=True, label=None):
    if label:
        print(f"\n--- {label} ---", flush=True)
    chan = client.get_transport().open_session()
    chan.get_pty()
    chan.exec_command(cmd)
    out_lines = []
    deadline = time.time() + wall_timeout
    while True:
        if chan.exit_status_ready():
            break
        if time.time() > deadline:
            print(f"\n[TIMEOUT {wall_timeout}s]", flush=True)
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
    rc = chan.recv_exit_status()
    chan.close()
    return "".join(out_lines), rc

def sftp_upload(client, local_rel, remote_path):
    sftp = client.open_sftp()
    dirs = []
    d = os.path.dirname(remote_path)
    while d and d != "/":
        dirs.append(d)
        d = os.path.dirname(d)
    for d in reversed(dirs):
        try:
            sftp.mkdir(d)
        except IOError:
            pass
    sftp.put(os.path.join(LOCAL_ROOT, local_rel), remote_path)
    if remote_path.endswith(".sh"):
        sftp.chmod(remote_path, 0o755)
    sftp.close()

def main():
    print("Connecting...", flush=True)
    client = ssh_connect()
    print("Connected!\n")

    # 1. Upload fixed scripts
    print("=== Uploading fixed scripts ===")
    for local_rel, remote_path in [
        ("task1-warmup/scripts/02_build_fastdeploy.sh", "/root/paddle-mx/task1-warmup/scripts/02_build_fastdeploy.sh"),
        ("task1-warmup/scripts/01_install_deps.sh",     "/root/paddle-mx/task1-warmup/scripts/01_install_deps.sh"),
        ("task1-warmup/scripts/03_verify_install.sh",   "/root/paddle-mx/task1-warmup/scripts/03_verify_install.sh"),
    ]:
        print(f"  {local_rel}", flush=True)
        sftp_upload(client, local_rel, remote_path)
    print("Done.\n")

    # 2. Check what paddle-metax-gpu versions are available
    print("=== Checking available paddle-metax-gpu versions ===")
    run_cmd(client,
        "pip index versions paddle-metax-gpu "
        "--index-url https://www.paddlepaddle.org.cn/packages/nightly/maca/ 2>&1 | head -5",
        wall_timeout=30, label="pip index query")

    # 3. Also check if it's already installed (image might pre-install it)
    run_cmd(client, "pip show paddle-metax-gpu 2>/dev/null || echo 'not installed'",
            wall_timeout=10, label="pip show")

    # 4. Try to install it (no version pin — take whatever is available)
    print("\n=== Installing paddle-metax-gpu (no version pin) ===")
    run_cmd(client,
        "pip install paddle-metax-gpu "
        "-i https://www.paddlepaddle.org.cn/packages/nightly/maca/ "
        "--break-system-packages --no-cache-dir 2>&1",
        wall_timeout=300, label="pip install paddle-metax-gpu")

    # 5. Kill any old build, clean, and relaunch
    print("\n=== Killing old build + cleaning ===")
    run_cmd(client, "pkill -f 02_build_fastdeploy.sh 2>/dev/null || true", wall_timeout=10)
    run_cmd(client, "rm -rf /root/FastDeploy /root/fastdeploy_build.log /root/fastdeploy_build.pid; echo cleaned", wall_timeout=15)

    print("\n=== Launching FastDeploy build ===")
    run_cmd(client,
        "nohup bash /root/paddle-mx/task1-warmup/scripts/02_build_fastdeploy.sh "
        "> /root/fastdeploy_build.log 2>&1 & echo $! > /root/fastdeploy_build.pid && "
        "echo \"Build PID: $(cat /root/fastdeploy_build.pid)\"",
        wall_timeout=30)

    time.sleep(20)
    print("\n--- Build log so far (first 60 lines) ---")
    run_cmd(client, "head -60 /root/fastdeploy_build.log", wall_timeout=10)

    print("\n=== Done ===")
    print("Monitor: tail -f /root/fastdeploy_build.log")
    client.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)

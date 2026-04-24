#!/usr/bin/env python3
"""
fix_pep668_and_rebuild.py — Remove EXTERNALLY-MANAGED marker, inspect build.sh, relaunch build.
"""
import paramiko
import time
import sys
import os

HOST = "140.207.205.81"
PORT = 32222
USER = "root+vm-1Fe2g2PVUjoRh4Zq"
PASSWORD = "Internet=o1!"
LOCAL_ROOT = os.path.dirname(os.path.abspath(__file__))

def ssh_connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30, auth_timeout=30)
    return c

def run_cmd(client, cmd, wall_timeout=120, print_output=True, label=None):
    if label:
        print(f"\n=== {label} ===", flush=True)
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
    dirs, d = [], os.path.dirname(remote_path)
    while d and d != "/":
        dirs.append(d)
        d = os.path.dirname(d)
    for d in reversed(dirs):
        try: sftp.mkdir(d)
        except IOError: pass
    sftp.put(os.path.join(LOCAL_ROOT, local_rel), remote_path)
    if remote_path.endswith(".sh"):
        sftp.chmod(remote_path, 0o755)
    sftp.close()

def main():
    print("Connecting...", flush=True)
    client = ssh_connect()
    print("Connected!\n")

    # 1. Remove EXTERNALLY-MANAGED marker — safe in Docker containers
    print("=== Remove PEP 668 EXTERNALLY-MANAGED marker ===")
    run_cmd(client,
        "find /usr/lib/python3* -name 'EXTERNALLY-MANAGED' 2>/dev/null && "
        "rm -f /usr/lib/python3*/EXTERNALLY-MANAGED && echo 'removed' || echo 'not found'",
        wall_timeout=10)

    # 2. Inspect what build.sh actually does (pip calls)
    print("\n=== Inspect build.sh pip calls ===")
    run_cmd(client, "grep -n 'pip\\|install' /root/FastDeploy/build.sh | head -30",
            wall_timeout=10, label="build.sh pip lines")
    run_cmd(client, "head -80 /root/FastDeploy/build.sh", wall_timeout=10, label="build.sh head")

    # 3. Upload latest fixed scripts
    print("\n=== Upload fixed scripts ===")
    for rel, remote in [
        ("task1-warmup/scripts/02_build_fastdeploy.sh", "/root/paddle-mx/task1-warmup/scripts/02_build_fastdeploy.sh"),
    ]:
        sftp_upload(client, rel, remote)
        print(f"  uploaded {rel}")

    # 4. Kill old build, relaunch
    run_cmd(client, "pkill -f 02_build_fastdeploy.sh 2>/dev/null || true; rm -rf /root/FastDeploy /root/fastdeploy_build.log; echo cleaned", wall_timeout=15)
    run_cmd(client,
        "nohup bash /root/paddle-mx/task1-warmup/scripts/02_build_fastdeploy.sh "
        "> /root/fastdeploy_build.log 2>&1 & echo $! > /root/fastdeploy_build.pid && "
        "echo 'Build PID:' $(cat /root/fastdeploy_build.pid)",
        wall_timeout=30, label="Launch build")

    time.sleep(25)
    run_cmd(client, "wc -l /root/fastdeploy_build.log; tail -40 /root/fastdeploy_build.log", wall_timeout=10, label="Build log tail")

    client.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
check_build_and_metax.py — Monitor build log and investigate paddle-metax-gpu wheel.
"""
import paramiko
import time
import sys

HOST = "140.207.205.81"
PORT = 32222
USER = "root+vm-1Fe2g2PVUjoRh4Zq"
PASSWORD = os.environ.get("GITEEAI_PASS", "")

def ssh_connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30, auth_timeout=30)
    return c

def run_cmd(client, cmd, wall_timeout=60, print_output=True, label=None):
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

def main():
    print("Connecting...", flush=True)
    client = ssh_connect()
    print("Connected!\n")

    # 1. Check build progress
    run_cmd(client, "wc -l /root/fastdeploy_build.log && tail -30 /root/fastdeploy_build.log", wall_timeout=10, label="Build log tail")
    run_cmd(client, "ps aux | grep -E 'build|cmake|make' | grep -v grep | head -10", wall_timeout=10, label="Build processes")

    # 2. Investigate paddle-metax-gpu — check /opt/maca/wheel dir
    run_cmd(client, "ls /opt/maca/wheel/ 2>/dev/null | head -30 || echo 'no /opt/maca/wheel'", wall_timeout=10, label="MACA wheel dir")
    run_cmd(client, "find /opt/maca -name '*.whl' 2>/dev/null | head -20 || echo 'no whl in /opt/maca'", wall_timeout=15, label="MACA whl files")
    run_cmd(client, "find / -name 'paddle_metax*' -o -name 'paddle-metax*' 2>/dev/null | grep -v proc | head -20 || echo 'not found'", wall_timeout=20, label="Find paddle-metax files")
    run_cmd(client, "pip list 2>/dev/null | grep -i 'paddle\\|metax\\|maca'", wall_timeout=10, label="Installed paddle/metax packages")

    # 3. Check if the image pre-installed things via conda or elsewhere
    run_cmd(client, "conda list 2>/dev/null | grep -i 'paddle\\|metax' | head -20 || echo 'conda not found'", wall_timeout=10, label="Conda paddle/metax packages")
    run_cmd(client, "python3 -c 'import paddle; print(paddle.__version__, paddle.device.get_all_custom_device_type())' 2>&1", wall_timeout=15, label="Paddle version + devices")

    # 4. Check the maca index page via curl
    run_cmd(client, "curl -s 'https://www.paddlepaddle.org.cn/packages/nightly/maca/' 2>/dev/null | grep -o 'paddle[^\"]*\\.whl' | head -20 || echo 'curl failed'", wall_timeout=30, label="MACA index page")

    client.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)

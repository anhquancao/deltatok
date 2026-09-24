# Enable proxy via SSH tunnel

This guide creates a SOCKS5 proxy on your **local machine** and exposes it on **BSC** through a reverse SSH tunnel.

## Prerequisite

**Purpose:** make sure SSH login to BSC works without password prompts, so tunnels can be started and kept stable.

You need SSH key login enabled (as described here):

- <https://iffmd.fz-juelich.de/telCKzx5QRezSjAsr7M1Ww#SSH-key-procedure>

If you already have this SSH alias in `~/.ssh/config`, you can use `bsc` instead of the full hostname:

```sshconfig
Host bsc
    HostName glogin1.bsc.es
    User vale352205
    IdentityFile ~/.ssh/id_rsa
```

## Machines used in this guide

- **Machine A (Local machine):** your laptop/workstation.
- **Machine B (BSC login node):** `glogin1.bsc.es` (or alias `bsc`).

## Step 1: Start SSH service on Machine A (Local machine)

**Purpose:** ensure your local machine can accept SSH connections to `localhost`, which is required for the local dynamic SOCKS tunnel in Step 2.

Run on **Machine A**:

```sh
sudo apt update
sudo apt install openssh-server
sudo systemctl enable ssh
sudo systemctl start ssh
```

## Step 2: Start local SOCKS5 TCP socket on Machine A

**Purpose:** create a local SOCKS5 proxy endpoint at `localhost:1080` on Machine A.

Open a terminal on **Machine A** and run:

```sh
ssh -N -D 1080 localhost
```

Keep this terminal open.

## Step 3: Establish reverse SSH tunnel to BSC from Machine A

**Purpose:** publish the local SOCKS5 proxy from Machine A to Machine B, so BSC can reach it as `localhost:1080`.

Open a second terminal on **Machine A** and run:

```sh
ssh -o ExitOnForwardFailure=yes -N -R 15432:localhost:1080 bsc
```

If you do not use an SSH alias, use:

```sh
ssh -o ExitOnForwardFailure=yes -N -R 15432:localhost:1080 glogin1.bsc.es
```

Keep this terminal open too.

What this does:

- Opens `localhost:15432` on **Machine B** (`glogin1.bsc.es`).
- Forwards it to `localhost:1080` on **Machine A** (your local SOCKS5 socket).

## Step 4: Set proxy variables on Machine B (BSC shell)

**Purpose:** tell applications running on BSC to send HTTP/HTTPS traffic through the SOCKS5 endpoint exposed by the reverse tunnel.

Open a third terminal on **Machine A**, connect to **Machine B**, and set:

```sh
ssh bsc
export http_proxy="socks5://localhost:1080"
export https_proxy="socks5://localhost:1080"
```

If you do not use an SSH alias, connect with:

```sh
ssh glogin1.bsc.es
```

Then run:

```sh
export http_proxy="socks5://localhost:1080"
export https_proxy="socks5://localhost:1080"
```

After this, tools that respect these environment variables (for example `curl`, `pip`, `apt`, etc.) will use the tunnel proxy.

## Step 5: Test that everything works

**Purpose:** verify each layer of the setup (env vars, tunnel port, real outbound request) and quickly identify where a failure happens.

Run these checks on **Machine B** (`glogin1.bsc.es`):

1) Check that proxy variables are set:

```sh
echo "$http_proxy"
echo "$https_proxy"
```

Expected: both show `socks5://localhost:1080`.

2) Check that the reverse tunnel port is open on BSC:

```sh
nc -vz localhost 1080
```

Expected: connection succeeds (`succeeded` / `open`).

3) Test an HTTP request through the SOCKS proxy:

```sh
curl -I --proxy socks5h://localhost:1080 https://example.com
```

Expected: a response header with `HTTP/1.1 200 OK` (or `HTTP/2 200`).

4) Test `pip install` through the tunnel proxy:

```sh
http_proxy="socks5h://localhost:1080" https_proxy="socks5h://localhost:1080" python3 -m pip install --no-cache-dir -v colorama==0.4.6
```

Expected: output contains `Successfully installed colorama-0.4.6`.

If you get `Missing dependencies for SOCKS support.`, follow Step 6 first.

Optional cleanup:

```sh
python3 -m pip uninstall -y colorama
```

If one test fails:

- Verify both SSH commands (Step 2 and Step 3) are still running on **Machine A**.
- Re-run Step 4 in your current shell on **Machine B**.
- If needed, restart the two SSH tunnels from **Machine A**.

## Step 6: Fix `pip` SOCKS error on offline BSC (`PySocks` bootstrap)

**Purpose:** `pip` needs `PySocks` to use `socks5://` or `socks5h://` proxies. If BSC has no direct internet, install `PySocks` from a wheel copied from Machine A.

Run on **Machine A** (local machine with internet):

```sh
mkdir -p ~/wheelhouse_pysocks
python3 -m pip download --only-binary=:all: --dest ~/wheelhouse_pysocks PySocks==1.7.1
scp ~/wheelhouse_pysocks/PySocks-*.whl bsc:~/wheelhouse_pysocks/
```

Run on **Machine B** (BSC), inside your target Python environment:

```sh
source /gpfs/projects/ehpc551/envs/maskgit/bin/activate
python -m pip install --no-index ~/wheelhouse_pysocks/PySocks-*.whl
python -c "import socks; print('PySocks OK')"
```

Then retry the proxy test:

```sh
http_proxy="socks5h://localhost:1080" https_proxy="socks5h://localhost:1080" python -m pip install --no-cache-dir -v colorama==0.4.6
```

## Notes

- Both SSH commands from Step 2 and Step 3 must remain running.
- If either command exits, the proxy path breaks.

# Accessing JeanZay and Cougar from Computer 3 via SSH Tunnel

## Problem

- **Computer 2 (cougar)** can connect to both Karolina and JeanZay.
- **Computer 3** can connect to Karolina but **not** JeanZay.
- Computer 2 has no public IP, so it cannot be SSH'd into directly.

## Solution

Computer 2 opens **reverse SSH tunnels** through Karolina, forwarding ports to JeanZay and back to itself. Computer 3 then connects through Karolina using those forwarded ports.

```
Computer 3 ──SSH──▶ Karolina ──port 2222 (tunnel maintained by Computer 2)──▶ JeanZay
Computer 3 ──SSH──▶ Karolina ──port 2223 (tunnel maintained by Computer 2)──▶ Cougar
```

## Setup

### Step 1 — Open the tunnel (on Computer 2 / cougar)

Run the following command on Computer 2:

```bash
ssh -N -R 2222:jean-zay3.idris.fr:22 -R 2223:localhost:22 karolina
```

This tells Karolina to listen on port 2222 (forwarding to JeanZay) and port 2223 (forwarding back to cougar).

The `-N` flag means "don't open a shell" — the session just holds the tunnel open.

#### Keeping the tunnel alive permanently

The tunnel will drop if Computer 2 disconnects or the session times out. To keep it alive automatically, use `autossh`:

```bash
autossh -M 0 -N \
  -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" \
  -o "ExitOnForwardFailure=yes" \
  -R 2222:jean-zay3.idris.fr:22 \
  -R 2223:localhost:22 \
  karolina
```

`ExitOnForwardFailure=yes` is important: without it, if a remote forward fails to
bind (e.g. the port is still held on Karolina by a stale session after a network
blip), ssh keeps the session alive *without* the forwards. autossh then sees a
"healthy" ssh and never restarts it — a silent deadlock where autossh is running
but the tunnel is dead. With this option, a failed forward makes ssh exit, so
autossh retries until Karolina reaps the stale forward and the bind succeeds.

Optionally, create a systemd service at `~/.config/systemd/user/tunnel-jeanzay.service`:

```ini
[Unit]
Description=Reverse SSH tunnel to JeanZay via Karolina
After=network-online.target

[Service]
ExecStart=ssh -N -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" -o "ExitOnForwardFailure=yes" -R 2222:jean-zay3.idris.fr:22 -R 2223:localhost:22 karolina
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
```

Enable it with:

```bash
systemctl --user enable --now tunnel-jeanzay
```

### Step 2 — Configure SSH on Computer 3

Add the following to `~/.ssh/config` on Computer 3:

```ssh-config
Host karolina
    HostName login1.karolina.it4i.cz
    User it4i-anhquan
    Port 22
    IdentityFile ~/.ssh/id_rsa
    ServerAliveInterval 60
    ServerAliveCountMax 3

Host jeanzay
    HostName localhost
    Port 2222
    User uyl37fq
    ProxyJump karolina

Host cougar
    HostName localhost
    Port 2223
    User acao
    ProxyJump karolina
```
### Step 3 — Connect from Computer 3

```bash
ssh jeanzay   # connect to JeanZay
ssh cougar    # connect to cougar
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Connection refused` on port 2222 | The tunnel on Computer 2 is not running. Restart it. |
| `channel 0: open failed` | JeanZay may be down, or the hostname `jean-zay3.idris.fr` is unreachable from Computer 2. |
| Tunnel drops frequently | Use `autossh` or the systemd service with `ServerAliveInterval`. |
| `Warning: remote port forwarding failed` | Port 2222 is already in use on Karolina. Pick a different port (e.g., 2223) and update both sides. |
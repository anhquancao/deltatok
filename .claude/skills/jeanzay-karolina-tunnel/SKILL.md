---
name: jeanzay-karolina-tunnel
description: Connect between Jean Zay (IDRIS) and Karolina over the reverse SSH tunnel cougar holds open. Use when reaching Jean Zay from this machine, pushing data/checkpoints Karolina→Jean Zay, diagnosing `Connection refused` / `channel open failed` on `ssh jean-zay`, or restoring the tunnel.
---

# Jean Zay ↔ Karolina reverse SSH tunnel

This machine and Karolina cannot reach Jean Zay (IDRIS) directly. A third box, **cougar**, can reach both, so it holds **reverse SSH tunnels open through Karolina**. Everything that reaches Jean Zay from here or from Karolina rides those forwards.

Source of truth: `docs/ssh_tunnel_jz.md` (this repo).

## Topology

```
this machine ──ssh──▶ Karolina ──port 2222 (reverse forward held by cougar)──▶ Jean Zay (jean-zay3.idris.fr:22)
this machine ──ssh──▶ Karolina ──port 2223 (reverse forward held by cougar)──▶ cougar (localhost:22)
```

cougar runs one `ssh -N -R` session to Karolina that binds **two** ports on Karolina:
- **`localhost:2222` on Karolina → Jean Zay login (jean-zay3:22)**
- **`localhost:2223` on Karolina → cougar (cougar:22)**

Both forwards live in the **same** ssh session, so they go up and down together: if 2222 is dead, 2223 is almost certainly dead too.

## Host aliases on this machine (`~/.ssh/config`)

| Alias | Route | Reaches |
|---|---|---|
| `karolina` | direct (`login1.karolina.it4i.cz`) | Karolina login |
| `jean-zay` | `ProxyCommand ssh -W %h:%p impala` | Jean Zay via the **impala** jump host (a separate, non-tunnel route) |
| `test-jeanzay` | `ProxyJump karolina` → `localhost:2222` | Jean Zay **via the Karolina tunnel** |
| `cougar-via-karolina` | `ProxyJump karolina` → `localhost:2223` | cougar via the Karolina tunnel |

The ssh config is machine-level, so these aliases are the same in every repo. Two distinct routes to Jean Zay exist: the **impala** route (`jean-zay`) and the **tunnel** route (`test-jeanzay`, port 2222). If `jean-zay` fails because impala is unreachable, try the tunnel route `test-jeanzay`, and vice-versa.

## Reach Jean Zay from this machine

```bash
ssh jean-zay "..."        # via impala
ssh test-jeanzay "..."    # via the Karolina reverse tunnel (port 2222)
```

Both are reliable for **read-only inspection** (`squeue`, `sacct`, log tails, `ls`, `module avail`) and `sbatch` submission. For job/env conventions use the `jeanzay-job` skill — this skill is only about the connection.

## Connect Karolina → Jean Zay (the `localhost:2222` trick)

From a shell **on Karolina**, Jean Zay's login node is `localhost:2222` (that's where cougar's reverse forward lands). This is how data and checkpoints move Karolina→Jean Zay:

```bash
# run ON Karolina
ssh -p 2222 uyl37fq@localhost "..."                          # shell on Jean Zay
rsync -av -e 'ssh -p 2222' <src> uyl37fq@localhost:<dst>     # push files to Jean Zay
```

Example — push a checkpoint into the Jean Zay checkout (run on Karolina):

```bash
rsync -av -e 'ssh -p 2222' <ckpt> \
  uyl37fq@localhost:/lustre/fswork/projects/rech/trg/uyl37fq/code/deltatok/checkpoints/
```

`$TRG_WORK` = `/lustre/fswork/projects/rech/trg/uyl37fq`; the checkout is `$TRG_WORK/code/deltatok`. Scripted Karolina→Jean Zay data sync (when present, run on Karolina) drives Jean Zay's **backup tier** `/lustre/fsstor/projects/rech/trg/uyl37fq/datasets_preprocess_backup` through this same port 2222. Pulling Jean Zay→Karolina also rsyncs **on Karolina** through `localhost:2222`.

## The tunnel is held by cougar — you cannot restart it from here

The forwards only exist while cougar's `ssh -N -R` session is up. If it dies, this machine has **no way to restart it** (the only route to cougar, port 2223, dies with the same session). Restoring it requires access to cougar. On cougar it is meant to self-heal via `autossh` / a systemd user service (`tunnel-jeanzay`):

```bash
# on cougar — restart manually if not managed by systemd
autossh -M 0 -N \
  -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" \
  -o "ExitOnForwardFailure=yes" \
  -R 2222:jean-zay3.idris.fr:22 -R 2223:localhost:22 karolina
```

`ExitOnForwardFailure=yes` matters: without it, a forward that fails to bind (stale port held on Karolina after a blip) leaves ssh "healthy" but tunnel-less, and autossh never restarts it — a silent dead tunnel. **If the tunnel is down, tell the user it needs restarting on cougar; do not try to fix it from here.**

## Troubleshooting

| Symptom (on `ssh test-jeanzay` / port 2222) | Cause | Action |
|---|---|---|
| `Connection refused` on port 2222 | cougar's tunnel session is down | Tunnel needs restarting **on cougar** (autossh/systemd) — ask the user |
| `channel 0: open failed: connect failed` | Tunnel up but Jean Zay unreachable from cougar (Jean Zay down, or `jean-zay3.idris.fr` not resolving) | Wait / check Jean Zay status; nothing to fix on this side |
| `Warning: remote port forwarding failed for listen port 2222` | Port 2222 still held on Karolina by a stale session | A stale forward must be reaped on Karolina before the new bind succeeds — restart on cougar with `ExitOnForwardFailure=yes` |
| `jean-zay` fails but `test-jeanzay` works (or vice-versa) | One of the two routes (impala vs tunnel) is down | Use the other alias |

## Hard rules

Read-only inspection over the tunnel is fine. Never `rsync`/`scp` **source files** local→cluster or run destructive `git` on the Jean Zay or Karolina checkouts (see CLAUDE.md). If a connection issue blocks an operation, stop and ask — don't "fix" the tunnel or sync state.

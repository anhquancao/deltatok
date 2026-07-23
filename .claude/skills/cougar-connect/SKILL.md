---
name: cougar-connect
description: Connect to an interactive job on cougar (vai GPU cluster) directly with plain ssh + the scheduler REST API — no interactive GPUCluster4U `cluster` CLI. Use to open/rebuild the tunnel to a running job non-interactively, query jobs/machines, or rsync files in/out of the job. Distinct from the reverse-tunnel "cougar" in `jeanzay-karolina-tunnel`.
---

# Connect to cougar (vai) directly — no `cluster` CLI

The GPUCluster4U `cluster` CLI (v2.0.0, `~/miniforge3/bin/cluster`) is interactive (prompts, auto-ssh). Under the hood it is only: **one `ssh -L` tunnel to the scheduler + a REST API + one `ssh -L` tunnel to the job**. All of it can be done with plain `ssh`/`curl`, fully non-interactively. Mechanics read from the package source (`~/miniforge3/lib/python3.12/site-packages/gpucluster4u_{client,core}/`), verified working 2026-07-23.

> **Not the same "cougar" as `jeanzay-karolina-tunnel`.** That skill reaches the base cougar box over the Karolina reverse tunnel (`cougar-via-karolina`). This skill reaches an **interactive job container on cougar** over the vai scheduler route. Same physical machine, different environments.

## Topology & constants (vai cluster)

```
WSL ──ssh -L 5001──▶ shared_user@10.46.63.92 (vai scheduler) ──▶ REST API on localhost:5001
WSL ──ssh -L <local>──▶ root@<machine_ip>:22 ──▶ localhost:<job_port> (job container sshd)
```

| Constant | Value |
|---|---|
| Scheduler | `shared_user@10.46.63.92`, ssh port 22, API port **5001** |
| API auth headers | `X-Username: acao` and `X-Client-Version: 2.0.0` (server checks the version) |
| cougar machine | ip `10.46.63.67`, job tunnels as **`root@`**, ssh port 22 |
| In-job login | user **`acao`** |
| Port bookkeeping | `~/.gpucluster4u/user_setup.yml` → `job_ports: {'<job_id>': <local_port>}` |

Auth for `shared_user@scheduler` and `root@machine` is the user's ssh key (already authorized). Machine IPs for other boxes: `GET /machines/<name>`.

## Step 1 — scheduler tunnel (replaces `cluster connect`)

```bash
ssh -f -N -q -T -o StrictHostKeyChecking=no -o ServerAliveInterval=60 \
    -o ConnectTimeout=15 -o ExitOnForwardFailure=yes \
    -L 5001:localhost:5001 shared_user@10.46.63.92 -p 22
curl -s -m 10 -H "X-Username: acao" -H "X-Client-Version: 2.0.0" http://localhost:5001/check
```

Skip if already up (`curl /check` answers). Unlike the CLI, this does **not** kill pre-existing tunnels.

## Step 2 — find the job and its sshd port

```bash
H=(-H "X-Username: acao" -H "X-Client-Version: 2.0.0")
curl -s "${H[@]}" "http://localhost:5001/jobs?user=acao"          # my jobs → id
curl -s "${H[@]}" "http://localhost:5001/jobs/<JOB_ID>/connect"   # → {"machine_name":"cougar","port":<job_port>}
curl -s "${H[@]}" "http://localhost:5001/machines/cougar"         # → {"ip":"10.46.63.67","shared_user":"root","ssh_port":22,...}
```

Other useful endpoints (all GET, same headers): `/jobs/<id>` (detail), `/jobs/<id>/logs`, `/gpustats?machine_name=cougar`, `/gpu_owners`, `/config`, `/version`.

## Step 3 — job tunnel + login (replaces `cluster jobs connect`)

Pick the local port from `job_ports` in `~/.gpucluster4u/user_setup.yml` if the job already has one (keeps VS Code `Host cougar_job-<id>` blocks valid); any free port works.

```bash
# tunnel: local <LOCAL_PORT> → job sshd <JOB_PORT> on cougar
ssh -f -N -q -T -o StrictHostKeyChecking=no -o ServerAliveInterval=60 \
    -o ConnectTimeout=15 -o ExitOnForwardFailure=yes \
    -L <LOCAL_PORT>:localhost:<JOB_PORT> root@10.46.63.67 -p 22

ssh -p <LOCAL_PORT> -o StrictHostKeyChecking=no acao@localhost "..."   # shell / commands
rsync -av -e "ssh -p <LOCAL_PORT> -o StrictHostKeyChecking=no" <src> acao@localhost:<dst>   # files in
```

`ssh -f` returns before the forward is usable — retry the first inner ssh once if it is refused immediately after tunnel creation.

The `<JOB_PORT>` is stable for the lifetime of the job (e.g. job 21905 → cougar:32875, local 44547); re-resolve via `/jobs/<id>/connect` after a job restart.

## Housekeeping

```bash
ps x | grep -- "-L <LOCAL_PORT>:localhost:" | grep -v grep     # tunnel alive?
pkill -f -- "-L <LOCAL_PORT>:localhost:"                       # kill one tunnel
```

- **Do not run the `cluster` CLI while hand-built tunnels are up**: `cluster connect` starts by killing *every* local ssh process whose cmdline matches `:localhost:` (see `kill_ssh_tunnels` in `gpucluster4u_client/utils/sshtunnel.py`) — it will take down these tunnels (and any other `-L x:localhost:y` forward on this machine).
- The job container is an **interactive dev environment** — the "never sync / read-only" HPC-cluster rules in CLAUDE.md do not apply to it.
- If the scheduler tunnel dies, everything in steps 2–3 fails with connection refused on 5001; just redo step 1.

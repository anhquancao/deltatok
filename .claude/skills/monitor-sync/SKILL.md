---
name: monitor-sync
description: Push the local deltatok checkout to BSC or Jean Zay through the ../monitor_jobs syncer — start its server if it is down, start the per-cluster watcher via the HTTP API, then verify by md5. Use whenever a cluster copy is stale before sbatch, or when the user asks to "sync", "push the code", or "start syncing" to a cluster.
---

# Syncing code to a cluster via monitor_jobs

This is the only allowed way to push source files from local to a cluster. Never raw `rsync` or `scp`.

The syncer runs `rsync -az --delete` with the `excludes` of the `Deltatok` entry in `../monitor_jobs/data/projects.json`. After the first push it keeps watching and pushes every local change (1 s debounce) until stopped.

**`--delete` removes remote files that do not exist locally**, unless they are excluded. Before the first push, make sure nothing cluster-only and un-excluded lives in the checkout.

## Targets

| `cluster=` | Remote checkout |
|---|---|
| `BSC` | `bsc:/gpfs/projects/ehpc1001/code/deltatok/` |
| `Jeanzay` | `jean-zay:/lustre/fswork/projects/rech/trg/uyl37fq/code/deltatok/` (needs the reverse tunnel: `jeanzay-karolina-tunnel` skill) |

The project key is `Deltatok`. `Karolina` is also listed but is deprecated, so never start it.

## 1. Make sure the server is up

```bash
curl -s -o /dev/null -w "%{http_code}\n" --max-time 5 http://127.0.0.1:9000/api/sync/state
```

`200` means it is up: go to step 2. `000` means it is down. Start it with Bash `run_in_background: true` and a long `timeout` (7200000), because it is a long-lived process. Run it from `~/code`, since `monitor_jobs` is the package directory:

```bash
cd /home/acao/code && python3 -m monitor_jobs
```

Then poll until it answers. On 2026-10-02 it took about 1 s:

```bash
for i in $(seq 1 30); do c=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 http://127.0.0.1:9000/api/sync/state); [ "$c" = "200" ] && break; sleep 1; done; echo "http=$c"
```

The server lives only as long as that background task, which is capped at 2 h (`timeout` 7200000 is the maximum). After that, or when the session ends, the server and every syncer stop. Re-run step 1 before relying on it again.

## 2. Start the syncer

All sync routes are **GET** (`do_GET` in `../monitor_jobs/__main__.py`):

```bash
curl -s --max-time 30 "http://127.0.0.1:9000/api/sync/start?project=Deltatok&cluster=BSC"
```

It returns the syncer snapshot with `status: starting`. The statuses are:
- **`starting`:** the watcher is coming up.
- **`running`:** an rsync is in flight.
- **`watching`:** live and idle.
- **`error`:** see the `error` field.
- **`stopped`:** off.

## 3. Wait for the first push

```bash
for i in $(seq 1 40); do r=$(curl -s --max-time 5 http://127.0.0.1:9000/api/sync/state | python3 -c "
import json,sys
s=[x for x in json.load(sys.stdin)['syncers'] if x['key']=='Deltatok:BSC'][0]
print(s['status'], s['last_sync_rc'], s['last_sync_at'], s['error'])"); set -- $r; [ "$1" = "watching" ] && [ "$2" != "None" ] && break; [ "$1" = "error" ] && break; sleep 3; done; echo "$r"
```

It is done when this prints `watching 0 <timestamp> None`. The first full push to BSC took about 16 s. If you get a non-zero rc or `error`, report it. Don't retry blindly.

## 4. Verify the file you are about to run

An rc of 0 is not proof. md5 the exact file you need:

```bash
md5sum <file>; ssh bsc "md5sum /gpfs/projects/ehpc1001/code/deltatok/<file>"
```

A later local edit is pushed a few seconds after the save. Re-check the md5 after every edit, before `sbatch`.

## Stop

```bash
curl -s --max-time 30 "http://127.0.0.1:9000/api/sync/stop?project=Deltatok&cluster=BSC"
```

## Rules

- **Only this skill pushes code.** No `rsync` or `scp`, and no `git pull` on the cluster.
- **Leave other syncers alone.** Don't start or stop OccAny or VATIX syncers unless asked.
- **Sync only.** Read job state from `../monitor_jobs/data/*.json`, not from the API (CLAUDE.md).
- **Excluded paths never reach the cluster.** A new local output dir must be added to the `Deltatok` `excludes` and to `.gitignore`.

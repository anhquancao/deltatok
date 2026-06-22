"""SLURM allocation introspection helpers.

Used by the training loop to stop cleanly right after a checkpoint when the
wall-clock limit is near (so a chained resume picks up checkpoint-last) instead
of being killed mid-epoch (TIMEOUT).
"""
import datetime
import os
import re
import subprocess


def slurm_time_remaining_sec():
    """Seconds until the current SLURM allocation's wall-clock limit.

    Returns ``None`` when not running under SLURM, when ``scontrol`` is
    unavailable, or when ``EndTime`` cannot be parsed -- callers treat ``None``
    as "no limit known" and keep training. ``scontrol show job <id>`` reports
    ``EndTime`` in node-local time, matching ``datetime.now()``.
    """
    job_id = os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOBID')
    if not job_id:
        return None
    try:
        proc = subprocess.run(
            ['scontrol', 'show', 'job', job_id, '-o'],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    m = re.search(r'\bEndTime=(\S+)', proc.stdout)
    if not m or m.group(1) in ('Unknown', 'None', 'N/A'):
        return None
    try:
        end = datetime.datetime.strptime(m.group(1), '%Y-%m-%dT%H:%M:%S')
    except ValueError:
        return None
    # Compare as POSIX timestamps; naive wall-clock subtraction would be off by
    # an hour across a DST change. EndTime is node-local, matching now().
    return end.timestamp() - datetime.datetime.now().timestamp()

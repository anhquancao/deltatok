# SLURM-friendly metric logging (one flushed print line per print_freq steps).
# Copied from third_party/croco/utils/misc.py (MAE-derived), with log_every
# extended to print this rank's CPU RSS (process + its dataloader workers)
# next to the peak GPU memory.
import datetime
import os
import resource
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def _iter_cgroup_dirs():
    """Yield this process's memory-cgroup dir and all its ancestors.

    Handles both cgroup v2 (unified hierarchy, /proc/self/cgroup line '0::…')
    and v1 (line whose controller list contains 'memory'). SLURM typically
    sets the job's memory limit on an ancestor of the leaf cgroup the task
    runs in, hence the upward walk.
    """
    try:
        with open('/proc/self/cgroup') as f:
            lines = [line.rstrip('\n').split(':', 2) for line in f]
    except OSError:
        return
    for hier_id, controllers, path in lines:
        if hier_id == '0':                          # cgroup v2 unified hierarchy
            base = '/sys/fs/cgroup'
        elif 'memory' in controllers.split(','):    # cgroup v1 memory controller
            base = '/sys/fs/cgroup/memory'
        else:
            continue
        d = os.path.normpath(base + path)
        while d.startswith(base):
            yield d
            if d == base:
                break
            d = os.path.dirname(d)


def _read_cgroup_bytes(cgroup_dir, filenames):
    """First parseable byte count among `filenames` in `cgroup_dir`, else None.

    'max' (v2) and ~2**63 (v1) both mean "no limit set here" and are skipped.
    """
    for name in filenames:
        try:
            with open(os.path.join(cgroup_dir, name)) as f:
                raw = f.read().strip()
        except (OSError, ValueError):
            continue
        if raw and raw != 'max':
            value = int(raw)
            if value < 1 << 60:
                return value
    return None


def max_cpu_mem_mb():
    """Peak CPU memory of the job in MB.

    Prefers the cgroup-wide peak (memory.peak / memory.max_usage_in_bytes),
    which covers every process in the SLURM job cgroup on this node —
    dataloader workers and co-located DDP ranks included — i.e. the number
    the OOM killer actually compares against the limit. Falls back to this
    process's peak RSS where the kernel doesn't expose a cgroup peak.
    """
    for cgroup_dir in _iter_cgroup_dirs():
        peak = _read_cgroup_bytes(cgroup_dir, ('memory.peak', 'memory.max_usage_in_bytes'))
        if peak is not None:
            return peak / (1024.0 * 1024.0)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # ru_maxrss is KB on Linux


_TOTAL_CPU_MEM_MB = None


def total_cpu_mem_mb():
    """CPU memory ceiling in MB (cached).

    The tightest cgroup limit over the process's cgroup ancestry — under
    SLURM that is the --mem budget the OOM killer enforces. Falls back to
    the node's MemTotal when no cgroup limit is set.
    """
    global _TOTAL_CPU_MEM_MB
    if _TOTAL_CPU_MEM_MB is None:
        limits = []
        for cgroup_dir in _iter_cgroup_dirs():
            limit = _read_cgroup_bytes(cgroup_dir, ('memory.max', 'memory.limit_in_bytes'))
            if limit is not None:
                limits.append(limit)
        if limits:
            _TOTAL_CPU_MEM_MB = min(limits) / (1024.0 * 1024.0)
        else:
            _TOTAL_CPU_MEM_MB = 0.0
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        _TOTAL_CPU_MEM_MB = int(line.split()[1]) / 1024.0  # kB -> MB
                        break
    return _TOTAL_CPU_MEM_MB


def rank_cpu_mem_mb():
    """Current RSS of this rank's process tree in MB.

    Sums this process plus its children (the rank's own dataloader workers),
    so the number scales with the per-GPU `num_workers` setting — unlike the
    cgroup peak, which aggregates every rank co-located on the node. Pages
    shared copy-on-write between workers are counted once per process, so
    this slightly overestimates.
    """
    try:
        import psutil
        proc = psutil.Process()
        rss = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return rss / (1024.0 * 1024.0)
    except ImportError:
        # ru_maxrss is this process's lifetime peak (KB on Linux), workers excluded.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def rank_cpu_budget_mb():
    """This rank's share of the job memory limit in MB (limit / local ranks).

    The OOM killer enforces the node-level job limit, so one rank may borrow
    another's headroom — this is the fair-share guideline, not a hard cap.
    """
    local_ranks = 1
    # torchrun sets LOCAL_WORLD_SIZE; srun-launched DDP sets SLURM_NTASKS_PER_NODE
    # (which can look like '2(x4)' for multi-node jobs).
    # SLURM_GPUS_ON_NODE last: one rank per GPU is this repo's launch contract.
    for var in ('LOCAL_WORLD_SIZE', 'SLURM_NTASKS_PER_NODE', 'SLURM_GPUS_ON_NODE'):
        raw = os.environ.get(var, '')
        digits = raw.split('(')[0]
        if digits.isdigit():
            local_ranks = max(1, int(digits))
            break
    return total_cpu_mem_mb() / local_ranks


def total_gpu_mem_mb():
    """Total memory of the current CUDA device in MB (0 if no CUDA)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024.0 * 1024.0)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, max_iter=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        len_iterable = min(len(iterable), max_iter) if max_iter else len(iterable)
        space_fmt = ':' + str(len(str(len_iterable))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max gpu mem: {gpu_memory:.0f} ({gpu_total:.0f})')
        log_msg.append('cpu mem: {cpu_rank:.0f} ({cpu_rank_budget:.0f})')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for it, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len_iterable - 1:
                eta_seconds = iter_time.global_avg * (len_iterable - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len_iterable, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        gpu_memory=torch.cuda.max_memory_allocated() / MB,
                        gpu_total=total_gpu_mem_mb(),
                        cpu_rank=rank_cpu_mem_mb(),
                        cpu_rank_budget=rank_cpu_budget_mb()), flush=True)
                else:
                    print(log_msg.format(
                        i, len_iterable, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        cpu_rank=rank_cpu_mem_mb(),
                        cpu_rank_budget=rank_cpu_budget_mb()), flush=True)
            i += 1
            end = time.time()
            if max_iter and it >= max_iter:
                break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len_iterable), flush=True)

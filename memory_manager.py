"""
memory_manager.py — aggressive resource scaling via psutil.
"""
import os
import math
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch as _torch
    _CUDA = _torch.cuda.is_available()
except ImportError:
    _CUDA = False


class MemoryManager:
    _USAGE_CEILING = 0.85   # raised from 0.80
    _OS_RESERVE_GB = 0.5    # reduced from 1.0

    def __init__(self):
        total_bytes = 8 * (1024 ** 3)  # default 8GB
        available_bytes = 4 * (1024 ** 3)  # default 4GB
        used_bytes = 4 * (1024 ** 3)
        percent_used = 50.0

        if HAS_PSUTIL:
            try:
                mem = psutil.virtual_memory()
                total_bytes = mem.total
                available_bytes = mem.available
                used_bytes = mem.used
                percent_used = mem.percent
            except Exception:
                pass
        else:
            # Fallback for Windows using ctypes to avoid psutil compilation issues
            import sys
            if sys.platform == "win32":
                try:
                    import ctypes
                    from ctypes import wintypes

                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", wintypes.DWORD),
                            ("dwMemoryLoad", wintypes.DWORD),
                            ("ullTotalPhys", ctypes.c_uint64),
                            ("ullAvailPhys", ctypes.c_uint64),
                            ("ullTotalPageFile", ctypes.c_uint64),
                            ("ullAvailPageFile", ctypes.c_uint64),
                            ("ullTotalVirtual", ctypes.c_uint64),
                            ("ullAvailVirtual", ctypes.c_uint64),
                            ("ullAvailExtendedVirtual", ctypes.c_uint64),
                        ]

                    stat = MEMORYSTATUSEX()
                    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                        total_bytes = stat.ullTotalPhys
                        available_bytes = stat.ullAvailPhys
                        used_bytes = stat.ullTotalPhys - stat.ullAvailPhys
                        percent_used = float(stat.dwMemoryLoad)
                except Exception:
                    pass

        self.total_bytes     = total_bytes
        self.available_bytes = available_bytes
        self.used_bytes      = used_bytes
        self.percent_used    = percent_used
        self.total_gb        = self.total_bytes     / (1024 ** 3)
        self.available_gb    = self.available_bytes / (1024 ** 3)

        budget_ceiling = self.available_bytes * self._USAGE_CEILING
        budget_reserve = max(0, self.available_bytes - self._OS_RESERVE_GB * (1024 ** 3))
        self.budget_bytes = int(min(budget_ceiling, budget_reserve))
        self.budget_gb    = self.budget_bytes / (1024 ** 3)

        self.cpu_count = os.cpu_count() or 4

        # ── GPU ──────────────────────────────────────────────────────────────
        self.gpu_available = _CUDA
        self.refresh_gpu_stats()

    # ── Display ──────────────────────────────────────────────────────────────

    def report(self) -> None:
        bar_len = 30
        filled  = int(bar_len * self.percent_used / 100)
        bar     = "█" * filled + "░" * (bar_len - filled)
        print("\n╔══════════════════════════════════════════════════════╗")
        print("║              SYSTEM MEMORY MANAGER                  ║")
        print("╠══════════════════════════════════════════════════════╣")
        print(f"║  Total RAM    : {self.total_gb:>6.2f} GB                          ║")
        print(f"║  Used         : {self.used_bytes/1024**3:>6.2f} GB  [{bar}]      ║")
        print(f"║  Available    : {self.available_gb:>6.2f} GB                          ║")
        print(f"║  Budget       : {self.budget_gb:>6.2f} GB (usable)                ║")
        print(f"║  CPU cores    : {self.cpu_count:>6}                                ║")
        if self.gpu_available:
            print(f"║  GPU          : {self.gpu_name:<36} ║")
            print(f"║  GPU VRAM     : {self.gpu_total_gb:>5.1f} GB total / {self.gpu_free_gb:>5.1f} GB free       ║")
        else:
            print( "║  GPU          : Not available                        ║")
        print("╠══════════════════════════════════════════════════════╣")
        print(f"║  process_pool_size   = {self.process_pool_size():<4}                        ║")
        print(f"║  xgb_n_jobs          = {self.xgb_n_jobs():<4}                        ║")
        print(f"║  rl_n_envs           = {self.rl_n_envs():<4}                        ║")
        print(f"║  rl_total_timesteps  = {self.rl_total_timesteps():<8}                    ║")
        print("╚══════════════════════════════════════════════════════╝\n")

    def refresh_gpu_stats(self) -> None:
        """Re-read GPU memory (call each cycle / before sizing pools)."""
        if not _CUDA:
            self.gpu_total_gb = 0.0
            self.gpu_free_gb = 0.0
            self.gpu_name = "None"
            return
        try:
            _torch.cuda.synchronize()
        except Exception:
            pass
        props = _torch.cuda.get_device_properties(0)
        self.gpu_total_gb = props.total_memory / (1024 ** 3)
        free_b, _total_b = _torch.cuda.mem_get_info(0)
        self.gpu_free_gb = free_b / (1024 ** 3)
        self.gpu_name = props.name

    def rl_gpu_reserve_gb(self) -> float:
        """VRAM headroom for the concurrent RL thread on the same GPU."""
        if not self.gpu_available:
            return 0.0
        return min(2.0, max(0.75, self.gpu_total_gb * 0.18))

    # ── Core helpers ─────────────────────────────────────────────────────────

    def _pow2(self, val: int, lo: int, hi: int) -> int:
        val = max(lo, min(val, hi))
        return 2 ** int(math.log2(max(val, 1)))

    def optimal_batch_size(self, base: int = 64, element_bytes: int = 4096) -> int:
        max_samples  = self.budget_bytes // max(element_bytes, 1)
        scale_factor = self.budget_gb / 4.0
        return self._pow2(int(base * scale_factor), 16, min(max_samples, 16384))

    def optimal_workers(self, max_workers: int = 32) -> int:
        mem_per_worker_gb = 0.15
        mem_limited = max(1, int(self.available_gb / mem_per_worker_gb))
        return min(self.cpu_count, mem_limited, max_workers)

    def optimal_chunk_size(self, row_bytes: int = 200, target_chunks: int = 10) -> int:
        max_rows = self.budget_bytes // max(row_bytes, 1)
        return max(1000, max_rows // target_chunks)

    # ── Process pool (TRUE parallelism — replaces ThreadPoolExecutor) ─────────

    def process_pool_size(self, df_ram_gb: float = 0.0) -> int:
        """
        Max parallel XGBoost workers from CPU and RAM.
        With spawn, each worker holds a copy of the dataframe in RAM.
        """
        mem_per_process = max(0.4, 0.25 + df_ram_gb + 0.1)
        ram_limited = max(1, int(self.budget_gb / mem_per_process))
        return max(1, min(self.cpu_count, ram_limited, 32))

    # ── XGBoost ──────────────────────────────────────────────────────────────

    def xgb_n_jobs(self) -> int:
        return self.cpu_count

    def xgb_n_jobs_parallel(self, n_parallel: int = 1) -> int:
        """Divide XGBoost threads across parallel processes."""
        return max(1, self.cpu_count // max(1, n_parallel))

    # ── Reinforcement Learning (PPO) ─────────────────────────────────────────

    def rl_n_envs(self, base: int = 8) -> int:
        """
        Number of parallel environments for VecEnv.
        More envs → richer experience per rollout → better sample efficiency.
        Each env copy is ~50–150 MB.
        """
        mem_per_env_gb = 0.15
        mem_limited    = max(1, int(self.available_gb / mem_per_env_gb))
        cpu_limited    = max(1, self.cpu_count // 2)
        return min(base, mem_limited, cpu_limited)

    def rl_total_timesteps(self, base: int = 80_000) -> int:
        scale = max(0.5, min(self.budget_gb / 4.0, 5.0))
        return int(base * scale)

    def rl_batch_size(self, base: int = 256) -> int:
        return self._pow2(int(base * self.budget_gb / 4.0), 64, 4096)

    def rl_n_steps(self, base: int = 1024) -> int:
        scale = max(0.5, min(self.budget_gb / 4.0, 6.0))
        val   = int(base * scale)
        return max(64, (val // 64) * 64)

    def __repr__(self) -> str:
        return (f"MemoryManager(total={self.total_gb:.1f}GB, "
                f"available={self.available_gb:.1f}GB, "
                f"budget={self.budget_gb:.1f}GB, "
                f"cpus={self.cpu_count})")

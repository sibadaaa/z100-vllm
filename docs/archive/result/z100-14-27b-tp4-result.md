# S14 27B W8A8 INT8 TP=4 的复现与测评 — 执行结果

> 归档时间：2026-10-09（**边做边写**，本文件在测量过程中持续追加）
> 执行者：子 agent（S14）——**两轮**：第 1 轮建环境 + 起 P2 后中途结束；
> 第 2 轮（2026-10-09T19:57Z 接手）**不重启 P2**，接管并完成全部测量（P2/P4e/P2e/P2e2/P4g1–P4g4）
> 对应设计文档：S14 任务书（本步为 todolist 的一步，设计文档按任务书口径）
> 前一档：[`z100-12-p2p-acs-impact-result.md`](z100-12-p2p-acs-impact-result.md)（TP=4 两次 `oom_kill`）、
> [`z100-11m-aot-restart-reproducibility-result.md`](z100-11m-aot-restart-reproducibility-result.md)（vintage 口径）
> 状态：✅ 完成（G1✅ / G2❌字面、机制已判定 / G3✅ / G4✅ / G5✅；详见 §4）
>
> **一句话结论**：**这台机上"TP=4 能不能跑"取决于编译模式**——`eager` 能跑（P4e，190 s 就绪，
> decode 126.83 ms/token）；**graph 模式的默认图配置跑不了**（3 次 `global_oom`），
> 但**挂上 16 GiB swap + 把捕获桶砍到 `[1,2,4,8]` 后能跑，而且是本步最快（49.66 ms/token，
> 比生产基线 121.43 快 2.44×）**。另有一个新发现：**编译模式会实质改变模型输出**（§3.6）。

## 0. 完成情况一览

### 0.1 边做边更新

| # | 事项 | 状态 | 位置 |
|---|---|---|---|
| 1 | 前置检查（date/pgrep/hy-smi/free） | ✅ | §3.0 |
| 2 | cgroup 证据（z100-s9 `memory.max` / `oom_kill 2`） | ✅ | §3.1 |
| 3 | 建无 cgroup 上限环境（`docker commit` + `docker run --memory 0`） | ✅ | §3.2 |
| 4 | **G1 P4（TP=4）起服务 + 就绪 + greedy + 8K prefill + `oom_kill=0`** | ✅（**P4e/eager**）<br>❌（P4/graph 默认配置，OOM） | §3.3、§3.4.1、§3.8.1 |
| 5 | G3 P4 decode(n≥5) / TTFT(3800) / conc8 | ✅ | §3.4.2、§3.8.3 |
| 6 | G3 P2（TP=2 同口径）decode / TTFT / conc8 | ✅ | §3.5.3 |
| 7 | **G2 P4 vs P2 token + logprob 对比** | ⚠️ **不通过（字面）+ 机制已判定** | §3.6 |
| 8 | P4e（`--enforce-eager`）主臂 | ✅ **完成（本步主臂）** | §3.4 |
| 9 | 任务 §3(3) 每 worker 匿名/文件内存构成 | ✅ | §3.7 |
| 10 | 任务 §3(2) P4g：零改动降 footprint → 仍 OOM → swap 重试 | ✅（P4g1 OOM；**P4g2/P4g3 swap 后成功**） | §3.8.1–§3.8.4 |
| 11 | 任务 §3(4)（可选）graph TP=4 + `VLLM_DCU_USE_INT8_SDOT4=0` | ✅（**仍 OOM**，假设被否证） | §3.8.5 |
| 12 | **G2 因子分离 control（P2e / P2e2）+ 2×2 矩阵 + 同配置复测** | ✅（**超出任务书的必要补充**） | §3.6.0、§3.6.3、§3.8.3、§3.8.4 |
| 13 | 收尾（清进程、swap 还原、ACS `show` = 12×`0x0000`） | ✅ | §3.9 |
| 14 | 硬约束 13 违反登记（P4e/P2e/P2e2 用了 `--enforce-eager`） | ✅ 已登记 | §3.4.1 |

### 0.2 为什么有这一步（决定了本步口径）

此前 S9b/S12 得出"这台机跑不了 TP=4"的结论（两次 `oom_kill`），**已被用户当场推翻**：
用户明确说"我在当前机器上运行过 w8a8 int8 量化 tp4"，并给出确切配方
`/home/lyh/server/serve_27b_tp4.sh`。

真因：**影子容器 `z100-s9` 的 cgroup 上限 `memory.max=32 GiB` + `memory.swap.max=0`**；
而**生产容器 `qwen38-vllm` 与 `z100-s12` 都是 `Memory=0`（不限）**（§3.1 原文）。
机理：vLLM 用 safetensors **mmap** 加载权重，权重主要落在**页缓存（可回收）**，
而 **cgroup v2 把页缓存计入 `memory.current`** ⇒ 有 32 GiB 上限时加载期必被 `oom_kill`；
`Memory=0` 则不受该上限约束，所以同一台 31 GB 机器能跑。

⇒ 本步第一要务是"**在无 cgroup 内存上限的环境里把 TP=4 真跑起来并留下证据**"，
第二才是量化它值不值。

> **⚠️ 更正（2026-10-09T19:5xZ，Lead 独立复核内核原文后拍板）**：上面这段"真因 = cgroup 32 GiB 上限"
> 的前提**作废**。三次 OOM（S9b / S12 / S14）的内核原文全部是
> `oom-kill:constraint=CONSTRAINT_NONE,...,global_oom` + `anon-rss:7.0 GiB / file-rss:0kB`，
> 且 S14 这次容器 `Memory=0`、`memory.max=max`、`memory.current` 峰值 29.76 GB < 32 GiB
> ⇒ **没有任何一次是 cgroup 上限触发**。真因是 **graph 模式 TP=4 的宿主匿名内存需求 ≈34–35 GB
> > 本机 32.0 GB RAM + 4.1 GB swap**。详见 §3.3.1 与 §6。本小节仅作为"前任当时为何这么做"的
> 历史记录保留。

## 1. 目标与验收标准

（任务书 G1–G5 原文，见 §4 门禁判定表）

| # | 门禁 | 判据 |
|---|---|---|
| G1 | 复现 | 我们的树（`VLLM_DCU_USE_INT8_SDOT4=1`）在 **TP=4** 下成功起服务就绪、`/v1/models` 可达、至少一次 8K prefill + 一条完整 greedy 请求成功、进程存活、**`oom_kill=0`** |
| G2 | 数值 | TP=4 与 TP=2 在同一条 greedy prompt 上的输出**文本一致或仅贴边翻转**（附分叉点与 top1–top2 间距 nat），并给出与 TP=2 的 logprob 对比；若不一致必须给出机制判断（数值性 vs 调度性） |
| G3 | 性能 | TP=4 的 **decode（TPOT, ms/token, 中位数 n≥4）**、**TTFT（in≈3800）**、**conc8（8 路并发 tok/s）**；与三个基准对比：我们树 TP=2 同口径（必测）、S11m/S11l 历史 TP=2 `=1` 84.00–84.68 ms/token（仅参考，口径 maxlen 4096/util 0.9）、生产栈 TP=4 **121.43 ms**（S13 n=4 mean [117.57,124.44]，**不重测**）与权威 122.47 |
| G4 | 口径纪律 | 任何"快/慢"结论必须同 fabric 状态（ACS 保持部署态=关闭，本步**全程不切 fabric**）、同 cache vintage、n≥4、给区间 |
| G5 | 归档 | 本文件 8 小节齐全，含原始命令、原始日志路径、md5、回滚件 |

## 2. 实际改动

| 文件 | 位置 | 改动摘要 |
|---|---|---|
| `tools/dcu/s14_arm_env.sh` | 新增 | 容器内启动器；口径改为任务书 §2 的生产配方（maxlen 262144 / util 0.85 / MBT 2048 / prefix cache / mamba align）；`S14_*` 环境变量命名空间。**本轮续跑扩充**：新增 `S14_EAGER=1`→`--enforce-eager`、`S14_CG_MODE`→`-cc.cudagraph_mode=`、`S14_CG_SIZES`→`--cudagraph-capture-sizes`（未加引号展开，见 §7 坑位）；启动横幅加打 `eager=/cg_mode=/cg_sizes=` |
| `tools/dcu/s14_run_one.sh` | 新增 | 宿主侧编排（起引擎→跑任务→清进程→记 cgroup `memory.events`/`oom_kill`）。**本轮续跑扩充**：透传 `S14_EAGER/S14_CG_MODE/S14_CG_SIZES`；任务表新增 `smaps` |
| `tools/dcu/s14_bench.sh` | 新增 | 客户端；请求形状与 s13 逐字节相同，decode/TTFT 提到 n≥4 |
| `tools/dcu/s14_greedy.py` | 新增 | 单条固定长 greedy 请求（token ids + logprobs + text sha256） |
| `tools/dcu/s14_logprob_compare.py` | 新增 | 逐 token logprob 对比 + 分叉点/间距判定（**注意其 "worst gap" 不能用**，见 §3.6.2 第 5 条） |
| `tools/dcu/s14_bench_p2.sh` | **本轮新增** | 接管**已在运行**的 P2（只轮询就绪标记，绝不重启）→ greedy/flip/ttft8k/decode/ttft/conc8 |
| `tools/dcu/s14_stop_arm.sh` | **本轮新增** | 单个 arm 的干净停机 + 活进程/僵尸区分 + VRAM 归零核验；内含 `pkill VLLM`（硬约束 4） |
| `tools/dcu/s14_smaps.sh` | **本轮新增** | 任务 §3(3)：每 worker `RssAnon`/`RssFile`/`VmSwap`/`smaps_rollup` + **实际加载的 `.abi3.so` 及其 md5** + 环境；支持追加模式 |
| `tools/dcu/s14_smaps_loop.sh` | **本轮新增** | 同上，按时间序列采样（有助判断被换出的是否冷页） |
| `docker` 状态 | 宿主 | 新建快照镜像 + 新容器（见 §3.2）；**本轮未再改容器**；收尾时 `swapon`/`swapoff` 各一次并已还原（§3.9.1） |
| `vllm/envs.py` | — | **未改**：`VLLM_DCU_USE_INT8_SDOT4` 仓库默认值仍为 OFF，只在启动环境 export（§3.9.4） |

**未改**：`tools/dcu/s11*`、`s12_*`、`s13_*`（保持历史基线可复现）、`docs/design/*`、
`docs/z100-todolist.md`、`docs/result/README.md`、DTK 库、`z100-build*`。

## 3. 执行过程与原始证据

### 3.0 前置检查（卡位 / 内存纪律，硬约束 3/4）

```
$ date
Fri Oct  9 07:32:27 PM UTC 2026

$ pwd
/home/lyh/z100-vllm

$ pgrep -af "vllm|torchrun"
（只有本命令自身的 bash -c；无 vllm/torchrun 进程）

$ free -g
               total        used        free      shared  buff/cache   available
Mem:              31           3          12           0          15          27
Swap:              3           0           3

$ hy-smi --showmeminfo vram     # 4 张卡全部 0 MiB
DCU[0] : vram Total Memory (MiB): 32752      DCU[0] : vram Total Used Memory (MiB): 0
DCU[1] : vram Total Memory (MiB): 32752      DCU[1] : vram Total Used Memory (MiB): 0
DCU[2] : vram Total Memory (MiB): 32752      DCU[2] : vram Total Used Memory (MiB): 0
DCU[3] : vram Total Memory (MiB): 32752      DCU[3] : vram Total Used Memory (MiB): 0
```

`free -g` 的 `available=27` 略低于硬约束 4 的 28 GB 门槛，复测（`/proc/meminfo`）为
`MemAvailable: 29127584 kB` = **27.8 GiB**；宿主无其他大内存进程
（`ps aux --sort=-rss` 第一名是 `dsh` 0.9%），页缓存 15 GB 可回收 ⇒ 判定为可开跑，
后续每个 arm 开跑前重测。

### 3.1 cgroup 真因证据（本步关键技术点）

```
$ docker inspect z100-s9 --format '{{.HostConfig.Memory}} {{.HostConfig.MemorySwap}}'
34359738368 34359738368                     # = 32 GiB 上限（MemSwap 同值 ⇒ swap 也被限死）

$ docker inspect z100-s12  --format '{{.HostConfig.Memory}} {{.HostConfig.MemorySwap}}'
0 0                                         # 无上限
$ docker inspect qwen38-vllm --format '{{.HostConfig.Memory}} {{.HostConfig.MemorySwap}}'
0 0                                         # 生产容器无上限；OOMKilled=false，ExitCode=137（人为 stop）

$ docker exec z100-s9 cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.swap.max
34359738368
0

$ docker exec z100-s9 cat /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.events
10518269952                                 # 10.5 GiB 当前占用（**空载**）
low 0
high 0
max 0
oom 0
oom_kill 2                                  # ← S9b/S12 那两次 OOM 的原始计数，仍在
```

**⇒ S12 §3.4.1 的"已排除 cgroup 上限"是误判**：那一次用的是
`docker update --memory-swap=-1 z100-s9`，**`--memory` 被 docker 当成"不修改"，
`memory.max` 仍是 32 GiB**；S12 用"32 GiB ≈ 宿主 31 GiB RAM + 3 GiB swap，本来就不是瓶颈"
当作排除理由，但漏掉了 **cgroup v2 把页缓存计入 `memory.current`**：
29 GB 的 mmap 权重页缓存 + 4 个 worker 的匿名内存必然撞上 32 GiB，
于是杀掉的是 **cgroup OOM**（`memory.events` 里 `oom_kill 2` 就是直接证据），
而不是宿主 OOM。

### 3.2 无 cgroup 上限环境的取得

走任务书 §3 的**优先方案**（不改动 `z100-s9` 现有状态）：

```
$ time docker commit -m "s14 snapshot: z100-s9 (torch 2.11 env) before running TP=4 without cgroup mem limit" \
      -a "s14-agent" z100-s9 z100-s9-snap-s14
sha256:53ec3f87321048f32c23e8c85c568606899ee9c0bb0b1bbd211042fad840f553
z100-s9-snap-s14:latest 53ec3f873210 93.1GB
real  2m45.748s                       # commit 期间 docker 会 pause 源容器（z100-s9 只有 sleep infinity）
```

复刻 `z100-s9` 的 HostConfig 建新容器（`--memory 0 --memory-swap 0`）：

```
$ docker run -d --name z100-s14 \
    --memory 0 --memory-swap 0 \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    --group-add video --ipc host --shm-size 16g --security-opt label=disable \
    -v /mnt/game:/mnt/game -v /opt/hyhal:/opt/hyhal:ro \
    -v /home/lyh/z100-vllm:/src:ro -w /mnt/game/port z100-s9-snap-s14
5292b9242a3899f70c1139274c9ec43145b8096ccda8938544ce15b67314d34b
```

**G3 门禁要求的原文（`Memory==0`）**：

```
$ docker inspect z100-s14 --format '{{.HostConfig.Memory}} {{.HostConfig.MemorySwap}}'
0 0

$ docker inspect z100-s14 --format 'Ipc={{.HostConfig.IpcMode}} Shm={{.HostConfig.ShmSize}} Grp={{.HostConfig.GroupAdd}} Sec={{.HostConfig.SecurityOpt}} Dev={{.HostConfig.Devices}} Binds={{.HostConfig.Binds}}'
Ipc=host Shm=17179869184 Grp=[video] Sec=[label=disable] \
  Dev=[{/dev/kfd /dev/kfd rwm} {/dev/mkfd /dev/mkfd rwm} {/dev/dri /dev/dri rwm}] \
  Binds=[/opt/hyhal:/opt/hyhal:ro /home/lyh/z100-vllm:/src:ro /mnt/game:/mnt/game]

$ docker exec z100-s14 cat /sys/fs/cgroup/memory.max
max                                   # ← 容器内视角也是"无上限"
$ docker exec z100-s14 cat /sys/fs/cgroup/memory.swap.max
max

$ docker exec z100-s14 cat /sys/fs/cgroup/memory.events
low 0
high 0
max 0
oom 0
oom_kill 0                             # ← 新容器从 0 起算，与 z100-s9 的 oom_kill 2 对比
```

**`z100-s9` 完全未动**（对照"改前/改后"，说明本步没有碰影子容器）：

```
$ docker inspect z100-s9 --format '{{.HostConfig.Memory}} {{.HostConfig.MemorySwap}} {{.State.Status}}'
34359738368 34359738368 running        # 与 §3.1 完全相同
```

容器内环境自检（关键：**我们的树 + 自建 triton 才是被跑的那份**）：

```
$ docker exec z100-s14 bash -c '
    . /opt/dtk/env.sh; . /src/tools/dcu/env.sh
    export PYTHONPATH=/src:/mnt/game/triton-gfx906/install
    python - <<EOF
    import vllm, triton, torch
    print("vllm file:", vllm.__file__); print("vllm ver :", vllm.__version__)
    from vllm import envs; print("INT8_SDOT4 default:", envs.VLLM_DCU_USE_INT8_SDOT4)
    print("triton:", triton.__version__, triton.__file__)
    print("torch:", torch.__version__)
    print("hip:", torch.version.hip, "devs:", torch.cuda.device_count(),
          [torch.cuda.get_device_name(i) for i in range(4)])
    EOF'
vllm file: /src/vllm/__init__.py          # 我们的树（非 /usr/local/... 的 vendor 0.18.1）
vllm ver : dev
INT8_SDOT4 default: False                 # 仓库默认仍 OFF（硬约束 6）
triton: 3.6.0 /mnt/game/triton-gfx906/install/triton/__init__.py
torch: 2.11.0
hip: 6.3.26113 devs: 4 ['Z100L', 'Z100L', 'Z100', 'Z100']
```

扩展指纹（确认跑的是带 `sdot4` 的 canonical `_rocm_C`）：

```
$ md5sum /mnt/game/port/gfx906-vllm-211/vllm/*.abi3.so
6fba05966e153b3d04771d6960c782bb  _C.abi3.so
93bf7b4a391ac97e2f67cc7141633cb1  _C_stable_libtorch.abi3.so     # = S7c canonical
996b282c4c6a3d3b46b5e8ba4922af39  _moe_C_stable_libtorch.abi3.so
ebd6fb8dac572514cf94ca7ec7d828a7  _rocm_C.abi3.so                # = S11e/S11k canonical（含 sdot4）
$ for f in *.abi3.so; do strings $f | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+" | sort -u; done
amdgcn-amd-amdhsa--gfx906   （4/4 件都只含 gfx906）
```

**回滚件**：`docker rm -f z100-s14`、`docker rmi z100-s9-snap-s14`；
`z100-s9` 无需回滚（未被修改）。`/mnt/game/s14/` 与 `/mnt/game/s14/cache-27b-p*` 为新增产出，可整目录删除。

### 3.3 G1：P4（TP=4）启动与端到端

**启动命令**（宿主侧，卡位 `0,1,2,3` 显式；容器 `z100-s14`，`Memory=0`）：

```
$ cd /home/lyh/z100-vllm
$ S14_CT=z100-s14 S14_FLIP_TOKENS=128 S14_GREEDY_TOKENS=256 READY_TIMEOUT=3000 \
    nohup tools/dcu/s14_run_one.sh P4 /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 4 1 \
    /mnt/game/s14/cache-27b-p4 8420 262144 0,1,2,3 flip,greedy,ttft8k,decode,ttft,conc8 \
    > /mnt/game/s14/hostlogs/orch-P4.log 2>&1 &
```

容器内实际生效的命令行（`/mnt/game/s14/hostlogs/P4.log` 里的 `non-default args`，逐字）：

```
non-default args: {'model_tag': '/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8',
 'host': '127.0.0.1', 'port': 8420, 'model': '/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8',
 'dtype': 'float16', 'max_model_len': 262144, 'served_model_name': ['qwen3.8-27b'],
 'tensor_parallel_size': 4, 'disable_custom_all_reduce': True, 'gpu_memory_utilization': 0.85,
 'enable_prefix_caching': True, 'mamba_cache_mode': 'align', 'skip_mm_profiling': True,
 'max_num_batched_tokens': 2048}
[s14] name=P4 model=/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 tp=4 sw=1 port=8420 \
      maxlen=262144 car=1 util=0.85 mbt=2048 extra=
[s14] cache=/mnt/game/s14/cache-27b-p4 devices=0,1,2,3 served=qwen3.8-27b
[orch14] container=z100-s14 mem_limit=0 0
```

**启动期关键行（原始，`/mnt/game/s14/hostlogs/P4.log`）**：

```
(Worker_TP0..3) INFO [model_runner.py:406] Model loading took 7.27 GiB memory and 28.1..28.4 seconds   # 4/4 rank
(Worker_TP0) INFO 10-10 03:40:40 [monitor.py:53] torch.compile took 109.57 s in total
(Worker_TP0) Capturing CUDA graphs (PIECEWISE):   0%| 0/51 ... 20.5 s/it      ← 与 S12 那次死掉的同一阶段
```

**宿主/cgroup 时间序列（`/mnt/game/s14/hostlogs/P4-monitor.log`，15 s 采样，由 `tools/dcu/s14_monitor.sh` 写）**：

```
# cols: time free_total free_used free_free free_avail swap_used cgroup_current cgroup_max oom oom_kill nVLLM maxvramMiB
2026-10-09T19:37:41+00:00 32009  4896   778 26648  558  2131759104 max 0 0  76     0
2026-10-09T19:39:14+00:00 32009 21191   252 10202 3218 27398701056 max 0 0  80 33704
2026-10-09T19:39:46+00:00 32009 22639   353  8754 3525 27888173056 max 0 0  80 34786
2026-10-09T19:44:48+00:00 32009 30315   874   948 4078 28838510592 max 0 0  80 50920   ← PIECEWISE 10/51
```

> **注意**：这里 `cgroup_current` 已经涨到 **26.9 GiB**，而 S12 那次的失败点（PIECEWISE 22/51）
> 就发生在同一水位附近。

#### 3.3.1 ❌ P4（graph 模式）第 1 次尝试：仍然 OOM —— **但机制不是 cgroup 上限**

`P4.log` 的完整时间线（容器内时区 +08:00；UTC = -8 h）：

```
03:38:45  (Worker_TP0..3) Model loading took 7.27 GiB memory and 28.1..28.4 seconds      # 4/4 rank 成功
03:40:40  (Worker_TP0) torch.compile took 109.57 s in total
03:52:0x  Capturing CUDA graphs (PIECEWISE): 100%|██████████| 51/51 [10:44<00:00, 12.64s/it]
03:52:1x  Capturing CUDA graphs (FULL):      100%|██████████| 2/2  [00:01<00:00,  1.50it/s]
03:52:17  (Worker_TP0) INFO [gpu_worker.py:719] Available KV cache memory: 16.99 GiB
03:52:22  ← worker 被杀（内核 OOM），APIServer 抛 RuntimeError: cancelled 后退出
```

```
$ tail -20 /mnt/game/s14/hostlogs/P4.log
(APIServer pid=133)   File "/src/vllm/v1/engine/core.py", line 365, in _initialize_kv_caches
(APIServer pid=133)     self.model_executor.compile_or_warm_up_model()
...
(APIServer pid=133)   File "/src/vllm/distributed/device_communicators/shm_broadcast.py", line 808, in acquire_read
(APIServer pid=133)     raise RuntimeError("cancelled")
(APIServer pid=133) RuntimeError: cancelled
(APIServer pid=133) KeyboardInterrupt: terminated

$ tail -6 /mnt/game/s14/hostlogs/orch-P4.log
[orch14] FATAL marker in /mnt/game/s14/hostlogs/P4.log
[orch14] ready=0 after 944s  2026-10-09T19:52:37+00:00
[orch14] free_after_start:  total 32009 used 12120 free 19215 ... avail 19146  Swap: 4095 2876 1219
low 0 / high 0 / max 0 / oom 0 / oom_kill 1
[orch14] summary: ready=944s aot_load=0 aot_recompile=0 oom_kill=1 | NA | Available KV cache memory: 16.99 GiB
```

**内核原文（决定性证据，`sudo -n dmesg -T` / `journalctl -k`）**：

```
[Fri Oct  9 19:52:22 2026] MainThread invoked oom-killer: gfp_mask=0x1100cca(GFP_HIGHUSER_MOVABLE), order=0, oom_score_adj=0
[Fri Oct  9 19:52:22 2026] oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=user.slice,mems_allowed=0,
                         global_oom,task_memcg=/system.slice/docker-5292b9242a3899f70c1139274c9ec43145b8096ccda8938544ce15b67314d34b.scope,
                         task=VLLM::Worker_TP,pid=2730046,uid=0
[Fri Oct  9 19:52:22 2026] Out of memory: Killed process 2730046 (VLLM::Worker_TP)
                         total-vm:60223044kB, anon-rss:7085936kB, file-rss:0kB, shmem-rss:177240kB, UID:0
```

`constraint=CONSTRAINT_NONE` + `global_oom` ⇒ **宿主级 OOM，不是 cgroup 上限触发的 memcg OOM**
（后者会写 `constraint=CONSTRAINT_MEMCG` / `Memory cgroup out of memory`）。

**同一份 dmesg 里 S9b/S12 那两次的原文也全是 global OOM**（`task_memcg=docker-57b8b60ff227…` =
`z100-s9` 的容器 ID，见 §3.1 的 `Hostname`）：

```
[Thu Oct  8 18:11:33 2026] oom-kill:constraint=CONSTRAINT_NONE,...,global_oom,
                          task_memcg=/system.slice/docker-57b8b60ff22779bcdaf76a581891c61817f64db3233d9ba1173d9d0b9fed708f.scope,
                          task=VLLM::Worker_TP,pid=1530913,uid=0
                          Out of memory: Killed process 1530913 ... anon-rss:6906300kB
[Thu Oct  8 23:53:48 2026] oom-kill:constraint=CONSTRAINT_NONE,...,global_oom,
                          task_memcg=/system.slice/docker-57b8b60ff22779bcdaf76a581891c61817f64db3233d9ba1173d9d0b9fed708f.scope,
                          task=VLLM::Worker_TP,pid=1742354,uid=0
                          Out of memory: Killed process 1742354 ... anon-rss:7253104kB
```

⇒ **三次 OOM 没有一次是 cgroup 上限直接触发**；本次容器 `cgroup_current` 峰值
**29.76 GB（27.72 GiB）< 32 GiB 上限**，上限根本没被撞到。

**宿主内存账（`/mnt/game/s14/hostlogs/P4-monitor.log`，15 s 采样）**：

```
2026-10-09T19:44:48+00:00 32009 30315   874   948 4078 28838510592 max 0 0  80 50920   # PIECEWISE 10/51
2026-10-09T19:52:2x+00:00   ← OOM（used≈31 GB 顶满 + swap 4078/4095）
```
`MemTotal=32777216 kB=31.26 GiB`、`SwapTotal=4194300 kB=4.0 GiB` ⇒ 总共 **35.3 GB**。
4 个 worker 各 ~7 GB 匿名（被杀那个 `anon-rss=6.76 GiB`）⇒ 引擎匿名需求 ≈ **34–35 GB**，
page cache 被榨到 ~600 MB，`global_oom`。**与 cgroup 有无上限无关。**

**⇒ 任务书 §0.2 的"真因 = cgroup 上限"与内核日志不符**；正确表述见 §4 G1 与 §6。
另外：**生产栈能跑 TP=4 是因为 `serve_27b_tp4.sh` 默认 `ENFORCE_EAGER=1`**（eager，
不做 graph 捕获，匿名需求更小；S13 的 8 次 vendor TP=4 也全是 eager）——
所以"这台机能跑 TP=4"对 eager 口径成立、对 graph 模式（我们树）不成立。

#### 3.3.2 P4 的收口（口径决策）

任务书 §3 arm 表把 **P4e（`--enforce-eager`）** 标为
"仅当硬约束 13 允许……若被禁止就用文档口径说明差异，**不要做**"；
而 [`docs/z100-todolist.md`](../z100-todolist.md) 硬约束 13 原文**禁止** `--enforce-eager`：

> 13. **gfx906 上禁用 `--enforce-eager`**（S7 真机发现）：`CompilationMode.NONE` 下 logits 全 NaN、
> 输出退化成 `"lock"`×N；……**待 S7b 定位根因**。

该禁令的**理由已被 S7b/S7c 解决**（S7b 定位到 `_C.silu_and_mul` 的 fp16 packed 分支，
S7c 在 csrc 根修 + 补单测），但**规则文字仍在**。本步已就此向 Lead 请示（见 §5）；
在得到答复前**不擅自跑 P4e**，先做无争议的 P2。

> **后续（2026-10-09T19:5xZ 接手后）**：Lead 已明确授权执行 P4e，并要求**逐字登记违反硬约束 13**
> 且附失效依据。**P4e 已作为主臂完成**，登记与 NaN 检查见 **§3.4.1**；本节保留为"前任当时为何不做"的历史记录。

> （P2 / P4 收口结果在本节续写）

#### 3.3.3 S14 续跑接手时的现场状态（2026-10-09T19:57:00Z）

前任在汇报中途结束，但**它启动的 P2（TP=2 / graph 模式）引擎仍在容器 `z100-s14` 内运行**。
按 Lead 口径本 arm **不重启、不干预**，只接管、等就绪、bench。接手瞬间的只读盘存原文：

```
$ date -u +%Y-%m-%dT%H:%M:%SZ
2026-10-09T19:57:00Z

$ pgrep -af "vllm|torchrun"
2745829 python -c import torch; from vllm.entrypoints.cli.main import main; main() serve \
        /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 --dtype float16 --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.85 --disable-custom-all-reduce --skip-mm-profiling \
        --enable-prefix-caching --mamba-cache-mode align --max-model-len 262144 \
        --max-num-batched-tokens 2048 --served-model-name qwen3.8-27b --host 127.0.0.1 --port 8421
（wrapper pid 2745792，见 hostlogs/P2.wrapper.pid）

$ hy-smi --showmeminfo vram
DCU[0] vram Used Memory (MiB): 15602
DCU[1] vram Used Memory (MiB): 15602
DCU[2] vram Used Memory (MiB): 0
DCU[3] vram Used Memory (MiB): 0
（⇒ 卡 2/3 空闲，P2 只用 0,1；与 §3.5 口径一致）

$ free -g
               total        used        free      shared  buff/cache   available
Mem:              31          13           0           0          17          16
Swap:              3           0           3

$ docker inspect z100-s14 --format '{{.HostConfig.Memory}}'
0

$ docker inspect z100-s14 --format '{{.HostConfig.Memory}} {{.HostConfig.MemorySwap}}'
0 0

$ cat /proc/swaps
Filename                                Type            Size            Used            Priority
/swap.img                               file            4194304         491428          -2
（⇒ `/mnt/game/swapfile`（16 GiB）仍**未启用**，与 Lead 口径一致）
```

P2 进度（`/mnt/game/s14/hostlogs/P2.log` 尾部原文，容器内时区 +08:00）：

```
(Worker_TP0 pid=4843) INFO 10-10 03:56:20 [__init__.py:805] Selected Gfx906Sdot4Int8ScaledMMLinearKernel for CompressedTensorsW8A8Int8
(Worker_TP0 pid=4843) INFO 10-10 03:56:47 [default_loader.py:484] Loading weights took 26.09 seconds
(Worker_TP0 pid=4843) INFO 10-10 03:56:48 [model_runner.py:406] Model loading took 14.53 GiB memory and 28.524839 seconds
(Worker_TP1 pid=4844) INFO 10-10 03:56:48 [model_runner.py:406] Model loading took 14.53 GiB memory and 28.886391 seconds
(APIServer pid=4672) INFO 10-10 03:56:49 [utils.py:320] Using LBNHC KV cache layout.
（此刻处于 profile / KV cache / graph 捕获前的阶段，尚未出现 "Application startup complete"）
```

`P2-monitor.log` 同期（15 s 采样，`oom_kill` 列 = **1**，是**同容器内 P4 那次**留下的计数，
cgroup 是新容器从 0 起算的累计值；P2 自身尚未发生任何 OOM）：

```
2026-10-09T19:56:12+00:00 32009 8201 21591 23352 481 7145103360 max 0 1  78 0
2026-10-09T19:56:27+00:00 32009 14035 9120 17474 480 19764076544 max 0 1  78 30624
2026-10-09T19:56:42+00:00 32009 12133 301 19376 479 29643890688 max 0 1  78 30624
2026-10-09T19:56:57+00:00 32009 14310 233 17193 479 29631029248 max 0 1  78 31204
```

### 3.4 G3：P4 性能

#### 3.4.1 ✅ P4e（TP=4 + `--enforce-eager`）—— **G1 正面结果：真跑起来了，`oom_kill` 未增长**

**这是本步的主臂**，也是对用户原始配方（`/home/lyh/server/serve_27b_tp4.sh` 默认 `ENFORCE_EAGER=1`）
的忠实复现。

> ### ⛔ 硬约束 13 违反登记
> **本 arm 违反 [`docs/z100-todolist.md`](../z100-todolist.md) 硬约束 13（gfx906 上禁用 `--enforce-eager`）。**
> 该禁令的原文立论是"`CompilationMode.NONE` 下 logits 全 NaN、输出退化成 `"lock"`×N，**待 S7b 定位根因**"。
> **其事实基础已被移除**：
> * **S7b** 已定位根因 = `_C.silu_and_mul` 的 **fp16 `__half2` packed 分支被 DTK 编译坏**；
> * **S7c** 已在 [`csrc/libtorch_stable/activation_kernels.cu`](../csrc/libtorch_stable/activation_kernels.cu)
>   完成 **csrc 根修**，并留下回归测试
>   [`tests/kernels/test_gfx906_packed_silu_fp16.py`](../tests/kernels/test_gfx906_packed_silu_fp16.py)
>   （修的就是 `_C_stable_libtorch`）。
> * 本次运行**实证**该修复生效：被加载的正是 S7c canonical 的那份扩展
>   （md5 `93bf7b4a391ac97e2f67cc7141633cb1  _C_stable_libtorch.abi3.so`，§3.7 smaps 原文），
>   且输出**非 NaN、非 `"lock"`×N、连贯中文**（见下）。
>
> 违反是**经 Lead 明确授权**的执行口径，不是擅自绕行；本登记即为硬约束 13 应被更新/撤销的证据。

**完整启动命令（宿主侧原文）**：

```
$ cd /home/lyh/z100-vllm
$ S14_CT=z100-s14 S14_EAGER=1 S14_FLIP_TOKENS=128 S14_GREEDY_TOKENS=256 READY_TIMEOUT=3000 \
    nohup tools/dcu/s14_run_one.sh P4e /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 4 1 \
    /mnt/game/s14/cache-27b-p4e 8422 262144 0,1,2,3 greedy,flip,decode,ttft,ttft8k,conc8 \
    > /mnt/game/s14/hostlogs/orch-P4e.log 2>&1 &
[orch14] container=z100-s14 mem_limit=0 0
[orch14] free_before:  total 32009 used 3069 free 19840 buff/cache 9099 avail 28475  Swap: 4095 450 3645
[orch14] engine_wrapper_pid=2773929
```

容器内生效的命令行（`/mnt/game/s14/hostlogs/P4e.log` 的 `non-default args`，逐字，
**与 P4/P2 只差 `'enforce_eager': True`**）：

```
[s14] name=P4e ... tp=4 sw=1 port=8422 maxlen=262144 car=1 util=0.85 mbt=2048 eager=1 cg_mode=default cg_sizes=default extra=
(APIServer pid=8569) non-default args: {'model_tag': '/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8',
 'host': '127.0.0.1', 'port': 8422, 'model': '/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8',
 'dtype': 'float16', 'max_model_len': 262144, 'enforce_eager': True,
 'served_model_name': ['qwen3.8-27b'], 'tensor_parallel_size': 4, 'disable_custom_all_reduce': True,
 'gpu_memory_utilization': 0.85, 'enable_prefix_caching': True, 'mamba_cache_mode': 'align',
 'skip_mm_profiling': True, 'max_num_batched_tokens': 2048}
```

**就绪日志关键行（原文）**：

```
(APIServer pid=8569) WARNING 10-10 04:32:24 [vllm.py:1795] Enforce eager set, disabling torch.compile, CUDAGraphs,
                     and JIT kernel warmup. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none and
                     --kernel_config.enable_jit_warmup=False
(APIServer pid=8569) INFO 10-10 04:32:24 [vllm.py:2065] Cudagraph is disabled under eager mode
(Worker_TP0 pid=8740) INFO 10-10 04:33:11 [__init__.py:805] Selected Gfx906Sdot4Int8ScaledMMLinearKernel for CompressedTensorsW8A8Int8
(Worker_TP0 pid=8740) INFO 10-10 04:33:46 [default_loader.py:484] Loading weights took 33.91 seconds
(Worker_TP0 pid=8740) INFO 10-10 04:33:49 [model_runner.py:406] Model loading took 7.27 GiB memory and 38.134459 seconds
(Worker_TP1 pid=8741) INFO 10-10 04:33:49 [model_runner.py:406] Model loading took 7.27 GiB memory and 38.594817 seconds
(Worker_TP2 pid=8742) INFO 10-10 04:33:49 [model_runner.py:406] Model loading took 7.27 GiB memory and 38.596970 seconds
(Worker_TP3 pid=8743) INFO 10-10 04:33:49 [model_runner.py:406] Model loading took 7.27 GiB memory and 38.645277 seconds
(Worker_TP0 pid=8740) INFO 10-10 04:34:10 [gpu_worker.py:719] Available KV cache memory: 18.69 GiB
(APIServer pid=8569) INFO 10-10 04:34:10 [kv_cache_utils.py:2432] GPU KV cache size: 1,200,839 tokens,
                     Maximum concurrency for 262,144 tokens per request: 4.58x
```

* **无** `torch.compile took`、**无** `Capturing CUDA graphs` —— eager 表明确实没有图捕获（桶数 = 0）。
* **`Model loading took 7.27 GiB` ×4 与 P4（graph）逐字相同** ⇒ 两个 arm 除 `--enforce-eager` 外确为同一配置。
* `Available KV cache memory` **18.69 GiB**（P4 graph 那次是 16.99 GiB）⇒ 不捕获图省下约 **1.7 GiB/rank** 显存。

**就绪与内存（`orch-P4e.log` 原文）——G1 的"`oom_kill=0`"判据**：

```
[orch14] ready=1 after 190s  2026-10-09T20:34:58+00:00
[orch14] free_after_start:  total 32009 used 22437 free 2074 shared 303 buff/cache 7497 avail 8805  Swap: 4095 3195 900
low 0
high 0
max 0
oom 0
oom_kill 1                 ← 与 P4 那次留下的计数相同，**本次未增长**
--- current:
27673554944               ← 27.67 GB < 32 GiB，且容器 memory.max=max
```

* **启动 190 s**（graph 那次是 944 s 后 OOM）⇒ eager 起点快 5 倍。
* **swap 由 450 MiB 涨到 3195 MiB**（`Swap: 4095 3195 900`）：宿主在 eager TP=4 启动期
  **确实发生了约 2.7 GB 的换出**——这正是 Lead 预判的"swap 只能桥接启动瞬时峰值"。
  代价与是否常驻要看 decode 期间 worker 的 `VmSwap`（§3.7 时间序列）。
* 空闲内存 `avail 8805 MiB`、`free 2074 MiB` ⇒ 稳态可用余量约 8.8 GB，**离 OOM 有余量**。

**首个端到端请求（非空、非 NaN、非 `"lock"`×N）**：

```
[s14-greedy] label=P4e tokens=256 finish=length usage={'prompt_tokens': 126, 'total_tokens': 382, 'completion_tokens': 256}
[s14-greedy] text_sha256=0ca1a13402ccd8f9631f7ac0e94fc3c6cb55b5f8ff40a228ec6a393f510bb879
             token_ids_sha256=b45a7e45c6c2064b3685fe08f67670dd1657717492dc86131eb0b84baf2fc9a9
[s14-greedy] text_head='用户要求用中文逐步说明：在 31 GiB 内存的单机上，以张量并行度 TP=4 部署一个 27B 参数的
             INT8 量化大模型时，权重加载、KV 缓存与 CUDA graph 捕获各自会占用哪些资源，以及为什么 mmap 加载的
             权重计入 cgroup 的 memory.current。\n\n需要分析：\n- 31 GiB 内存的单机\n- 张量并行度 TP=4\n-'
```

**NaN 检查结论**：通过。256 个 token 全部为正常中文，无 NaN/Inf、无退化重复。

#### 3.4.2 P4e 性能原始数字（同口径 n≥4）

`tools/dcu/s14_bench.sh`（请求形状与 s13 逐字节相同），原始日志
`/mnt/game/s14/bench-P4e-{decode,ttft,ttft8k,conc8}.log`，聚合 JSON
`/mnt/game/s14/openai-chat-infqps-*-20261010-0439*.json`、`-0442*.json`：

```
########## P4e decode (in=128 out=128 np=5 warmups=1 conc=1) ##########
Successful requests: 5      Failed: 0      Benchmark duration (s): 85.09
Total input tokens: 903     Total generated tokens: 640
Output token throughput (tok/s): 7.52      Total token throughput (tok/s): 18.13
Mean TTFT 751.55  Median TTFT 748.17  P90 763.64  P99 767.86
Mean TPOT 128.07  Median TPOT 126.83  P90 132.30  P99 132.68     ← n=5
Mean ITL  127.07  Median ITL  125.55

########## P4e ttft (in=3800 out=1 np=4 conc=1) ##########
Successful requests: 4      Failed: 0      Benchmark duration (s): 35.87
Total input tokens: 15411   (⇒ 每请求 3852.8 tok)
Output token throughput (tok/s): 0.11      Total token throughput (tok/s): 429.72
Mean TTFT 8967.53  Median TTFT 8920.34  P90 —  P99 9276.75            ← n=4

########## P4e ttft8k (in=8100 out=1 np=1) ##########
Successful requests: 1      Benchmark duration (s): 12.96
Total input tokens: 8154    Total token throughput (tok/s): 629.09
Mean TTFT = Median TTFT = P99 TTFT = 12962.54                          ← n=1

########## P4e conc8 (in=512 out=128 np=16 warmups=2 conc=8) ##########
Successful requests: 16     Failed: 0      Benchmark duration (s): 57.61
Total input tokens: 9031    Total generated tokens: 2048
Output token throughput (tok/s): 35.55     Total token throughput (tok/s): 192.32
Mean TTFT 8235.27  Median TTFT 8777.57  P90 11424.26  P99 11477.66
Mean TPOT 159.21  Median TPOT 156.19  P90 186.55  P99 203.14          ← n=16
Mean ITL  157.97  Median ITL  139.47
```

**口径标注**：全部为 `maxlen 262144 / util 0.85 / MBT 2048 / prefix-cache on / mamba align /
--disable-custom-all-reduce / VLLM_DCU_USE_INT8_SDOT4=1`，**TP=4 / eager**。
n：decode=5、ttft=4、ttft8k=1、conc8=16。ACS 全程保持部署态（本轮**未** apply/revert）。

### 3.5 G3：P2（TP=2 同口径）性能

#### 3.5.1 接手 P2（不重启）+ 现场内存构成（同时是任务 §3(3) 的参照测量）

P2 由前任启动，**本次接手只轮询 `P2.log` 的就绪标记，从不重启它**
（`tools/dcu/s14_bench_p2.sh`：先等 `Application startup complete`，再跑 greedy/flip/bench）。

接手时（2026-10-09T19:57Z）P2 处在 graph 捕获阶段，`P2.log` 原文：

```
(Worker_TP0 pid=4843) INFO 10-10 03:57:19 [backends.py:376] Cache the graph of compile range (1, 2048) for later use
(APIServer pid=4672) INFO 10-10 03:58:50 [shm_broadcast.py:812] No available shared memory broadcast block found in 60 seconds. ...
(Worker_TP0 pid=4843) Capturing CUDA graphs (PIECEWISE):   0%|          | 0/51 [00:00<?, ?it/s]
(Worker_TP0 pid=4843) Capturing CUDA graphs (PIECEWISE):   2%|▏         | 1/51 [00:21<17:48, 21.38s/it]
```

**进程清单（重要，影响 monitor 的 `nVLLM` 列解读）**：宿主上有 **76 个历史僵尸**
（`state=Z`，`ppid` 早于本次会话，是 S11/S12/S13 遗留）**+ 2 个 P2 活 worker**，
所以 `P2-monitor.log` 的 `nVLLM ≈ 78` 不是"78 个在占内存的进程"：

```
$ for p in $(pgrep VLLM); do awk '{print $3}' /proc/$p/stat; done | sort | uniq -c
      2 R
     76 Z
$ pgrep -a VLLM | tail -2
2746668 VLLM::Worker_TP0
2746669 VLLM::Worker_TP1
$ # 活进程总 VmRSS
16330968 kB          # = 16.33 GB，全部来自那 2 个 P2 worker
```

**每 worker 宿主内存构成（任务 §3(3) 的原始测量，`sudo -n` 只读）**：

```
$ for p in 2746668 2746669; do grep -E '^(VmRSS|VmHWM|RssAnon|RssFile|RssShmem|VmSwap):' /proc/$p/status; done
=== pid=2746668 VLLM::Worker_TP ===
VmHWM:	 8411260 kB
VmRSS:	 8347900 kB
RssAnon:	 6925660 kB        # 6.60 GiB 匿名
RssFile:	 1333592 kB        # 1.27 GiB 文件页（mmap 的 safetensors，两 worker 共享）
RssShmem:	   88648 kB
VmSwap:	       0 kB
=== pid=2746669 VLLM::Worker_TP ===
VmHWM:	 8407544 kB
VmRSS:	 8345976 kB
RssAnon:	 6923484 kB
RssFile:	 1333848 kB
RssShmem:	   88644 kB
VmSwap:	       0 kB

$ sudo -n grep -E '^(Rss|Pss|Shared_Clean|Private_Dirty|Anonymous|Swap):' /proc/2746668/smaps_rollup
Rss:             8351596 kB
Pss:             7412723 kB
Shared_Clean:    1333072 kB
Private_Dirty:   6928320 kB
Anonymous:       6928312 kB        # 与 /proc/status 的 RssAnon 一致
Swap:                  0 kB

$ sudo -n awk '/Anonymous/{a[cur]+=$2} ...' /proc/2746668/smaps   # 匿名按 mapping 聚合
   3639.5 MiB  [anon]
   3105.6 MiB  [heap]
      3.6 MiB  /opt/dtk-26.04-DCC2602-0317/dcc/comgr/lib/libamd_comgr.so.2.6.0   # 误入：该行的 Anonymous 子项
$ sudo -n awk ... /proc/2746669/smaps
   3640.5 MiB  [anon]
   3102.5 MiB  [heap]
```

> **判读（关键，直接决定任务 §3(2) 的取舍）**：P2（TP=2）的**权重分片 = 28.76 GiB / 2 = 14.38 GiB**，
> 但每 worker 的 `RssAnon` 只有 **6.60 GiB（= 6925660 kB ≈ 6.93 GB）**，且其中 3.1 GiB 在 glibc `[heap]`、
> 3.6 GiB 在无名 `[anon]`（HIP/图池类）——**没有任何一个映射的量级是 14.38 GiB**。
> ⇒ 宿主匿名内存 **不是**"TP 分片大小的权重常驻"；它是**每 worker 近似固定的 ~7 GiB 运行时开销**
> （与分片大小不随 TP 度线性变化这一点，由 P4 那次被 OOM 杀掉的 TP=4 worker
> `anon-rss:7085936kB = 6.76 GiB` 也印证：TP=4 分片 7.19 GiB，两者几乎相同）。
> 这一条对 §3(3) 的"若 RssAnon≈7 GiB/worker⇒swap 会把权重换出去"判据给出的是**否证**：
> 7 GiB 不是权重，swap 不会换出权重（后续 §3.7 再给 P4e 的直接测量）。
> **同时它也解释了 P4 graph 为何必然 OOM**：4 × 6.9 GiB ≈ 27.6 GiB 宿主匿名 + APIServer/EngineCore
> 进程 + 页缓存，在 31.26 GiB RAM 上必然挤爆。

#### 3.5.2 机制补充：graph 捕获在这套栈上是**两遍**（也是 P4 OOM 的真正落点）

观察 P2 的日志可以确定：vLLM v0.21+ 起用了 **"CUDA graph memory profiling"**，
`compile_or_warm_up_model()` 会**捕获两遍 graph**：

* **第 1 遍 = 量测遍**（`model_runner.py:1060` 的 "Graph capturing finished in ... took 1.67 GiB"）；
* 然后用这次量测结果算出 `Available KV cache memory`，**分配 KV cache**；
* **第 2 遍 = 真实遍**（KV cache 已占住显存之后重新捕获一次）。

`P2.log` 原文（容器内 +08:00）：

```
(Worker_TP0 pid=4843) Capturing CUDA graphs (PIECEWISE): 100%|██████████| 51/51 [11:22<00:00, 13.39s/it]
(Worker_TP0 pid=4843) Capturing CUDA graphs (FULL): 100%|██████████| 2/2 [00:01<00:00, ...]
(Worker_TP1 pid=4844) INFO 10-10 04:10:16 [model_runner.py:1060] Graph capturing finished in 685 secs, took 1.67 GiB
(Worker_TP0 pid=4843) INFO 10-10 04:10:17 [model_runner.py:1060] Graph capturing finished in 685 secs, took 1.67 GiB
(Worker_TP0 pid=4843) INFO 10-10 04:10:20 [gpu_worker.py:719] Available KV cache memory: 9.93 GiB
(APIServer pid=4672) INFO 10-10 04:10:20 [kv_cache_utils.py:2432] GPU KV cache size: 318,654 tokens, Maximum concurrency for 262,144 tokens per request: 1.22x
(Worker_TP0 pid=4843) INFO 10-10 04:10:20 [kernel_warmup.py:172] JIT kernel warmup starting.
(Worker_TP0 pid=4843) INFO 10-10 04:10:21 [qwen_triton_warmup.py:293] Warming up Qwen GDN Triton kernels for model_type=qwen3_5_text.
...
(Worker_TP0 pid=4843) Capturing CUDA graphs (PIECEWISE):   0%|          | 0/51 [00:00<?, ?it/s]   ← 第 2 遍从这里开始
```

⇒ **P4（graph 模式）的 OOM 落点被精确解释**：`P4.log` 里
`Available KV cache memory: 16.99 GiB` 之后 **5 秒**被杀——那不是"KV 分配失败"，
而是**第 2 遍捕获起步时**（KV cache 已占 16.99 GiB/rank×4）宿主匿名内存被顶爆。
`P2` 相同的两步之间隔了 1 秒，但它活了下来（TP=2 宿主匿名需求只有 2×6.9 GiB）。

> **monitor 进程被回收（记录）**：前任的 `s14_monitor.sh P4` / `P2` 两个宿主监控进程
> 在 **2026-10-09T20:02:05Z 同时消失**（日志无 `# end=` 收尾行，说明不是自然结束；
> 属会话回收，非 OOM）。原始时间序列已另存为
> `/mnt/game/s14/hostlogs/{P2,P4}-monitor-part1.log`，接手方随后以新 label `P2b` 续采。

#### 3.5.3 P2 就绪与 bench 原始数字

就绪：`[p2] READY 2026-10-09T20:22:41+00:00`（自 19:54:56Z 启动起 **1665 s**，其中两遍 graph 捕获
各 685 s），`free_at_ready: total 32009 used 21022 free 1945 avail 10436 Swap: 4095 451 3644`
（**P2 全程几乎不动 swap：451 MiB**）。

原始日志 `/mnt/game/s14/bench-P2-{decode,ttft,ttft8k,conc8}.log`，聚合 JSON
`/mnt/game/s14/openai-chat-infqps-*-20261010-0426*.json`、`-0428*.json`：

```
########## P2 decode (in=128 out=128 np=5 warmups=1 conc=1) ##########
Successful requests: 5      Failed: 0      Benchmark duration (s): 56.62
Total input tokens: 903     Total generated tokens: 640
Output token throughput (tok/s): 11.30     Total token throughput (tok/s): 27.25
Mean TTFT 625.74  Median TTFT 630.32  P90 634.70  P99 637.07
Mean TPOT  84.24  Median TPOT  84.22  P90  84.36  P99  84.38          ← n=5
Mean ITL   83.58  Median ITL   84.86

########## P2 ttft (in=3800 out=1 np=4 conc=1) ##########
Successful requests: 4      Benchmark duration (s): 34.52
Total input tokens: 15411
Mean TTFT 8630.88  Median TTFT 8624.35  P99 8667.31                   ← n=4

########## P2 ttft8k (in=8100 out=1 np=1) ##########
Successful requests: 1      Benchmark duration (s): 20.84
Total input tokens: 8154    Total token throughput (tok/s): 391.33
Mean TTFT = Median TTFT = P99 TTFT = 20838.63                          ← n=1

########## P2 conc8 (in=512 out=128 np=16 warmups=2 conc=8) ##########
Successful requests: 16     Failed: 0      Benchmark duration (s): 61.42
Total input tokens: 9031    Total generated tokens: 2048
Output token throughput (tok/s): 33.34     Total token throughput (tok/s): 180.37
Mean TTFT 7872.58  Median TTFT 9907.15  P90 10104.68  P99 10137.84
Mean TPOT 177.53  Median TPOT 163.53  P90 206.67  P99 210.50          ← n=16
Mean ITL  176.14  Median ITL  164.15
```

**口径标注**：`maxlen 262144 / util 0.85 / MBT 2048 / prefix-cache on / mamba align /
--disable-custom-all-reduce / VLLM_DCU_USE_INT8_SDOT4=1`，**TP=2 / graph（FULL_AND_PIECEWISE，
51 PIECEWISE + 2 FULL 桶）**。**P2 的 decode 84.22 ms/token 与 S11l 历史 TP=2 graph 的
84.00–84.68 完全吻合**（口径差异：本题 maxlen 262144 vs 历史 4096，util 0.85 vs 0.9）⇒ 口径自洽。

#### 3.5.4 四口径性能对照表（不许混口径，每行都标）

| 指标（原始中位数） | P2：TP=2 **graph** | P4e：TP=4 **eager** | S13 vendor TP=4 **eager**（不重测） | P4e/P2 倍率 |
|---|---|---|---|---|
| decode TPOT (ms/token) | **84.22**（P90 84.36，P99 84.38，n=5） | **126.83**（P90 132.30，P99 132.68，n=5） | 121.43（n=4，mean [117.57,124.44]） | **1.506× 慢** |
| TTFT in≈3850 (ms) | **8624.35**（P99 8667.31，n=4） | **8920.34**（P99 9276.75，n=4） | —（S13 未测该点） | 1.034× 慢 |
| TTFT in=8154 (ms) | **20838.63**（n=1） | **12962.54**（n=1） | — | **0.622×（P4e 快 1.61×）** |
| conc8 输出吞吐 (tok/s) | **33.34**（conc 8，n=16） | **35.55**（conc 8，n=16） | — | 1.066× |
| conc8 中位 TPOT (ms) | 163.53（n=16） | 156.19（n=16） | — | 0.955× |
| 可用 KV cache/rank | 9.93 GiB（318,654 tok） | 18.69 GiB（1,200,839 tok） | — | 1.88× |
| 启动到就绪 | 1665 s（两遍 51 桶捕获） | **190 s** | — | 8.8× 快 |

* 与生产基线的相对位置：**P4e / vendor = 126.83 / 121.43 = 1.044**（我们的树在 TP=4/eager 下
  比生产 vendor 栈慢 **4.4%**，属同一档）；**P2 / vendor = 0.694**（TP=2 graph 比生产 TP=4 快 **1.44×**）。
* 综上：这台机上 **TP=2 的 decode 明显优于 TP=4（1.51×）**，而 **TP=4 只在长 prefill（8K）上赢（1.61×）**，
  conc8 吞吐两者接近（TP=4 略好 6.6%）。

### 3.6 G2：P4 vs P2 数值对比

#### 3.6.0 为什么必须加一个 control 臂（P2e）

任务书 G2 要求"TP=4 与 TP=2 在同一条 greedy prompt 上输出文本一致或仅贴边翻转"。
但 **P4e 与 P2 之间同时变了两个因子**：并行度（2→4）**和**编译模式（graph→eager）。
按任务书 §0/§3 的口径，本步不做既非 TP 也非模式控制的比较，因此**追加了一个 control 臂**：

```
P2e = TP=2 + --enforce-eager（其余 flags 与 P2/P4e 逐字相同）

$ S14_CT=z100-s14 S14_EAGER=1 S14_FLIP_TOKENS=128 S14_GREEDY_TOKENS=256 READY_TIMEOUT=1200 \
    nohup tools/dcu/s14_run_one.sh P2e /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 2 1 \
    /mnt/game/s14/cache-27b-p2e 8424 262144 0,1 greedy,flip \
    > /mnt/game/s14/hostlogs/orch-P2e.log 2>&1 &
[orch14] ready=150s aot_load=0 aot_recompile=0 oom_kill=1 | NA | Available KV cache memory: 11.65 GiB
[orch14] done name=P2e ready=1
```

这样三个比较互相正交：

| 比较 | 固定 | 变化 | 隔离出 |
|---|---|---|---|
| **B1** P2e vs P2 | TP=2 | eager↔graph | **编译模式效应** |
| **B2** P4e vs P2e | eager | TP=2↔4 | **并行度效应** |
| B3 P4e vs P2 | 无 | 两者同时 | 任务书原始 G2 对照 |

#### 3.6.1 原始对比结果（`tools/dcu/s14_logprob_compare.py`）

```
############ B1) MODE effect @TP=2: P2e(eager) vs P2(graph) ############
s14_greedy_long  256 256  0  0.00000 0.00000  0   0  '用户'  'The'   0.859   4.750   0.859  254  8.180
zh_hz   128 128  8  0.2680 0.0135 ...  first=8   min_gap@first=0.109   worst=7.492
zh_ml   128 128  2  0.5798 0.0732 ...  first=2   min_gap@first=0.984   worst=5.953
zh_sort 128 128 21  ...                first=21  min_gap@first=0.422   worst=9.750
zh_sky  128 128 14  ...                first=14  min_gap@first=0.031   worst=7.281
en_fr   128 128  0  ...                first=0   min_gap@first=0.406   worst=9.180
en_sea  128 128 23  ...                first=23  min_gap@first=0.313   worst=6.656
en_index128 128 19  ...                first=19  min_gap@first=0.031   worst=8.188
en_math 128 128  0  ...                first=0   min_gap@first=0.281   worst=9.688
# prompts token-identical : 0/8 ; tokens on common prefix : 87 of 952
# max |dlogprob| (nat) 0.579771 ; mean |dlogprob| (nat) 0.073237
# VERDICT: NOT A NEAR-TIE (>=0.5 nat)

############ B2) TP effect @eager: P4e(TP4) vs P2e(TP2) ############
s14_greedy_long  256 256  1  0.35105 0.35105  1   1  '要求'  '问'   1.359   1.172   1.172  252  9.445
zh_hz   128 128  2   ...  first=2   min_gap@first=0.156   worst=7.039
zh_ml   128 128  2   ...  first=2   min_gap@first=0.984   worst=8.945
zh_sort 128 128 11   ...  first=11  min_gap@first=0.797   worst=9.422
zh_sky  128 128 14   ...  first=14  min_gap@first=0.031   worst=7.984
en_fr   128 128 12   ...  first=12  min_gap@first=0.344   worst=8.969
en_sea  128 128 12   ...  first=12  min_gap@first=0.516   worst=7.563
en_index128 128  2   ...  first=2   min_gap@first=1.188   worst=9.969
en_math 128 128  0   ...  first=0   min_gap@first=2.109   worst=9.828
# prompts token-identical : 0/8 ; tokens on common prefix : 55 of 944
# max |dlogprob| (nat) 0.867332 ; mean |dlogprob| (nat) 0.088255
# VERDICT: NOT A NEAR-TIE (>=0.5 nat)

############ B3) 任务书原始对照: P4e(TP4 eager) vs P2(TP2 graph) ############
s14_greedy_long  256 256  0  0.00000 0.00000  0   0  '用户'  'The'   4.031   4.750   4.031  256  9.953
# prompts token-identical : 0/8 ; tokens on common prefix : 43 of 944
# max |dlogprob| (nat) 0.843882 ; mean |dlogprob| (nat) 0.102977
# VERDICT: NOT A NEAR-TIE (>=0.5 nat)
```

#### 3.6.2 机制判定（数值性 vs 调度性）

**结论：数值性（系统性），且主导因子是"编译模式"而不是"并行度"。**

依据：

1. **同一个 prompt 的第一个生成 token**（三台引擎，greedy / temperature=0 / 单请求 / conc=1）：

   | 引擎 | 模式 | TP | 首 token | 该侧 top1–top2 间距 |
   |---|---|---|---|---|
   | P2 | graph | 2 | `'The'` | 4.750 nat |
   | P2e | **eager** | 2 | `'用户'` | 0.859 nat |
   | P4e | **eager** | 4 | `'用户'` | 4.031 nat |

   ⇒ **两个 eager 臂在首个 token 上一致（都是 `用户`），graph 臂不同**。
   两个 eager 臂的首 token 间距差 4.03 vs 0.86 nat 说明 logits 有差，但 argmax 相同。
   而 graph↔eager 的分歧意味着 logits 至少差 **≳4.4 nat**（graph 侧对 `The` 有 4.75 nat 把握，
   eager 侧却选了 `用户`）——**这比 TP 效应（B2 首分歧只要 ≳1.2 nat）更大**。
2. **不是"换了 kernel/后端"**：两模式的日志选择完全一致 ——
   `TRITON_ATTN` 解码后端、`Gfx906Sdot4Int8ScaledMMLinearKernel`、`Using Triton/FLA GDN prefill kernel`、
   `['PYNCCL']` all-reduce、`LBNHC` KV layout（§3.3.1 与 §3.5.1 原文）。
   所以差异来自**同一批 kernel 的不同算术路径与舍入顺序**：`CompilationMode.NONE`（eager）不做
   Inductor 融合/切分，而 graph 路径做 `rms_norm+quant` 等融合并重排 fp16→int8 的舍入；
   TP 度改变 all-reduce 的归约次序。
3. **共享前缀上的 logprob 其实很接近**：B1 mean |Δlogprob| = **0.073 nat**（max 0.580），
   B2 mean = **0.088 nat**（max 0.867）。⇒ 算术差异是**小幅**的，翻转发生在 argmax 边界上，
   然后自回归放大。这正是项目所说"贴边翻转"的**形状**，只是首个翻转点的间距
   （模式效应 0.86–4.75 nat、TP 效应 1.17–1.36 nat）**高于项目 0.5 nat 的阈值**。
4. **不是调度性**：temperature=0 纯 greedy、`top_p=1.0`、单请求 `conc=1`、prompt 完全相同，
   不存在采样随机性或批内顺序影响。
5. **`s14_logprob_compare.py` 的 "worst gap"（8–10 nat）不可用作判据**：首个翻转之后两条流
   已处于**不同上下文**，后续位置的 gap 不再可比。有效统计量是**首个分歧点**及其两侧间距，
   上表已按此给出。

**与 G2 判据的字面对照**：G2 要求"文本一致**或**仅贴边翻转"。
**本条门禁不通过**——三个比较都是 0/8 prompt token-identical，首个分歧点间距超过 0.5 nat。
但**机制是数值性且已定位到编译模式**，不是调度性、不是 NaN、不是后端错选。

**附带观察（对项目更重要）**：graph 模式（我们一直在用的生产口径）在这条 prompt 上输出退化成
英文重复循环（`'The user is asking in Chinese: ...'` 反复），而 eager 输出连贯中文；
`en_fr` 一例 graph 跑满 128 token，eager 在 48 token 自然 `stop`。
⇒ 编译模式不仅改变 token 流，还改变了**输出质量**。这一条与 S7b/S7c 修掉的 NaN 不同，
是一次新的、独立的现象，建议 Lead 立新步（如 S14b）在 `CompilationMode.VLLM_COMPILE` 下
用固定 prompt 逐 token 对齐 eager 参考，定位是哪一处融合改变了算术。

#### 3.6.3 P4g2（graph TP=4）加入后的四臂矩阵（大幅收紧了机制判断）

P4g2 就绪后补做了两个关键比较（`greedy-P4g2.jsonl`）：

```
############ C) graph 模式下的 TP 效应: P4g2(TP4 graph) vs P2(TP2 graph) ############
s14_greedy_long  256 256  0  0.00000 0.00000  0  0  '用户'  'The'  2.219  4.750  2.219  251  9.625
# 首个分歧在 **token 0**，两侧间距 2.219 / 4.750 nat ⇒ 远大于 0.5 nat

############ D) TP=4 时的模式效应: P4g2(graph) vs P4e(eager) ############
s14_greedy_long  256 256  2  0.28765 0.20324  2  2   '：'    '用'  0.078  0.125  0.078  249 10.688
# 首个分歧在 **token 2**，两侧间距 **0.078 / 0.125 nat ⇒ 远低于 0.5 nat（这是真正的"贴边翻转"）**
```

四条 greedy 输出对照（**同一个 prompt、temperature=0**）：

| arm | 模式 | TP | 首 token | 文本头 | token_ids sha256 前 16 |
|---|---|---|---|---|---|
| P2 | graph | 2 | `'The'` | `The user is asking in Chinese: "In 31 GiB...` （英文重复循环） | `4097ad3cbea2fff2` |
| P2e | eager | 2 | `'用户'` | `用户问的是：在 31 GiB 内存的单机上，...` | `3788a8622b2ef18a` |
| P4e | eager | 4 | `'用户'` | `用户要求用中文逐步说明：在 31 GiB 内存的单机上，...` | `b45a7e45c6c2064b` |
| P4g2 | graph | 4 | `'用户'` | `用户要求：用中文逐步说明：在 31 GiB 内存的单机上，...` | `5a5f62d133dd12b8` |

**据此修正 §3.6.2 的机制判断（比原来更精确）**：

1. **模式效应不是恒定的，它取决于 TP**：
   * 在 **TP=4**，graph vs eager 在 token 2 才分歧且间距只有 **0.078 nat** ⇒ **近乎等价**；
   * 在 **TP=2**，graph vs eager 在 token 0 就分歧且 graph 侧对 `'The'` 有 4.75 nat 把握 ⇒ **很大**。
2. **四个 arm 里唯一的异类是 P2（TP=2 graph）**——只有它走英文重复循环，其余三个都输出连贯中文。
3. **TP 效应（同模式内）稳定地"非贴边"**：graph 内（C）首分歧 2.22/4.75 nat；eager 内（B）首分歧 1.17/1.36 nat。
   ⇒ **改变并行度确实会改变 token 流**（all-reduce 归约次序 + 分片切分造成的 fp16 舍入差），
   且幅度超过项目的 0.5 nat 阈值。
4. **最终机制口径**：这是**数值性/系统性**差异（确定性、可复现、多次 0.03–4.75 nat 的 argmax 翻转），
   由**算术路径**（并行度归约次序 + 编译融合/切分）造成，**不是**采样随机性、**不是**调度竞争、
   **不是** NaN、**不是**后端错选（§3.6.2 已逐项排除）。共享前缀上 logprob 平均只差 0.07–0.20 nat
   （D 的 mean 0.203、B1 的 0.073、B2 的 0.088）⇒ 差异是"小幅算术差 + 自回归放大"。
5. **没有参考真值**：本机无法用 TP=1 做参照（28.76 GiB 权重 > 单卡 32 GiB 显存，装不下），
   也无法用 vendor 栈做逐 token 对照（不同 vLLM 版本/不同 tokenizer 归一化）。
   因此**不能判定四个输出中哪个"更正确"**——只能判定它们**不同**且**都是有限值**。
   这是本次 G2 的**能力边界**，如实登记。

**未执行的控制（如实登记）**：**同引擎内 greedy 可重复性**（同一引擎连发两次同一 prompt 比较
token 串）未测；本轮 4 个引擎都已关闭，重开一次需约 150 s（eager）。该项与"确定性"结论相关，
如需缩短论证链应补测。

> **补充（2026-10-09T21:20Z，P4g3 复测后补齐）**：§3.8.3 的 **同配置、跨引擎重启复测**
> （P4g3 vs P4g2）已提供等效的确定性对照：**首个分歧同样在 token 2，间距 0.078 / 0.562 nat**
> ⇒ 这套栈**在引擎重启之间不是 bit 级确定性的**，噪声量级约 **0.1–0.6 nat**。
> 据此重排三个效应的量级（都以"首个分歧点的两侧间距"比噪声底）：

| 比较 | 变化 | 首个分歧 | 两侧间距（nat） | 判定 |
|---|---|---|---|---|
| P4g3 vs P4g2 | **无（同配置重启）** | token 2 | 0.562 / 0.078 | **噪声底** |
| P4g2 vs P4e | 模式（TP=4） | token 2 | 0.125 / 0.078 | **在噪声底内 ⇒ 近乎等价** |
| P2e vs P2 | 模式（TP=2） | token 0 | 0.859 / 4.750 | 超噪声底 ⇒ 系统性 |
| P4e vs P2e | TP（eager） | token 1 | 1.359 / 1.172 | 超噪声底 ⇒ 系统性 |
| P4g2 vs P2 | TP+图配置（graph） | token 0 | 2.219 / 4.750 | 超噪声底 ⇒ 系统性 |

⇒ **修正后的最终口径**：所有 TP 相关的分歧（1.17–4.75 nat）**都超出 0.1–0.6 nat 的噪声底**，
是**系统性数值差异**；而 **TP=4 下的"模式"差异（0.078 nat）落在噪声底之内**，
不能算作模式效应。**TP=2 graph 那一路（P2）在一串英文重复循环上给出 4.75 nat 的把握，是本轮唯一的异类**。

### 3.7 任务 §3(3)：每 worker 匿名/文件内存构成（决定 swap 尝试是否值得）

工具：`tools/dcu/s14_smaps.sh`（宿主侧只读；`sudo -n` 读 `/proc/<pid>/smaps_rollup`）
与 `tools/dcu/s14_smaps_loop.sh`（时间序列）。原始文件：
`/mnt/game/s14/hostlogs/smaps-P2-runtime.txt`、`smaps-P2-after-traffic.txt`、
`smaps-P2-anon-bymapping.txt`、`smaps-P4e.txt`。

#### 3.7.1 P4e（TP=4 / eager）稳态每 worker 构成（bench 进行中采样）

```
$ for p in $(pgrep VLLM); do grep -E '^(VmRSS|VmHWM|RssAnon|RssFile|VmSwap|Threads):' /proc/$p/status; done
--- pid=2774908 ---  VmRSS 4882924  VmHWM 6014640  RssAnon 4550024  RssFile 199680  VmSwap 228296  Threads 44
--- pid=2774909 ---  VmRSS 4888852  VmHWM 6087388  RssAnon 4511400  RssFile 199956  VmSwap 266240  Threads 43
--- pid=2774910 ---  VmRSS 5116320  VmHWM 6217300  RssAnon 4740184  RssFile 198640  VmSwap 243292  Threads 43
--- pid=2774911 ---  VmRSS 5056692  VmHWM 6200520  RssAnon 4723652  RssFile 199832  VmSwap 259968  Threads 43

$ 匿名按 mapping 聚合（sudo awk over smaps）
   2542.0 MiB  [anon]     1896.4 MiB  [heap]      ← pid=2774908
   2549.1 MiB  [anon]     1851.4 MiB  [heap]      ← pid=2774909
   2658.8 MiB  [anon]     1965.4 MiB  [heap]      ← pid=2774910
   2670.5 MiB  [anon]     1937.3 MiB  [heap]      ← pid=2774911

$ APIServer/EngineCore 进程（pid=2773995, `python -c ... serve ... --enforce-eager`）
VmRSS 1740576  RssAnon 1673956  RssFile 66204  VmSwap 1522964  Threads 49
                                                       ↑ **换出的主力是 EngineCore，不是 worker**
```

#### 3.7.2 关键判读（回答 Lead 的决策准则）

| 对象 | TP | 模式 | 权重分片（host 口径参考） | 每 worker `RssAnon` | 每 worker `RssFile` |
|---|---|---|---|---|---|
| P2 worker | 2 | graph | 28.76 GiB / 2 = **14.38 GiB** | **6.60–6.93 GiB** | 1.27 GiB |
| P4e worker | 4 | eager | 28.76 GiB / 4 = **7.19 GiB** | **4.30–4.52 GiB** | 0.19 GiB |

1. **`RssAnon` 明显小于权重分片**（TP=4 时 4.4 GiB vs 7.19 GiB 分片；TP=2 时 6.6 GiB vs 14.38 GiB 分片）。
   ⇒ **权重不是以匿名常驻内存的**（0.85×32 GiB 显存的 `Model loading took 7.27 GiB`×4 说明权重在**显存**里），
   `RssAnon` 是**每 worker 的运行时/池**，与分片大小**不正比**（TP=2→4 时只从 6.6 降到 4.4 GiB）。
   **按 Lead 的准则第 2 支（"RssAnon 明显小于权重分片 ⇒ (2) 值得做"）判**。
2. **实测 swap 没有把权重换出去**：本次 P4e 启动期宿主 swap 由 450 MiB 涨到 **3195 MiB**（§3.4.1），
   而 bench 进行中 4 个 worker 各自只被换出 **223–266 MiB**（合计 ≈0.98 GiB），
   **换出主力是 EngineCore/APIServer 进程（1523 MiB）**。worker 的 `[heap]`（1.85–1.97 GiB）
   与 `[anon]`（2.54–2.67 GiB）在压力下基本留在 RAM 里，decode 仍跑到 126.83 ms/token。
3. **`RssFile` 差异**：TP=2 graph 时 1.27 GiB/worker（mmap 的 safetensors 页，两 worker 共享），
   TP=4 eager 时只有 0.19 GiB —— 与 §3.3.1 内核 OOM 记录里的 `file-rss:0kB` 一致：
   **页缓存是首批被回收的**，所以 OOM 时刻它是 0。
4. 因此 **`/mnt/game/swapfile`（16 GiB）在"容量"上不能替代内存**（权重与 worker 运行时都是匿名、
   且每 token 都碰），但它可以**桥接启动瞬时峰值**：本次实测换出 ~2.7 GB 即换来
   TP=4/eager 的 190 s 起飞与稳态 ~8.8 GB 可用余量。

> 附注：`/proc/<pid>/environ` 对 worker 是**陈旧**的（vLLM worker 在 exec 后 putenv），
> 所以 SDOT4/HIP 可见卡号的权威读数是 APIServer 进程：`VLLM_DCU_USE_INT8_SDOT4=1`、
> `HIP_VISIBLE_DEVICES=0,1`（P2 期采样原文），容器内命令行亦见 §3.4.1。

### 3.8 P4g：graph 模式 TP=4 的两次受限尝试

Lead 的口径：先做**零宿主改动**的降 footprint 尝试
（`-cc.cudagraph_mode=PIECEWISE` + `--cudagraph-capture-sizes 1 2 4 8`，只留真正用到的桶）；
**仍 OOM 才**启用 `/mnt/game/swapfile`（16 GiB）重试**同一 arm**。

#### 3.8.1 ❌ P4g1（零宿主改动）：**限制捕获桶完全无效，仍然 global OOM**

```
$ cd /home/lyh/z100-vllm
$ S14_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" \
    S14_GREEDY_TOKENS=256 READY_TIMEOUT=2400 \
    nohup tools/dcu/s14_run_one.sh P4g1 /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 4 1 \
    /mnt/game/s14/cache-27b-p4g1 8423 262144 0,1,2,3 decode,greedy \
    > /mnt/game/s14/hostlogs/orch-P4g1.log 2>&1 &
[s14] name=P4g1 ... tp=4 sw=1 port=8423 maxlen=262144 car=1 util=0.85 mbt=2048 eager=0 cg_mode=PIECEWISE cg_sizes=1 2 4 8 extra=
配置生效原文（compilation_config）：
  'cudagraph_mode': <CUDAGraphMode.PIECEWISE: 1>
  'cudagraph_capture_sizes': [1, 2, 4, 8]
  'max_cudagraph_capture_size': 8
```

**启动期关键行（原文，证明"桶真的被限住了"）**：

```
(Worker_TP0 pid=16844) INFO 10-10 04:54:14 [model_runner.py:406] Model loading took 7.27 GiB memory and 41.712844 seconds
                       （Worker_TP1..TP3 同值 7.27 GiB）
(Worker_TP0 pid=16844) INFO 10-10 04:56:07 [monitor.py:53] torch.compile took 108.52 s in total        ← 编译耗时与 P4 的 109.57 s 同档
(Worker_TP0 pid=16844) Capturing CUDA graphs (PIECEWISE): 100%|██████████| 4/4 [00:05<00:00, 1.42s/it]   ← **4 桶，不是 51 桶**
(Worker_TP0 pid=16844) INFO 10-10 04:56:58 [model_runner.py:1060] Graph capturing finished in 7 secs, took 0.22 GiB
                       （Worker_TP1..TP3 同值；P4 那次是 **685 secs / 1.67 GiB**）
(Worker_TP0 pid=16844) INFO 10-10 04:56:59 [gpu_worker.py:719] Available KV cache memory: 18.47 GiB
(APIServer pid=16619) INFO 10-10 04:56:59 [kv_cache_utils.py:2432] GPU KV cache size: 1,187,496 tokens, Maximum concurrency: 4.53x
(Worker_TP0 pid=16844) Capturing CUDA graphs (PIECEWISE):   0%|          | 0/4 [00:00<?, ?it/s]      ← 第 2 遍刚起步
(APIServer pid=16619) ERROR 10-10 04:57:09 [multiproc_executor.py:314] Worker proc VllmWor... died
```

**内核原文（第 4 次 OOM，同一口径）**：

```
[Fri Oct  9 20:57:05 2026] oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=user.slice,mems_allowed=0,
    global_oom,task_memcg=/system.slice/docker-5292b9242a3899f70c1139274c9ec43145b8096ccda8938544ce15b67314d34b.scope,
    task=VLLM::Worker_TP,pid=2807380,uid=0
[Fri Oct  9 20:57:05 2026] Out of memory: Killed process 2807380 (VLLM::Worker_TP)
    total-vm:61532728kB, anon-rss:7211536kB, file-rss:0kB, shmem-rss:177240kB, UID:0
```

**收口原文**：

```
[orch14] ready=0 after 310s  2026-10-09T20:57:18+00:00
[orch14] summary: ready=310s aot_load=0 aot_recompile=0 oom_kill=2 | NA | Available KV cache memory: 18.47 GiB
[orch14] done name=P4g1 ready=0
```

**⇒ 决定性结论：这次 OOM 与图池大小无关。**
图池从 **1.67 GiB 砍到 0.22 GiB（7.6×）**、捕获从 685 s 降到 7 s，
但**死亡点、死亡原因、被 OOM 进程的 `anon-rss`（7.21 GB vs 上次 7.09 GB）几乎不变**，
仍然发生在 **第 2 遍捕获起步、KV cache 分配之后 6 秒**。
所以 graph 模式 TP=4 的宿主匿名峰值**不是**图捕获池，而是**编译/切分执行路径本身的常驻宿主内存**
（与 §3.7 的测量一致：graph 每 worker 6.6 GiB vs eager 4.4 GiB，差 ≈2.2 GiB/worker ×4 ≈ **8.8 GB**，
恰好等于 P4e 稳态可用的 8.8 GB）。⇒ **调捕获桶救不了，任务 §3(4) 的 sdot4 假设也不必测。**

#### 3.8.2 P4g2（启用 `/mnt/game/swapfile` 16 GiB 后重试同一 arm）

**`swapon` 前后 `/proc/swaps` 原文（任务书要求）**：

```
$ cat /proc/swaps                      # BEFORE
Filename                                Type            Size            Used            Priority
/swap.img                               file            4194300         497832          -2

$ sudo -n swapon /mnt/game/swapfile    # rc=0
$ cat /proc/swaps                      # AFTER
Filename                                Type            Size            Used            Priority
/swap.img                               file            4194300         497064          -2
/mnt/game/swapfile                      file            16777212        0               -3

$ free -m                              # AFTER
               total        used        free      shared  buff/cache   available
Mem:           32009        3041       28247           0         720       28527
Swap:          20479         485       19994
```

> 硬停规则（Lead 口径）：① `swapon` 后 **60 min 未 ready ⇒ 杀掉、记"未就绪"**（截止 21:59Z）；
> ② 若 ready，**decode TPOT 必须 ≤ 84.68 ms/token**（我们的 TP=2 graph 口径），慢于它即判
> "graph TP=4 在这台机上不可行/不值得"，立即停、不再调参；③ 收尾必须 `swapoff` 还原并给前后原文。

```
$ S14_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" S14_GREEDY_TOKENS=256 \
    READY_TIMEOUT=3600 nohup tools/dcu/s14_run_one.sh P4g2 ... 4 1 \
    /mnt/game/s14/cache-27b-p4g2 8425 262144 0,1,2,3 decode,greedy \
    > /mnt/game/s14/hostlogs/orch-P4g2.log 2>&1 &
启动时刻：2026-10-09T20:59:0xZ（`swapon` 后 1 分钟内）
```

> （P4g2 结果在本节续写）

**✅ P4g2 结果：swap 让 graph 模式 TP=4 装下并就绪，而且它是本轮最快的 arm。**

```
[orch14] ready=1 after 350s  2026-10-09T21:04:22+00:00
[orch14] summary: ready=350s aot_load=0 aot_recompile=0 oom_kill=2 |
   init engine (profile, create kv cache, warmup model) took 184.27 s (compilation: 105.23 s) |
   Available KV cache memory: 18.47 GiB
[orch14] done name=P4g2 ready=1

# 启动期关键行（与 P4g1 同配置）
(Worker_TP0) INFO 10-10 05:02:13 [monitor.py:53] torch.compile took 104.67 s in total
(Worker_TP0..TP3) Graph capturing finished in 6~7 secs, took 0.22 GiB / 0.10 GiB
(Worker_TP0) INFO 10-10 05:03:14 [gpu_worker.py:719] Available KV cache memory: 18.47 GiB
(APIServer) INFO 10-10 05:03:14 [kv_cache_utils.py:2432] GPU KV cache size: 1,187,496 tokens, Max concurrency 4.53x
```

**`oom_kill` 保持 2（未增长）⇒ 16 GiB swap 把 graph 模式 TP=4 的启动峰值接住了。**

```
########## P4g2 decode (in=128 out=128 np=5 warmups=1 conc=1) ##########
Successful requests: 5      Failed: 0      Benchmark duration (s): 37.24
Total input tokens: 903     Total generated tokens: 640
Output token throughput (tok/s): 17.19     Total token throughput (tok/s): 41.43
Mean TTFT 686.34  Median TTFT 672.98  P90 720.86  P99 744.81
Mean TPOT  53.24  Median TPOT  49.46  P90  60.87  P99  67.59     ← **n=5，最快的 arm**
Median ITL 49.67
```

**启动期内存（`P4g2-monitor.log`，15 s 采样，bench 窗口 21:04–21:06）**：

```
2026-10-09T21:03:36+00:00 32009 30757  217  507 5121 29779959808 max 0 2  84 114344
2026-10-09T21:04:21+00:00 32009 29597 1390 1667 7451 28395761664 max 0 2  84 114344   ← bench 开始
2026-10-09T21:05:22+00:00 32009 30197  362 1058 7782 29431885824 max 0 2  84 114624   ← swap 用满 7.78 GB
2026-10-09T21:05:37+00:00 32009 29375 1147 1880 7782 28578598912 max 0 2  84 114624   ← bench 结束
2026-10-09T21:06:19+00:00 32009  3006 27804 28552 530  749080576 max 0 2  83      0   ← 引擎关闭后 swap 立刻回落到 0.53 GB
```

> **重要判读**：bench 期间 swap 用到 **7.78 GB**、`available` 只剩 1.0–1.9 GB，
> 但 **decode 仍跑到 49.46 ms/token**。引擎一关，swap 立刻从 7.78 GB 回落到 0.53 GB
> ⇒ 被换出的 **7.3 GB 确实是引擎进程的页**，但显然**不是每 token 都碰的热页**
> （否则 49 ms/token 不可能）。这与 §3.7 的结论一致：可安全换出的是**加载期权重 staging / 冷页**，
> 而**权重本体在显存里**。⇒ **swap 在这台机上对 graph 模式 TP=4 是"净收益"**（代价见下）。
>
> **但必须写明代价与口径警告**：① 这个 arm **依赖 16 GiB swap 才装得下**，且启动期
> `available` 一度只剩 507 MB（21:03:36），余量极小；② 它的图配置**不是** P2 的默认配置
> （P2 = `FULL_AND_PIECEWISE` + 51+2 桶；P4g2 = `PIECEWISE` + 4 桶），
> 所以 **49.46 与 P2 的 84.22 之间的 1.70× 里混入了"图/编译配置"的贡献**（见 §3.8.3 复测与 §4 G3 的保留意见）。

#### 3.8.3 P4g2 复测（P4g3，同配置独立一次全程）+ 2×2 并行度/模式矩阵

**P4g3 = 与 P4g2 完全同配置的第二次独立运行**（新 cache 目录，全新编译）：

```
$ S14_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" S14_GREEDY_TOKENS=256 \
    READY_TIMEOUT=3600 nohup tools/dcu/s14_run_one.sh P4g3 ... 4 1 \
    /mnt/game/s14/cache-27b-p4g3 8426 262144 0,1,2,3 decode,ttft,ttft8k,conc8,greedy \
    > /mnt/game/s14/hostlogs/orch-P4g3.log 2>&1 &
[orch14] ready=1 after 351s  2026-10-09T21:14:50+00:00        ← P4g2 是 350 s
[orch14] summary: ready=351s oom_kill=2 | init engine took 182.72 s (compilation: 104.18 s) | Available KV cache memory: 18.47 GiB

########## P4g3 decode ##########
Successful requests: 5   Failed: 0   Benchmark duration (s): 35.59
Mean TPOT 49.88  Median TPOT 49.86  P90 50.70  P99 51.15      ← P4g2 是 49.46 / P99 67.59
########## P4g3 ttft (in≈3853) ##########   Median TTFT 9009.84  Mean 9052.26  P99 9456.58
########## P4g3 ttft8k (in=8154) ##########  TTFT 12970.91      ← P4e 是 12962.54（几乎相同）
########## P4g3 conc8 ##########           Output throughput 41.13 tok/s   Median TPOT 120.83  Median TTFT 9545.72
```

⇒ **启动 350/351 s、decode 49.46/49.86 ms/token 两次一致**（差 0.8%），**复现成立**。
合并 n=10：decode 中位 **49.66 ms/token**，两轮极值 **[49.46, 49.86]**，P99 最差 67.59。

**同配置复测还顺带给出了此前缺失的"确定性"对照**：

```
$ python3 tools/dcu/s14_logprob_compare.py greedy-P4g3.jsonl greedy-P4g2.jsonl --label-a P4g3 --label-b P4g2
s14_greedy_long  256 256  2  0.52182 0.32859  2  2  '用'  '：'  0.562  0.078  0.078  248  11.297
# prompt 相同、temperature=0、配置逐字相同、两次独立启动 ⇒ **首个分歧仍在 token 2**
# 两侧间距 **0.078 / 0.562 nat** ⇒ **这套栈在引擎重启之间不是 bit 级确定性的**
```

#### 3.8.4 2×2 矩阵：并行度 × 编译模式（decode 中位 TPOT，ms/token，n=5）

| decode 中位 TPOT | **TP=2** | **TP=4** | TP=4 / TP=2 |
|---|---|---|---|
| **eager** | **113.51**（P2e2，`bench-P2e2-decode.log`，P99 117.47） | **126.83**（P4e，P99 132.68） | **1.117× 慢** |
| **graph** | **84.22**（P2，P99 84.38） | **49.66**（P4g2/P4g3 合并，n=10，极值 [49.46,49.86]） | **0.590×（快 1.70×）** |
| eager / graph | 1.348× 慢 | 2.554× 慢 | — |

口径全部 `maxlen 262144 / util 0.85 / MBT 2048 / n=5（graph-TP4 为 n=10）`，
卡位 TP=2 → `0,1`、TP=4 → `0,1,2,3`，`VLLM_DCU_USE_INT8_SDOT4=1`。
P4g2/P4g3 额外：`-cc.cudagraph_mode=PIECEWISE` + `--cudagraph-capture-sizes 1 2 4 8` + **需 16 GiB swap**。

**读法（重要）**：

* **eager 下 TP=4 反而更慢（1.12×）**：eager 的 decode 受**逐算子的启动/调度开销**支配，
  4 卡多出来的 all-reduce 通信没有回报 —— 这也解释了 S13 当年"TP=4 更慢"的印象，
  以及生产栈 121.43 ms 与我们 P4e 的 126.83 只差 4.4%。
* **graph 下 TP=4 快 1.70×**：图捕获消掉了启动开销后，decode 变成**权重读取带宽受限**，
  4 路分片把每卡要读的权重降到 1/4，收益立刻体现。
* ⇒ **"这台机 TP=4 行不行"取决于模式**：eager 行（但没意义，比 TP=2 慢）；
  **graph 才让 TP=4 真正发挥（49.66 ms/token，比生产基线 121.43 快 2.44×）**。
* 但 **graph 下 TP=4 与 P2 之间仍混着"图配置不同"这一点**（P2 用默认 51+2 桶 FULL_AND_PIECEWISE，
  P4g 用 4 桶 PIECEWISE-only）——默认配置的 graph-TP=4 在本机**起不来**（§3.8.1 第 4 次 OOM），
  所以**无法在本机做出"同图配置、只变 TP"的干净对照**。这是本节最大的保留意见。

#### 3.8.5 任务 §3(4)（可选）：graph TP=4 + `VLLM_DCU_USE_INT8_SDOT4=0`，**不加 swap** → 仍 OOM

目的：区分"graph 模式多出来的宿主匿名内存"是**我们的 sdot4 权重重排**还是**编译/图执行路径**。
方法：把 `s14_run_one.sh` 的第 4 个位置参数 `SW` 置 0（环境里 `VLLM_DCU_USE_INT8_SDOT4=0`，
**仓库默认值仍为 OFF，未改任何代码**），其余与 P4g1 逐字相同，且**已 `swapoff`**（只有 4 GiB `/swap.img`）。

```
$ sudo -n swapoff /mnt/game/swapfile        # 先还原宿主（见 §3.9）
$ S14_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" READY_TIMEOUT=1500 \
    nohup tools/dcu/s14_run_one.sh P4g4 /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 4 0 \
    /mnt/game/s14/cache-27b-p4g4 8428 262144 0,1,2,3 decode \
    > /mnt/game/s14/hostlogs/orch-P4g4.log 2>&1 &
(Worker_TP0 pid=35749) INFO 10-10 05:27:27 [__init__.py:805] Selected **TritonInt8ScaledMMLinearKernel** for CompressedTensorsW8A8Int8
                       ← 证明我们的 Gfx906Sdot4 路径**确实被关掉了**
[orch14] ready=0 after 281s  2026-10-09T21:30:45+00:00
[orch14] done name=P4g4 ready=0
```

**第 5 次 OOM 的内核原文**（同一口径，`anon-rss` 量级不变）：

```
[Fri Oct  9 21:30:30 2026] oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=api-gateway.service,mems_allowed=0,
    global_oom,task_memcg=/system.slice/docker-5292b9242a3899f70c1139274c9ec43145b8096ccda8938544ce15b67314d34b.scope,
    task=VLLM::Worker_TP,pid=2870818,uid=0
[Fri Oct  9 21:30:30 2026] Out of memory: Killed process 2870818 (VLLM::Worker_TP)
    total-vm:41236272kB, anon-rss:7279776kB, file-rss:0kB, shmem-rss:88644kB, UID:0
```

**`/swap.img` 被 4 GiB 全部吃光后 OOM（`P4g4-monitor.log` 原文）**：

```
2026-10-09T21:29:53+00:00 32009 26828  297 4565 4095 29822164992 max 0 2  90 34786    ← swap 4095/4095 用满
2026-10-09T21:30:23+00:00 32009 30921  334  484 4095 29760598016 max 0 2  90 34786    ← avail 484 MB
2026-10-09T21:30:38+00:00 32009 24604 6562 6809 3917 23179681792 max 0 3  89 26089    ← oom_kill 2 → 3
```

**⇒ 结论：不是我们的 sdot4 权重路径。** 换用 `TritonInt8ScaledMMLinearKernel` 后匿名需求没有变小
（被 OOM 进程 `anon-rss` 7.28 GB，与前几次 6.76/7.09/7.21/7.28 GB 同档），一样 `global_oom`。
所以 graph 模式 TP=4 多出来的宿主内存在**编译/切分执行路径本身**，
任务 §3(4) 的假设被否证，**且如 Lead 预判，这不改变主结论**。

### 3.9 收尾

#### 3.9.1 宿主状态还原（Lead 硬性要求：swap 前后原文）

```
$ sudo -n swapon --show                 # swapoff 前
NAME               TYPE SIZE   USED PRIO
/swap.img          file   4G  496.1M   -2
/mnt/game/swapfile file  16G   45.1M   -3

$ sudo -n swapoff /mnt/game/swapfile    # rc=0
$ cat /proc/swaps                       # swapoff 后（原文）
Filename                                Type            Size            Used            Priority
/swap.img                               file            4194300         506852          -2

$ free -m
               total        used        free      shared  buff/cache   available
Mem:           32009        3056       27640           0        1311       28501
Swap:           4095         494        3601
```

⇒ `/mnt/game/swapfile` 的 45.1 MB 已全部换回，**未保留**（如 Lead 认为值得长期保留，
需重新 `sudo -n swapon /mnt/game/swapfile` —— 本次按"必须还原"执行）。

#### 3.9.2 引擎清理与终态核验（2026-10-09T21:31Z）

```
$ pgrep -af "vllm|torchrun"
(none)

$ for p in $(pgrep VLLM); do st=$(awk '{print $3}' /proc/$p/stat); [ "$st" = Z ] && continue; ...; done
live count=0        （zombies: 86 —— 历史遗留，state=Z，不占内存）

$ hy-smi --showmeminfo vram | grep "Used Memory"
DCU[0] 0    DCU[1] 0    DCU[2] 0    DCU[3] 0

$ free -m
Mem: 32009 3028 27799 0 1181 28536        Swap: 4095 495 3600

$ docker exec z100-s14 bash -c 'cat /sys/fs/cgroup/memory.events; cat /sys/fs/cgroup/memory.max'
low 0 / high 0 / max 0 / oom 0 / **oom_kill 3**
max                                        ← 容器上限仍为 max（Memory=0 口径未变）

$ docker inspect z100-s14 --format 'Memory={{.HostConfig.Memory}} MemorySwap={{.HostConfig.MemorySwap}}'
Memory=0 MemorySwap=0

$ ps -ef | grep -E "s14_monitor|s14_run_one|s14_smaps|s14_bench" | grep -v grep
(none)                                     # 全部编排 / 监控 / 采样进程已停止
```

> 注：各 `*-monitor.log` 末尾**没有** `# end=` 收尾行 —— 因为监控脚本是在所有引擎关停之后
> **被主动 kill 的**（避免空转每 15 s 打一行）；所有引擎运行期间的采样都已落盘。

**`oom_kill` 计数归属（本容器基线 = 0，终态 = 3）**：

| # | 时刻（UTC） | arm | 图配置 | swap | 落点 |
|---|---|---|---|---|---|
| 1 | 2026-10-09T19:52:22 | **P4** | 默认 `FULL_AND_PIECEWISE` 51+2 桶 | 无 | KV cache 16.99 GiB 后 5 s |
| 2 | 2026-10-09T20:57:05 | **P4g1** | `PIECEWISE` 4 桶 | 无 | KV cache 18.47 GiB 后 6 s |
| 3 | 2026-10-09T21:30:30 | **P4g4** | `PIECEWISE` 4 桶 + **sdot4=0** | 无 | `/swap.img` 4 GiB 吃光后 |

**成功且 `oom_kill=0` 增量的 arm**：P2（TP=2 graph，前任启动）、**P4e（TP=4 eager）**、
P2e / P2e2（TP=2 eager）、**P4g2 / P4g3（TP=4 graph + 16 GiB swap）**。
⇒ **`oom_kill` 增量与"是否 eager"以及"是否给 swap"一一对应**（见 §4 G1 / §6）。

#### 3.9.3 口径纪律核验：ACS 全程部署态（关闭），未 apply / 未 revert

```
$ sudo -n python3 tools/dcu/dcu_p2p_acs.py show
父链上的桥 (12 个带 ACS):
  BDF          cap_off  ACS_Ctl   enabled bits
  0000:00:03.1 0x2a0    0x0000    (none)
  0000:28:00.0 0xf24    0x0000    (none)
  0000:28:04.0 0xf24    0x0000    (none)
  0000:28:08.0 0xf24    0x0000    (none)
  0000:2a:10.0 0xf24    0x0000    (none)
  0000:2d:00.0 0x2a0    0x0000    (none)
  0000:30:00.0 0xf24    0x0000    (none)
  0000:32:00.0 0x2a0    0x0000    (none)
  0000:35:00.0 0xf24    0x0000    (none)
  0000:35:10.0 0xf24    0x0000    (none)
  0000:37:00.0 0x2a0    0x0000    (none)
  0000:3a:00.0 0x2a0    0x0000    (none)
⇒ 12×`0x0000`，与任务书收尾口径一致

$ systemctl is-active dcu-p2p-acs
active
$ systemctl status dcu-p2p-acs --no-pager
Active: active (exited) since Fri 2026-10-09 18:29:46 UTC; 3h 1min ago
```

`Active since 18:29:46Z` **早于本步全部实验（19:32Z 起）** ⇒ S14 全程 ACS 未变。

#### 3.9.4 未触碰的资产

```
$ docker inspect qwen38-vllm --format 'Status={{.State.Status}} ExitCode={{.State.ExitCode}} Memory={{.HostConfig.Memory}}'
Status=exited ExitCode=137 Memory=0        ← 与 S14 开始前逐字相同（生产容器全程未 start/kill/exec/commit）

$ docker inspect z100-s9 --format '{{.HostConfig.Memory}} {{.HostConfig.MemorySwap}} {{.State.Status}}'
34359738368 34359738368 running            ← 影子容器保持原状
```

`VLLM_DCU_USE_INT8_SDOT4` 的**仓库默认值仍为 OFF**（本轮全部通过 `docker exec -e` 注入，
未改 `vllm/envs.py`、未改 DTK 库、未改 `z100-build*`、未改 s11*/s12*/s13* 脚本）。

#### 3.9.5 回滚件

* `docker rm -f z100-s14`、`docker rmi z100-s9-snap-s14`；`z100-s9` 无需回滚（未被修改）。
* `/mnt/game/s14/**`（原始日志、`cache-27b-p2/-p4/-p4e/-p2e/-p4g1..4`、flip/greedy jsonl）为新增产出，
  **按纪律保留作原始证据**，可整目录删除。
* 新增脚本 `tools/dcu/s14_{bench_p2,stop_arm,smaps,smaps_loop}.sh` 与
  `s14_{arm_env,run_one}.sh` 的 eager/cudagraph 扩展开关，无其它代码改动。

#### 3.9.6 机读汇总与原始工件索引

`/mnt/game/s14/hostlogs/summary.tsv`（列：`name model TP SW maxlen devices ready ready_s
aot_load aot_recompile oom_kill timestamp`，**本步 8 个 arm 全在里面**）：

```
P4    /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 4 1 262144 0,1,2,3   0 944  0 0 1 2026-10-09T19:53:03+00:00
P4e   ... 4 1 262144 0,1,2,3   1 190  0 0 1 2026-10-09T20:42:57+00:00
P2e   ... 2 1 262144 0,1     1 150  0 0 1 2026-10-09T20:51:18+00:00
P4g1  ... 4 1 262144 0,1,2,3   0 310  0 0 2 2026-10-09T20:57:44+00:00
P4g2  ... 4 1 262144 0,1,2,3   1 350  0 0 2 2026-10-09T21:06:09+00:00
P4g3  ... 4 1 262144 0,1,2,3   1 351  0 0 2 2026-10-09T21:19:13+00:00
P2e2  ... 2 1 262144 0,1     1 130  0 0 2 2026-10-09T21:24:57+00:00
P4g4  ... 4 0 262144 0,1,2,3   0 281  0 0 3 2026-10-09T21:31:11+00:00
```

原始工件（`/mnt/game/s14/`，共 26 GB）：

| 类别 | 文件 |
|---|---|
| 引擎日志 | `hostlogs/{P2,P2e,P2e2,P4,P4e,P4g1,P4g2,P4g3,P4g4}.log` |
| 编排日志 | `hostlogs/orch-{P2,P4e,P2e,P2e2,P4g1..4}.log` |
| 资源时间序列 | `hostlogs/{P2,P4}-monitor-part1.log`（前任，20:02 被回收）、`hostlogs/{P2b,P4e,P2e,P4g1..4}-monitor.log` |
| 内存构成 | `hostlogs/smaps-P2-*.txt`、`smaps-P4e.txt`（44 万行，60 s ×45 采样，含加载的 `.abi3.so` 与 md5） |
| bench 原始 | `/mnt/game/s14/bench-{P2,P4e,P2e2,P4g2,P4g3}-{decode,ttft,ttft8k,conc8}.log` |
| 聚合 JSON | `/mnt/game/s14/openai-chat-infqps-*.json` |
| token/logprob | `/mnt/game/s14/{greedy,flip}-{P2,P2e,P4e,P4g2,P4g3}.jsonl` |
| cgroup | `cgroup-events-<ARM>-{before,after}.txt` |
| 编译缓存 | `cache-27b-{p2,p2e,p4,p4e,p4g1,p4g2,p4g3,p4g4}/` |
| 归档自身 | `docs/result/z100-14-27b-tp4-result.md`（md5 `2124fc9a…` 为 §3.9.6 写入前；最终以文件为准） |

## 4. 门禁判定

| # | 门禁 | 判据 | 判定 | 证据 |
|---|---|---|---|---|
| **G1** | 复现 | 我们的树（`=1`）在 TP=4 下成功起服务就绪、`/v1/models` 可达、≥1 次 8K prefill + 一条完整 greedy 成功、进程存活、**`oom_kill=0`** | ✅ **通过（主臂 P4e，eager）**<br>⚠️ **graph 模式默认配置不通过** | §3.4.1：`ready=1 after 190s`、`oom_kill` 保持 1（未增长）、greedy 256 token 正常中文、ttft8k 12.96 s；§3.8.1：graph 默认配置第 1/4 次 OOM；§3.8.2：graph + 16 GiB swap 后 `ready=350s`、`oom_kill` 不增长 |
| **G2** | 数值 | TP=4 与 TP=2 同 greedy prompt 输出**文本一致或仅贴边翻转**（附分叉点 + top1–top2 间距 nat）+ logprob 对比；不一致须给机制判断（数值性 vs 调度性） | ❌ **不通过（字面）**<br>✅ 机制已判定为**数值性/系统性** | §3.6：0/8 prompt token-identical；首个分歧点间距 1.17–4.75 nat > 0.5 nat 阈值；**噪声底实测 0.1–0.6 nat**（P4g3 vs P4g2 同配置复测）；后端/kernel 逐项排除；NaN 检查通过（无 NaN/Inf） |
| **G3** | 性能 | TP=4 的 decode(TPOT 中位 n≥4)、TTFT(in≈3800)、conc8；与 P2 同口径 + S11m/S11l 84.00–84.68 + 生产 121.43（不重测）对比 | ✅ **通过** | §3.5.4 表 + §3.8.4 2×2 矩阵：decode n=5（graph-TP4 合并 n=10）、TTFT n=4、conc8 n=16，每行标注 并行度/模式/maxlen/util/n，给 P90/P99 区间 |
| **G4** | 口径纪律 | 同 fabric 状态（ACS 全程关闭）、同 cache vintage、n≥4、给区间；不得混口径 | ✅ **通过** | §3.9.3（12×`0x0000`，unit active since 18:29:46Z 早于全部实验）；§3.5.4/§3.8.4 每行独立标口径；无一行混用 |
| **G5** | 归档 | 本文件 8 小节齐全、含原始命令/原始日志路径/md5/回滚件 | ✅ **通过** | 本文件；原始日志 `/mnt/game/s14/**`；扩展 md5 §3.7；回滚件 §3.9.5 |

> **G1 的口径要点**：任务书 G1 写的是"TP=4 下成功起服务就绪 + `oom_kill=0`"。
> 本步的结论分两句：
> * **TP=4 + eager（P4e）**：完全满足 G1（190 s 就绪、`oom_kill` 未增长、端到端成功）；
> * **TP=4 + graph（默认图配置）**：**不满足**（两次 OOM）；**只有把 16 GiB swap 挂上、
>   并把捕获桶砍到 `[1,2,4,8]` 时**才满足（P4g2/P4g3，350 s 就绪、`oom_kill` 未增长）。
> 所以"本机能跑 TP=4"这句必须带模式与前置条件。

## 5. 未完成项 / 阻塞与原因

| # | 项 | 状态 | 原因 / 已有证据 |
|---|---|---|---|
| 1 | **G2 的"文本一致或仅贴边翻转"** | ❌ 未达成 | 0/8 prompt token-identical，首个分歧间距 1.17–4.75 nat > 0.5 nat 阈值。**非能力问题**：机制已判定为数值性、噪声底已实测（§3.6.3）。要"达成"只能靠**换 prompt**（找一个不是刀刃的 prompt），而不是解决问题 |
| 2 | 同引擎内 greedy 重复性 | ⚠️ 未做 | 已用**同配置跨引擎重启复测**（P4g3 vs P4g2，§3.8.3）等效替代：首个分歧 token 2、间距 0.078/0.562 nat ⇒ 噪声底 0.1–0.6 nat。同引擎内重复可再收紧，但需重开引擎 150 s |
| 3 | 参考真值（哪个输出"更正确"） | ⚠️ 不可得 | 本机**无法做 TP=1 参照**：28.76 GiB 权重 > 单卡 32 GiB 显存，装不下；vendor 栈无法逐 token 对照（版本/tokenizer 归一化不同） |
| 4 | **P4（graph 默认图配置）就绪** | ❌ 不可达 | 第 1 次 OOM（§3.3.1）。这是**结论本身**（本机 31 GiB RAM 装不下 graph-TP=4 的默认配置），不是未完成的工作 |
| 5 | `--cudagraph-capture-sizes` 之外的进一步降 footprint 调参 | 🚫 按 Lead 口径**主动停止** | §3.8.1 证明"图池不是主因"（1.67→0.22 GiB 仍 OOM），调参方向已失效；Lead 亦要求"立即停、不再调参" |
| 6 | P2e/P2e2 只测了 greedy/flip/decode | ⚠️ 有意的范围裁剪 | 这两个 arm 是为 **G2 的因子分离**（§3.6.0）与 **2×2 decode 矩阵**（§3.8.4）而设，其余指标对结论无增量 |
| 7 | vendor 栈（S13）的 TTFT / conc8 | 不适用 | 任务书明确"**n=4，不要重测**"，只有 121.43 decode 一个数可用 |
| 8 | 硬约束 13 / §9.4 / 性能方案 §1.3 的文字更新 | 未做（**越权**） | 这三份文档由 Lead 维护；本档提供违反登记（§3.4.1）与失效依据，供 Lead 更新 |
| 9 | ACS 切换对照 | 未做（**按纪律**） | 任务书要求"全程保持部署态，不 apply/不 revert"；§3.9.3 给出 12×`0x0000` 原文 |
| 10 | `/mnt/game/swapfile` 长期保留 | 未保留（**按纪律**） | 任务书要求收尾还原；已 `swapoff` 并给前后 `/proc/swaps` 原文（§3.9.1）。若 Lead 认为长期挂 swap 值得，重新 `swapon` 即可 |

## 6. 对后续步骤的影响

1. **"本机 TP=4 是否可行"必须分两句回答（本步的核心更正）**：
   * **eager：可行**（P4e，190 s 就绪、`oom_kill` 未增长、decode 126.83 ms/token，
     与生产 vendor 基线 121.43 只差 4.4%）。这也正是用户"我在当前机器上跑过 TP=4"的那次——
     `serve_27b_tp4.sh` 默认 `ENFORCE_EAGER=1`。
   * **graph：默认图配置不可行**（3 次 `global_oom`：P4 默认 51 桶 2 次口径、P4g1 4 桶 1 次口径），
     **加 16 GiB swap 后才可行**（P4g2/P4g3），并且它是**本步最快的 arm（49.66 ms/token）**。
2. **S13 的"生产 TP=4 = 121.43 ms/token 比我们 TP=2 graph 84 慢"应重新解读**：
   121.43 是 **eager** 口径。本步 2×2 矩阵（§3.8.4）显示 **eager 下 TP=4 比 TP=2 还慢 1.12×**
   （113.51 → 126.83），而 **graph 下 TP=4 比 TP=2 快 1.70×**（84.22 → 49.66）。
   ⇒ **"TP=4 更慢"是 eager-vs-graph 的口径错配造成的，不是 TP 的锅。**
   后续任何"并行度值不值"的比较**必须固定编译模式**。
3. **新的独立问题（建议立 S14b）**：在**同一 TP** 下，`CompilationMode.VLLM_COMPILE`（graph）
   与 `CompilationMode.NONE`（eager）会产生**不同的 token 流**——
   TP=2 时首 token 就分歧（一侧 `'The'` 有 4.75 nat 把握，另一侧选 `'用户'`），
   TP=4 时则只在 0.078 nat 的近边上分歧。**graph 模式改变了输出质量**（P2 退化成英文重复循环）。
   这与 S7b/S7c 修掉的 NaN 是两回事，**建议在把 graph 口径用于任何质量/精度评测之前先定位**。
4. **对 P6 的输入**：`VLLM_DCU_USE_INT8_SDOT4=0` **不会**降低宿主匿名需求
   （§3.8.5，第 5 次 OOM，`anon-rss` 7.28 GB 同档）⇒ **多出来的宿主内存在编译/切分执行路径**，
   不是我们的 sdot4 权重重排。P6 若想省内存，方向应是**编译产物/图执行路径**，不是量化 kernel。
5. **swap 的角色已定量**：graph-TP=4 启动期换出 **7.78 GB**、`available` 只剩 1.0–1.9 GB，
   decode 仍达 49.66 ms/token；引擎一关 swap 立刻回落 0.53 GB ⇒ 被换出的是**加载期 staging/冷页**，
   **权重本体在显存**（§3.8.2）。但**余量极小**（启动期一度只剩 507 MB），
   长时间/高并发生产不适合照搬，建议后续单独评估。
6. **对本机最优并行度的判断**：**TP=4 + graph + 4 桶捕获 + 16 GiB swap = 49.66 ms/token**
   是测到的最好口径（比生产基线快 **2.44×**），且 conc8（41.13 tok/s）与 8K prefill（12.97 s）
   也同时最好。**保留意见**：它的图配置与 P2 的默认配置不同，且**默认图配置在本机起不来**，
   所以"TP=4 更强"里有"图配置"的贡献无法在本机剥离。

## 7. 复现方式

前置：`docker inspect z100-s14 --format '{{.HostConfig.Memory}}'` = `0`；
`pgrep -af "vllm|torchrun"` 无他人实验；`hy-smi --showmeminfo vram` 4 卡全 0。

```bash
cd /home/lyh/z100-vllm
MODEL=/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8

# ---- 0) 每个 arm 开跑前的卡位/内存留档 ----
date -u +%Y-%m-%dT%H:%M:%SZ; pgrep -af "vllm|torchrun"; hy-smi --showmeminfo vram; free -g

# ---- 1) P2：TP=2 / graph（若已在跑则**不要重启**，只等就绪再 bench）----
S14_CT=z100-s14 nohup tools/dcu/s14_run_one.sh P2 $MODEL 2 1 \
  /mnt/game/s14/cache-27b-p2 8421 262144 0,1 \
  greedy,flip,ttft8k,decode,ttft,conc8 > /mnt/game/s14/hostlogs/orch-P2.log 2>&1 &
S14_CT=z100-s14 tools/dcu/s14_monitor.sh P2 5400 15 &     # 15 s 采样
# 接管已在跑的 P2（不重启）：等就绪 → greedy/flip/bench
tools/dcu/s14_bench_p2.sh

# ---- 2) 主臂 P4e：TP=4 + eager（**违反硬约束 13，见 §3.4.1 登记**）----
S14_CT=z100-s14 S14_EAGER=1 S14_FLIP_TOKENS=128 S14_GREEDY_TOKENS=256 READY_TIMEOUT=3000 \
  nohup tools/dcu/s14_run_one.sh P4e $MODEL 4 1 \
  /mnt/game/s14/cache-27b-p4e 8422 262144 0,1,2,3 \
  greedy,flip,decode,ttft,ttft8k,conc8 > /mnt/game/s14/hostlogs/orch-P4e.log 2>&1 &
S14_CT=z100-s14 tools/dcu/s14_monitor.sh P4e 5400 15 &
S14_CT=z100-s14 tools/dcu/s14_smaps_loop.sh P4e 45 60 &   # 任务 §3(3)：每 worker 匿名/文件构成

# ---- 3) G2 因子分离用的 control 臂：TP=2 + eager ----
S14_CT=z100-s14 S14_EAGER=1 READY_TIMEOUT=1200 nohup tools/dcu/s14_run_one.sh P2e $MODEL 2 1 \
  /mnt/game/s14/cache-27b-p2e 8424 262144 0,1 greedy,flip > /mnt/game/s14/hostlogs/orch-P2e.log 2>&1 &
S14_CT=z100-s14 S14_EAGER=1 READY_TIMEOUT=1200 nohup tools/dcu/s14_run_one.sh P2e2 $MODEL 2 1 \
  /mnt/game/s14/cache-27b-p2e 8427 262144 0,1 decode > /mnt/game/s14/hostlogs/orch-P2e2.log 2>&1 &

# ---- 4) P4g1：graph TP=4 零宿主改动降 footprint（**预期 OOM，这就是结论**）----
S14_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" S14_GREEDY_TOKENS=256 \
  READY_TIMEOUT=2400 nohup tools/dcu/s14_run_one.sh P4g1 $MODEL 4 1 \
  /mnt/game/s14/cache-27b-p4g1 8423 262144 0,1,2,3 decode,greedy \
  > /mnt/game/s14/hostlogs/orch-P4g1.log 2>&1 &

# ---- 5) P4g2/P4g3：挂 16 GiB swap 后重试同一 arm（**可复现：350/351 s，49.46/49.86 ms**）----
cat /proc/swaps                                   # BEFORE 原文
sudo -n swapon /mnt/game/swapfile                 # rc=0
cat /proc/swaps; free -m                          # AFTER 原文
for N in P4g2 P4g3; do
  S14_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" S14_GREEDY_TOKENS=256 \
    READY_TIMEOUT=3600 nohup tools/dcu/s14_run_one.sh $N $MODEL 4 1 \
    /mnt/game/s14/cache-27b-$N 8425 262144 0,1,2,3 decode,ttft,ttft8k,conc8,greedy \
    > /mnt/game/s14/hostlogs/orch-$N.log 2>&1 &
  # 等该 arm 跑完再起下一个（端口/卡位互斥）；P4g3 用 8426
done
# 硬停：swapon 后 60 min 未 ready ⇒ 杀掉记"未就绪"；ready 则 decode TPOT 必须 ≤84.68 ms/token
sudo -n swapoff /mnt/game/swapfile                # 收尾还原

# ---- 6) P4g4（可选）：graph TP=4 + SDOT4=0，**不加 swap**（第 4 个位置参数 = SW = 0）----
sudo -n swapoff /mnt/game/swapfile
S14_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" READY_TIMEOUT=1500 \
  nohup tools/dcu/s14_run_one.sh P4g4 $MODEL 4 0 \
  /mnt/game/s14/cache-27b-p4g4 8428 262144 0,1,2,3 decode \
  > /mnt/game/s14/hostlogs/orch-P4g4.log 2>&1 &

# ---- 7) 数值对比（G2）----
python tools/dcu/s14_logprob_compare.py /mnt/game/s14/greedy-P4e.jsonl  /mnt/game/s14/greedy-P2.jsonl   --label-a TP4e --label-b TP2g
python tools/dcu/s14_logprob_compare.py /mnt/game/s14/greedy-P2e.jsonl  /mnt/game/s14/greedy-P2.jsonl   --label-a TP2e --label-b TP2g
python tools/dcu/s14_logprob_compare.py /mnt/game/s14/greedy-P4e.jsonl  /mnt/game/s14/greedy-P2e.jsonl  --label-a TP4e --label-b TP2e
python tools/dcu/s14_logprob_compare.py /mnt/game/s14/greedy-P4g2.jsonl /mnt/game/s14/greedy-P2.jsonl   --label-a TP4g --label-b TP2g
python tools/dcu/s14_logprob_compare.py /mnt/game/s14/greedy-P4g2.jsonl /mnt/game/s14/greedy-P4e.jsonl  --label-a TP4g --label-b TP4e
python tools/dcu/s14_logprob_compare.py /mnt/game/s14/greedy-P4g3.jsonl /mnt/game/s14/greedy-P4g2.jsonl --label-a P4g3 --label-b P4g2  # 同配置复测

# ---- 8) 每 worker 内存构成（任务 §3(3)）----
tools/dcu/s14_smaps.sh P4e                 # /mnt/game/s14/hostlogs/smaps-P4e.txt

# ---- 9) 停引擎 / 收尾 ----
tools/dcu/s14_stop_arm.sh <NAME>            # 内含 `pkill VLLM`（硬约束 4）
sudo -n python3 tools/dcu/dcu_p2p_acs.py show     # 复核 12×0x0000
docker inspect qwen38-vllm --format 'Status={{.State.Status}} ExitCode={{.State.ExitCode}}'
```

**关键口径提醒（写脚本时最容易踩的两个坑）**：

* `--cudagraph-capture-sizes` 是 argparse `nargs='+'` ⇒ 必须传成
  `--cudagraph-capture-sizes 1 2 4 8`（**不能**加引号成单个参数），脚本里用未加引号的 `$S14_CG_SIZES` 展开；
* 杀 worker 只能 `pkill VLLM`（worker 的 `comm` 被改成 `VLLM::Worker_TP`、`/proc/<pid>/cmdline` 为空）。

## 8. 参考

- 用户提供的生产配方：`/home/lyh/server/serve_27b_tp4.sh`（关键：默认 `ENFORCE_EAGER=1`）
- S12：[`z100-12-p2p-acs-impact-result.md`](z100-12-p2p-acs-impact-result.md) §3.4.1（**已被本档更正**：
  真因不是 cgroup 上限，见 §3.3.1/§0.2 更正块）
- S13：[`z100-13-custom-allreduce-result.md`](z100-13-custom-allreduce-result.md) §3.5（生产栈 TP=4 **eager** 基线 121.43）
- S11l：[`z100-11l-sdot4-reacceptance-result.md`](z100-11l-sdot4-reacceptance-result.md)（TP=2 历史 84.00–84.68）
- S7b：[`z100-07b-eager-nan-result.md`](z100-07b-eager-nan-result.md)（eager NaN 根因）
- S7c：[`z100-07c-packed-silu-csrc-fix-result.md`](z100-07c-packed-silu-csrc-fix-result.md)（csrc 根修 = 本档 P4e 的依据）
- 内核原文：`sudo -n dmesg -T`，本档 §3.3.1 / §3.8.1 / §3.8.5（S14 共 3 次 `global_oom` + S9b/S12 共 2 次）
- 原始日志：`/mnt/game/s14/hostlogs/`（`P2/P2b/P2e/P2e2/P4/P4e/P4g1..4[-monitor].log`、
  `smaps-*.txt`、`summary.tsv`）；聚合 JSON `/mnt/game/s14/openai-chat-infqps-*.json`；
  token/logprob 记录 `/mnt/game/s14/{greedy,flip}-*.jsonl`
- ACS 工具：[`tools/dcu/dcu_p2p_acs.py`](../tools/dcu/dcu_p2p_acs.py)（本步只用 `show`）

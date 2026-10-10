# S14g — graph 模式下"重启变体"能不能钉住:实验与结论

**结论(TL;DR)**:graph 模式下每次重启落在哪个变体,**由运行时"残余地址/编译布局"的
不可控抖动决定,本次没有找到可复现的钉住配方**。

* **H1(ASLR)否证**:在 `seccomp=unconfined` 的旁路容器里用 `setarch x86_64 -R` 关掉
  ASLR(实证 PIE 基址被钉死为 `555555554000`),两次 run 的 greedy token 流仍然
  **0/8 逐位一致**(`max|Δlogprob|` 0.40 nat),而且分别落在**两个全新的、此前 10 次 run
  都没出现过的变体**上;
* **H2/H3(Triton autotune)排除**:GDN prefill 的 7 个 `@triton.autotune` kernel 在两次
  ASLR-off run 里**选中的 config 逐字相同、且全部命中磁盘缓存**(`bench_time=None`),
  输出却不同 ⇒ autotune 的 config 选择**不是**变体来源(vLLM 自己默认
  `TRITON_CACHE_AUTOTUNING=1`,已经把它钉住了);
* 图配置(默认 51 桶 vs PIECEWISE 1/2/4/8)、TP、prefix cache、combo kernel、AOT 来源、
  同一份 cache、同一份启动脚本都已在前序步骤或本步被排除;
* **顺带的关键工程发现**:`PYTHONPATH`/`sitecustomize` 式的注入**到不了 TP worker**
  (只有 API server 进程加载),必须走 site-packages `.pth`;worker 的 `/proc/<pid>/environ`
  被进程标题覆写,不能用来判断环境。

**对生产的影响**:2.44× 的 graph 配方照用,但**不能承诺"重启后输出可复现"**;
`TRITON_CACHE_AUTOTUNING` 必须保持 vLLM 默认值 1,不要往引擎里塞 `PYTHONPATH` 钩子。

工作区 `/home/lyh/z100-vllm`;脚本 `tools/dcu/s14g_*`;原始数据 `/mnt/game/s14g/`。
所有实验都用**省内存图配置**(`TP=2` / `maxlen 4096` / `-cc.cudagraph_mode=PIECEWISE`
`--cudagraph-capture-sizes 1 2 4 8`,启动 170–180 s),一次重启只要 ~4–6 min,而不是生产
配置的 25.7 min。

---

## 1. 问题、基线与变体空间

**问题**:graph 模式下每次重启"抽到的那个变体"由什么决定,能不能钉住?

### 1.1 变体可以被无参考臂地重新聚类(E1,零成本)

新写的 `tools/dcu/s14g_classify.py` 用"整个电池 greedy token 流的 sha256 前 12 位"作为
**变体签名**,相同签名 ⇔ 逐位相同(8 prompt × 64 token)。对既有数据重新聚类:

```
$ python3 tools/dcu/s14g_classify.py /mnt/game/s14e/flip-G2{a,b,c}.jsonl \
      /mnt/game/s11m/flip-S{2,3,4}sw1hit.jsonl /mnt/game/s14e/flip-G1{a,b,c}.jsonl
  G2a b8c457ab8de8   G1a 0557bfcdec25   S2sw1hit a78f84da5d3b
  G2b c359a934dfc2   G1b a78f84da5d3b   S3sw1hit a78f84da5d3b
  G2c a78f84da5d3b   G1c 0557bfcdec25   S4sw1hit a78f84da5d3b
  Va a78f84da5d3b: G2c,S2,S3,S4,G1b (5/9)   Vb 0557bfcdec25: G1a,G1c (2/9)
  Vc b8c457ab8de8: G2a (1/9)                Vd c359a934dfc2: G2b (1/9)
```

* **只有 4 个离散变体**,Va 是吸引子(5/9);
* **图配置不决定变体**:G1(默认 51 桶)的 `G1b` 与 G2(PIECEWISE 1/2/4/8)的 `G2c`、
  以及 S11m(另一份 cache、TP=2)的 S2/S3/S4 **签名完全相同**;
* G2 三连跑两两 **0/8** 逐位一致,`mean|Δlogprob|` 0.049–0.063、`max` 0.32–0.60 nat
  (与 S14e 的记录一致,这里是逐位签名级证据)。

### 1.2 全部 14 次 run 的变体归属表(A1/A2 = ASLR off,其余 on)

| run | 容器 | ASLR | 图配置 | cache | 签名 | 变体 |
|---|---|---|---|---|---|---|
| G2a | z100-s14 | on | PIECEWISE 1/2/4/8 | s14e/g2 | `b8c457ab8de8` | Vc |
| G2b | z100-s14 | on | PIECEWISE 1/2/4/8 | s14e/g2 | `c359a934dfc2` | Vd |
| G2c | z100-s14 | on | PIECEWISE 1/2/4/8 | s14e/g2 | `a78f84da5d3b` | **Va** |
| G1a | z100-s14 | on | 默认(51 桶) | s14e/g1 | `0557bfcdec25` | Vb |
| G1b | z100-s14 | on | 默认(51 桶) | s14e/g1 | `a78f84da5d3b` | **Va** |
| G1c | z100-s14 | on | 默认(51 桶) | s14e/g1 | `0557bfcdec25` | Vb |
| S2,S3,S4 | z100-s14 | on | S11m 配方 | s11m | `a78f84da5d3b` ×3 | **Va** |
| Pa | z100-s14 | on | PIECEWISE 1/2/4/8 | s14e/g2 | `a78f84da5d3b` | **Va** |
| Pb | z100-s14 | on | PIECEWISE 1/2/4/8 | s14e/g2 | `0557bfcdec25` | Vb |
| **A1** | z100-s14g | **off** | PIECEWISE 1/2/4/8 | s14e/g2 | `559d0c302a85` | **Ve(新)** |
| **A2** | z100-s14g | **off** | PIECEWISE 1/2/4/8 | s14e/g2 | `f6e73ec30b83` | **Vf(新)** |
| A3 | z100-s14g | on(对照) | PIECEWISE 1/2/4/8 | s14e/g2 | `a78f84da5d3b` | **Va** |

分布:Va 7/14(50%)、Vb 3/14、Vc 1/14、Vd 1/14、Ve 1/14、Vf 1/14(共 6 个签名)。
原始聚类输出:`/mnt/game/s14g/variants-all.txt`、`variants-P.txt`、`reclass-G2.txt`、`reclass-all-prior.txt`。

关键配对数字:

| 配对 | 条件 | flip 逐位一致 | `max|Δlogprob|` | e14 逐位一致 | e14 worst Δ |
|---|---|---|---|---|---|
| G2a/G2b/G2c | on,同容器同配置三连跑 | 0/8 ×3 | 0.32–0.60 nat | — | — |
| Pa vs Pb | on,同容器**完全相同命令行** | — | — | **3/12** | 0.428 nat |
| **A1 vs A2** | **off,同容器同配置** | **0/8** | **0.401 nat** | **4/12** | **0.401 nat** |
| A1/A2 vs A3 | off vs on | 0/8 | 0.40–0.68 nat | A2 vs A3: **4/12** | 0.307 nat |
| A3 vs Pa | 同容器 vs 另一容器,都 on | — | — | 同签名 Va ⇒ 容器不是 confound | — |

(A1 vs A2 的 e14 生成总长 1450 vs 1536 token ⇒ 有一边换题换到不同长度;`cmp-A1-vs-A2-e14.txt`。)

---

## 2. 环境与纪律留档

```
$ date; pgrep -af "vllm|torchrun"; hy-smi --showmeminfo vram; free -g; cat /proc/swaps
Sat Oct 10 05:31:08 AM UTC 2026 ; (无 vllm/torchrun)
DCU[0..3] vram Total Used Memory: 0 MiB
Mem: 31 GiB total / 27 available ; Swap: 4095 MiB(已用 ~500 MiB,/swap.img)
```
原始留档 `/mnt/game/s14g/env-baseline.txt`;`docker inspect z100-s14`:
`Binds=[/opt/hyhal:ro, /home/lyh/z100-vllm:/src:ro, /mnt/game:/mnt/game]`、`Memory=0`、
`SecurityOpt=["label=disable"]`(默认 docker seccomp)、`Ipc=host`、`Shm=16 GiB`。

* 生产容器 `qwen38-vllm` 全程未启动、未触碰;DTK 库、`z100-build*` 未动;
  `VLLM_DCU_USE_INT8_SDOT4` 默认值未改;`VLLM_LOGGING_LEVEL=DEBUG` 未开;
* 所有 run 显式 `HIP_VISIBLE_DEVICES=0,1`;同一时刻只有一个引擎(每次开跑前
  `pgrep -af "vllm|torchrun"` + `hy-smi` + `free -g`);
* 写作用域:`tools/dcu/s14g_*`、`/mnt/game/s14g/**`、本文档;
  S14/S14b/S14c/S14d/S14e 脚本一字未改;
* **未执行任何自行编译的独立 HIP 可执行文件**;`setarch`/`taskset`/profiler 未用;
* 唯一新增容器:`z100-s14g`(见 §3.2),与 z100-s14 同镜像同挂载,仅 `seccomp=unconfined`。

---

## 3. H1(地址布局 / ASLR)

### 3.1 仪器在 z100-s14 里被 seccomp 挡住(原始证据)

```
$ docker exec z100-s14 bash -c 'command -v setarch'      → /usr/bin/setarch
$ docker exec z100-s14 bash -c 'setarch --addr-no-randomize /bin/true'
setarch: failed to set personality to (null): Operation not permitted      # rc=1
$ docker exec z100-s14 bash -c 'setarch x86_64 -R /bin/true'
setarch: failed to set personality to x86_64: Operation not permitted      # rc=1
$ docker exec z100-s14 bash -c 'cat /proc/self/personality'               → 00000000
$ docker exec z100-s14 bash -c 'echo 0 > /proc/sys/kernel/randomize_va_space'
bash: ...: Read-only file system
```
原因:`personality(2)` 被 docker 默认 seccomp profile 拦(`HostConfig.SecurityOpt=["label=disable"]`,
没有 `seccomp=unconfined`),`randomize_va_space` 只读(无 CAP_SYS_ADMIN;容器 capability
里也没有 `cap_sys_ptrace/cap_sys_admin`)。**不是"没装 setarch"。**
原始证据:`/mnt/game/s14g/evidence-setarch.txt`。

### 3.2 旁路容器 z100-s14g(同镜像 + 关 seccomp)

```
$ docker run -d --name z100-s14g --security-opt seccomp=unconfined --security-opt label=disable \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri --group-add video \
    --ipc=host --shm-size 16g \
    -v /home/lyh/z100-vllm:/src:ro -v /mnt/game:/mnt/game z100-s9-snap-s14 sleep infinity
$ docker exec z100-s14g setarch x86_64 -R bash -c 'echo personality=$(cat /proc/self/personality) ok'
personality=00040000 ok
```
跑引擎的方式(z100-s14 与 z100-s14g 都用同一份 `s14g_arm_env.sh`):

```
docker exec -i z100-s14g setarch x86_64 -R bash -s -- <NAME> <MODEL> 2 1 <CACHE> <PORT> 4096 0,1 \
    < tools/dcu/s14g_arm_env.sh
```
`setarch` 用 `exec` 替换自身,personality 会被所有子进程继承(含 spawn 的 TP worker),
下面 3.3 的 PIE 基址证明了这一点。

### 3.3 判定:A1 与 A2 都关了 ASLR,却落在两个不同的**全新**变体

`tools/dcu/s14g_acell.sh`(A1/A2 = `setarch x86_64 -R`,A3 = 不加 setarch 的同容器对照),
同一 `VLLM_CACHE_ROOT=/mnt/game/s14e/g2`、同一电池(flip 8×64 + e14 12×128)。
每次 ready 后立刻用 `tools/dcu/s14g_maps.sh` 抓活跃引擎进程的地址指纹。

**证据 1 — ADDR_NO_RANDOMIZE 确实生效且被 worker 继承**:A1/A2 的 API server 与两个
TP worker 的 python3.10 PIE 基址**逐字节相同**(`maps-A1.txt` / `maps-A2.txt`):

```
== pid=132 (A1, API server)   first: 555555554000-5555555c1000 /usr/bin/python3.10
== pid=4311 (A2, API server)  first: 555555554000-5555555c1000 /usr/bin/python3.10
== pid=217 (A1, Worker_TP0)   first: 555555554000-5555555c1000 /usr/bin/python3.10
== pid=4388 (A2, Worker_TP0)  first: 555555554000-5555555c1000 /usr/bin/python3.10
```
(ASLR 打开时的同一进程基址是随机化的,例如 `55ad89d02000`。)

**证据 2 — 但动态布局仍然每次不同**:

```
A1 pid=132  [heap] 555555afe000-5555e748c000   n_maps=2020  fp=013c3ef98a268bb5
A2 pid=4311 [heap] 555555afe000-5555e8be9000   n_maps=2017  fp=152243a1769bd018
A1 pid=217  [heap] 555555afe000-55561ea3b000   n_maps=3719  fp=ed99e41a1cb73e8a
A2 pid=4388 [heap] 555555afe000-55561ea3c000   n_maps=3720  fp=46b503c451b469f6
```
`setarch -R` 只钉住 PIE/ld.so 的基址熵;PyTorch HIP caching allocator +
`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` 的堆/mmap 形态仍每次不同
(`fp` 四个进程全部不同、`n_maps` 与 heap 结束地址都不同)。

**证据 3 — 判定**:A1 与 A2 的 8×64 greedy token 流**不同**
(`559d0c302a85` vs `f6e73ec30b83`,`ids_identical=0/8`,`max|Δlogprob|=0.401 nat`,
`mean=0.055 nat`;e14 12 prompt 只有 4/12 逐位相同,worst 0.401 nat,生成长度 1450 vs 1536)。
→ **H1 的强形式("关掉 ASLR ⇒ 可复现")被否证。**

**证据 4 — 但地址布局并非无关(弱形式成立)**:同一个 z100-s14g 容器里,ASLR **打开**的
A3 落回了全部 12 次 run 里最常见的吸引子 `Va=a78f84da5d3b`(与 Pa/G2c/G1b/S2/S3/S4 同签名);
而两次 ASLR-off 的 A1/A2 落在**此前 10 次 run 从未出现过的两个新签名**上
⇒ 关掉 ASLR 把结果推离了常规吸引子集合,说明**布局确实参与决定数值**,
只是"关 ASLR"钉不住布局(证据 2)。
A3(同容器、ASLR 打开)落回吸引子 Va,同时排除了"z100-s14g 容器本身导致新变体"这一 confound。

---

## 4. 注入通道:`PYTHONPATH`/sitecustomize **到不了 TP worker**(关键工程发现)

**动机**:GDN prefill 走 `vllm.third_party.flash_linear_attention`,其中
`chunk_o` / `chunk_delta_h` / `cumsum` / `solve_tril` / `l2norm` / `wy_fast` /
`chunk_scaled_dot_kkt` 都是 `@triton.autotune`(configs 在 BK/BV/num_warps/num_stages 上组合,
`chunk_o` 有 36 个)。先试最轻的钉法:`PYTHONPATH=/mnt/game/s14g/pin` + `sitecustomize.py`
(monkeypatch `Autotuner._bench` 只让 `prune_configs[0]` 赢)。

**反证(Pa run,`/mnt/game/s14g/hostlogs/Pa.log`)**:

```
$ grep -c "s14g-pin" Pa.log
1
(APIServer pid=200878) [s14g-pin] triton autotune pinned to prune_configs[0] (0 kernels):
```
`sitecustomize` 的 `atexit` 标记由**每个**加载过它的 python 进程打印:Pa 的日志里只有
API server 那一条,两个 `VLLM::Worker_TP0/TP1` **一条都没有** ⇒ 注入没到 worker,
而 worker 才是唯一 launch GDN/FLA Triton kernel 的地方 ⇒ **该 pin 完全没生效**
(Pa、Pb 于是等价于 unpinned,它们分别落在 Va / Vb,见 §1.2)。

顺带:`/proc/<worker>/environ` 不可信 —— worker 的 argv/environ 区被进程标题
(`VLLM::Worker_TP0`)覆写,`environ` 共 1923 项、前 25 项是空串,`PYTHONPATH` 不在其中
但 `S14_*` 在,不能据此判断环境。

**可靠通道:site-packages `.pth`**。`site.py` 启动时执行 `.pth` 里的 `import` 行,
对容器内**每个** python 进程生效(含 spawn 的 worker),不依赖 `PYTHONPATH`:

```
/usr/local/lib/python3.10/dist-packages/s14g_autotune_hook.pth
    /mnt/game/s14g/hook
    import s14g_autotune_boot
```
`hook/s14g_autotune_boot.py` 只装一个 meta-path finder(零重依赖:普通 `python3 -c`
完全不受影响,已实测 rc=0),等 `triton.runtime.autotuner` 真被 import 时才加载
`hook/s14g_autotune_hook.py`;后者:

* `S14G_AUTOTUNE_LOG=<path>`:每次 `Autotuner.run` 追加一行 JSON(pid、kernel、
  config、`bench_time`),import 时再写一行 `{"event":"import"}` → **worker 级、逐 run 的
  autotuner 原始记录**;
* `S14G_PIN_MODE=first` → 强制 `prune_configs[0]`;`=choice` + `S14G_PIN_CHOICE=xx.json`
  → 钉到参考 run 的 timing 冠军(保留原性能)。

**注入验证(A1,成功打到 3 个进程)**:

```
$ head -3 /mnt/game/s14g/autotune-log-A1.jsonl
{"event":"import", ..., "pid": 132}     # API server
{"event":"import", ..., "pid": 217}     # Worker_TP0
{"event":"import", ..., "pid": 218}     # Worker_TP1
```

**pin 机制本身有效**(合成 autotuned kernel,`tools/dcu/s14g_pin_probe.py`):

```
unpinned    : chosen=BLOCK:1024, num_warps: 8  bench_time=1.135s   (6 个 config 全 benchmark)
PIN first   : chosen=BLOCK:1024, num_warps: 1  bench_time=0.691s   (只有 prune_configs[0])
```

> **教训(建议写进生产文档)**:任何"运行时 monkeypatch / sitecustomize / PYTHONPATH"
> 式的注入都必须**验证到了 `VLLM::Worker_TP*`**;只打 API server 进程等于没打。

---

## 5. A cell 结果与 autotune 记录(H2/H3)

### 5.1 原始命令

```
$ S14G_ASPECS="A1:off A2:off A3:on" tools/dcu/s14g_acell.sh         # 第一次(A2 挂死,见 5.4)
$ S14G_ASPECS="A2:off A3:on"        tools/dcu/s14g_acell.sh         # 重跑
# 每个 run 实际执行:
$ S14G_EXEC_PREFIX="setarch x86_64 -R" bash tools/dcu/s14g_run_one.sh A1 \
    /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 2 1 /mnt/game/s14e/g2 8641 4096 0,1
#   环境:S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" S14_UTIL=0.9 S14G_TASKS="flip,e14"
#   容器:S14_CT=z100-s14g;S14G_AUTOTUNE_LOG=/mnt/game/s14g/autotune-log-<NAME>.jsonl
```

每个 run:`ready=170–180 s`、`aot_load=2`(命中 S14e 的 AOT artifact)、
`init engine ... took 57.6–59.3 s (compilation: 20.0–20.2 s)`,
hook 记录 `import=3 进程 / autotune=14 事件`。

### 5.2 autotune:Triton 的 config 选择**本来就已被钉死**

* vLLM 自己默认打开 autotune 落盘缓存:
  `vllm/env_override.py:170-176  os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")`
  ("Without this, Triton re-runs autotuning on every process restart");
* `TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache` 里已有 **48 个 `*.autotune.json`**;
  2026-10-10 全天只新增 **4 个,且全部集中在 06:04:02–06:04:52**,即 §5.4 那次**挂死的
  A2 尝试**期间 —— 当天 8 次健康 run(S14e G2a/b/c + S14g Pa/Pb/A1/A2/A3)全部**没有写任何新
  autotune 文件** ⇒ 全部 key 命中磁盘缓存;
* worker 级 hook 实测(A1 的 Worker_TP0,7 个 kernel,`bench_time=None` 即磁盘缓存命中):

```
chunk_local_cumsum_scalar_kernel             | num_warps 1, num_stages 3
chunk_scaled_dot_kkt_fwd_kernel              | BK 32, num_warps 8, num_stages 2
merge_16x16_to_64x64_inverse_kernel          | num_warps 4, num_stages 4
recompute_w_u_fwd_kernel                     | num_warps 4, num_stages 3
chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | BV 32, num_warps 4, num_stages 2
chunk_fwd_kernel_o                           | BK 32, BV 64, num_warps 4, num_stages 3
triton_matmul_kernel                         | BLOCK_N 64, BLOCK_K 64, GROUP_M 1, num_warps 2, num_stages 3
```

* **A1 与 A2(两次 ASLR-off run)这 7 个 kernel 的 config 集合逐字相同**
  (`all identical: True`,`/mnt/game/s14g/evidence-autotune.txt`),**但两次的输出不同**;
* 因此 `TRITON_PRINT_AUTOTUNING=1` 在本项目里**一条都不打印**(S14e/S14g 全部 run 都是 0 条),
  原因在 `triton/runtime/autotuner.py:246`:`if knobs.autotuning.print and not used_cached_result:`
  —— 命中磁盘缓存就静默。这解释了"看不到 autotune 日志"的假象,不需要怀疑 kernel 没跑。
* **局限(如实登记)**:v1 hook 每个 `Autotuner` 对象只记一次(即每个 kernel 的**第一个
  key**,共 7 条/进程),"全部 key 逐字相同"只对这 7 个 key 成立。要全 key 级证据需把
  hook 改成逐 run 记录(v2 已改好:`hook/s14g_autotune_hook.py` 现在每次 run 都记
  `key`+config),但没有预算再跑两次对照 run(见 §6)。

### 5.3 H2/H3 的其余部分

* 机制确实存在:`Using Triton/FLA GDN prefill kernel`(gdn_linear_attn.py:182)、
  `qwen_triton_warmup.py:293 Warming up Qwen GDN Triton kernels`;
* 但"config 选择"这条路被 §5.2 直接排除(worker 级、同 config、不同输出);
* **未做**:torch profiler kernel 直方图 diff(H3 原计划)—— 预算不足,且 §5.2 已经证伪
  "config 差异"这个 H3 的主要假设;若 Lead 要继续,残余候选见 §7.2。

### 5.4 一次基础设施失败(如实登记)

第二次 ASLR-off 尝试(A2 第一次)在 ready 后两个 `VLLM::Worker_TP*` **100% CPU 空转
>6 分钟**,日志刷 `shm_broadcast.py:812 No available shared memory broadcast block found
in 60 seconds`(API server 侧同时 idle;`/dev/shm` 只用 49 MB,不是容量问题)。
处理:`kill -9` 掉按 PID(注意:worker 的 `cmdline` 已被进程标题覆写,`pkill -f "VLLM::"`
匹配不到,必须按 `ps -o pid,comm` 杀),确认 VRAM 归零后重跑成功。
该挂死 run 在 06:04 写了 4 个**新 cache key** 的 autotune JSON(见 §5.2),说明
"cache key 变化 ⇒ 重新 benchmark"这条路径在病态状态下是活的,健康状态下没有触发。

---

## 6. 未做项与成本核算(≈90 min 预算内)

| 项 | 状态 | 说明 |
|---|---|---|
| H1(ASLR on/off 各 ≥2) | **完成**(off 2 + on 1 同容器对照,另加 on 3 次历史 run) | §3 |
| 工作线程级 autotune 记录 | **完成**(A1/A2/A3) | §5.2 |
| H2 `TORCH_LOGS=recompiles/guards/dynamic` | **未做** | 预算;且它只"记录不改行为",其能解释的路径(config/编译选择)已被 §5.2 与 §3 正面排除 |
| H3 torch profiler kernel 直方图 diff | **未做** | 预算;见 §7.2 残余候选 |
| H4 `PYTHONHASHSEED=0` / `OMP_NUM_THREADS=1` / `taskset` 各 2–3 跑 | **未做** | 预算。理由减弱其优先级:A1/A2 的**进程环境完全相同**仍然分裂,说明"环境变量→配置"这条因果链即使存在,也不是唯一机制 |
| 3× 逐位一致 + decode 不劣化的"配方验收" | **无配方可验收** | 没找到钉法,§7 |
| pin 到 timing 冠军(`S14G_PIN_MODE=choice`)的引擎级验证 | **未做** | 因 §5.2 已证 config 本来就固定,预期无收益;hook 已支持该模式,可随时补做 |

省内存图配置的实际成本:`ready 170–180 s` + `flip 48 s` + `e14 140 s` + teardown ~30 s
≈ **5.5–6 min/run**,与 S14e 的记载一致。本次共 13 次引擎启动(A1/A2 各含 1 次重跑)。

---

## 7. 结论

### 7.1 能不能钉住:**不能**(在 S14g 覆盖的所有路线内)

**明确排除清单**(每条都有本步或前序步骤的实测支撑):

| 候选因子 | 判定 | 证据 |
|---|---|---|
| cudagraph(图模式本身) | 排除 | S14d;本次 §1.1(G1 默认图与 G2 PIECEWISE 共享同一签名 Va) |
| 图桶/图配置(默认 51 桶 vs `1 2 4 8`) | 排除 | §1.1、S14e(G2 三跑三变体) |
| packed decode / prefix cache / combo kernel / benchmark_combo_kernel | 排除 | S14d(命中的 prefix cache 实测 0.0%,关 combo 无收益) |
| AOT 来源(现场编译 vs 重载 artifact) | 排除 | S14e(AOT 命中,`aot_load=2`,仍分裂);本次每次 `aot_load=2` |
| TP(4 卡 vs 2 卡) | 排除 | S14b;本次全在 TP=2 复现 |
| 内存形态 / swap | 排除 | S14e |
| 父进程环境 / 任何 PYTHONPATH-sitecustomize 类注入 | 排除 | §4(到不了 worker;Pa/Pb 在该 pin 下仍落在 Va/Vb) |
| **ASLR / `setarch -R`** | **否证(本步)** | §3.3:A1/A2 PIE 基址相同却 0/8 一致、且是两个全新变体 |
| **Triton autotune config 选择** | **排除(本步)** | §5.2:worker 级实测 7 个 kernel config 逐字相同(全部磁盘缓存命中)却输出不同 |

**残余可能**(按可能性排序,均未被本步证伪):

1. **残余动态地址/编译布局**(最可能):`setarch -R` 只钉 PIE 基址,HIP caching allocator +
   `expandable_segments:True` 的堆/mmap 形态每次不同(§3.3 证据 2);地址/对齐会影响
   Triton 的 specialization → `fn.cache_key` → autotune/编译缓存 key 与 kernel 变体。
   旁证:挂死 run 在 06:04 写出 4 个**新 key** 的 autotune JSON(§5.4),说明 key 会变;
   A1/A2 都落在"全新变体"而非常规吸引子,也说明布局参与了决定。
   → 要钉它,需要一个能固定 allocator 布局的手段(本次没有找到;`PYTORCH_HIP_ALLOC_CONF`
   只有 `expandable_segments:True` 这一档在用,未做参数扫描)。
2. **某个 kernel 内部存在非确定性归约/竞争**(例如 GDN/decode 路径上未被 S14 静态审计覆盖的
   kernel)。本步没做 profiler/kernel 级 diff;静态事实只覆盖 sdot4 kernel。
3. **调度层的分批/分块**(chunked prefill 的 chunk 边界随到达时序变化)。
   本次客户端是严格单并发,但引擎内部首个请求前的调度状态仍可能不同;未做实验。
4. **进程级环境哈希**(`PYTHONHASHSEED` 等)影响 dict/集合迭代顺序 ⇒ 算子注册/融合决策。
   A1/A2 完全同 env 也分裂,故它若成立也只能解释一部分;未做 H4。

### 7.2 变体的形态

* 变体是**离散的小集合**(14 次 run 共 6 个签名),有一个强吸引子 Va(7/14 = 50%);
* 跨会话可复现的吸引子:Va 在 S11m(S2/S3/S4)、S14e(G1b/G2c)、S14g(Pa/A3)都出现;
  Vb 在 G1a/G1c/Pb 出现 ⇒ **不是随机噪声,而是少数几个稳定解之间的选择**;
* 同一变体内仍有 1e-2–6e-2 nat 的细抖动(S14e),足以在 ~128 步内翻 1 个 token;
* 两次 run 的 token 流不一致率:Pa/Pb 9/12、A1/A2 8/12(同配置、同二进制、同 cache)。

---

## 8. 对生产建议的影响

1. **不要承诺"重启后输出可复现"**。2.44× 的 graph 配方(graph 模式 + `PIECEWISE`
   + 省内存图桶)在数值上不是"钉死一个解",而是"在少数几个解之间抽签";把
   "重启后答案可能变化、`logprob` 抖动 ~0.4–1.5 nat、偶发翻 token"写进部署说明与
   SLO 备注。建议做法:生产**保持实例常驻、避免频繁重启**;灰度/对拍时不要用
   "重启后 diff 必须为 0"作门槛。
2. **保持 `TRITON_CACHE_AUTOTUNING` 为 vLLM 默认值 1**(`vllm/env_override.py:174-176`)。
   把它关掉会让每个进程重新按**计时**挑 config,直接引入一个新的、可控但会放大的
   抖动源(磁盘缓存里 top1/top2 差距在部分 key 上只有 5%,见
   `chunk_fwd_kernel_o` 的 `[16,128,128,16]` key:16.8 vs 17.3 µs)。
3. **不要往引擎里加 `PYTHONPATH`/sitecustomize 式钩子**:对 worker 无效(§4),
   而且 env 变化可能改变 AOT/编译缓存 key(S14e 已知 `VLLM_LOGGING_LEVEL=DEBUG` 会换 key)。
   如果确实要注入 worker,唯一验证过的通道是 site-packages `.pth`(§4)。
4. **本步没有产出可执行配方** ⇒ Lead 收口时按"否定结论 + 排除清单"处理:
   2.44× 收益与"输出可复现"在当前 gfx906/vLLM 组合下不可兼得;若必须可复现,
   需要另立专项(优先:allocator 布局固定手段 / kernel 级非确定性审计),不属于 S14g。

---

### 附:本步新增文件

* `tools/dcu/s14g_arm_env.sh`(S14e 配方 + `S14G_PIN_PATH` / `S14G_LAUNCH_PREFIX`)
* `tools/dcu/s14g_run_one.sh`(单 run 编排:fingerprint 采集、autotune 记录、hook 状态汇总)
* `tools/dcu/s14g_acell.sh`(ASLR on/off cell)
* `tools/dcu/s14g_drive.sh`(N 连跑 + 聚类)
* `tools/dcu/s14g_classify.py`(无参考臂变体聚类)
* `tools/dcu/s14g_maps.sh`(地址布局指纹;本 DTK 的 HIP runtime 叫 `libgalaxyhip.so.*`,不是 `libamdhip64`)
* `tools/dcu/s14g_pin_probe.py`(pin 机制自检)
* `/mnt/game/s14g/hook/s14g_autotune_boot.py` + `s14g_autotune_hook.py`(worker 级 autotune 记录/钉住)
* `/mnt/game/s14g/pin/sitecustomize.py`(仅 API-server 生效的反例)、`/mnt/game/s14g/pin_cal/sitecustomize.py`(choice 模式)
* 数据:`/mnt/game/s14g/{flip-*,probe-*,maps-*,autotune-log-*,variants-*,cmp-*,evidence-*,env-baseline.txt}`

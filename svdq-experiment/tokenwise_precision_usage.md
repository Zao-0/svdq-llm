# tokenwise_precision.py 使用文档

> 目标：在**不跑端到端前向**的前提下，基于真实数据缓存的逐层输入/输出激活 `(X, Y)`，评估 **Fake SVD-Quant（伪量化）** 对各个线性投影层的 **tokenwise 输出误差**。
>
> 本脚本实现 3 个流程：
> 1) 保存原始模型的各层 `(X, Y)`（可复用）
> 2) 无校准：对原始模型做伪量化网格搜索，并用缓存的 `X` 计算 `X @ W~^T` 与原始 `Y` 的误差
> 3) 有校准（SmoothQuant）：同上，但对 `X` 先按 `smooth_factor` 做缩放再评估

---

## 0. 依赖与约定

### 0.1 依赖

脚本默认依赖你仓库中的以下接口：

- `data_gen.get_veri_data(tokenizer, seed, cn, cn_len, eu, eu_len, device)`
  - 用于构造代表性输入（脚本默认：`seed=0, cn=50, eu=80, length=128`）。
- `fake_svdq.fake_op(model, svdq_cfg)`
  - **in-place** 对模型目标线性层做伪量化（不落盘）。
- `smooth_util.load_smoothed_model(model_dir)`（可选）
  - 用于加载已注册好 `smooth_factor` 的 smooth 模型；如果不可用，会 fallback 到 HF 普通加载。

此外：
- 推荐安装 `safetensors`，否则会自动 fallback 到 `torch.save(.pt)`。

### 0.2 严格 allowlist（只处理这些线性层）

脚本只会处理满足如下路径规则的线性层（严格匹配）：

- 模块路径正则：

  ```text
  ^model\.layers\.(\d+)\.(mlp|self_attn)\.([A-Za-z0-9_]+)$
  ```

- 且投影名必须在 allowlist：
  - `self_attn.{q_proj,k_proj,v_proj,o_proj}`
  - `mlp.{gate_proj,up_proj,down_proj}`

也就是说：
- **不会**处理 `model.embed_tokens`、`lm_head`、以及所有非 allowlist 的层。

---

## 1) 功能一：保存原始 X/Y（save_xy）

### 1.1 做什么

对代表性输入跑一次原模型前向，并用 hook 捕获每个目标线性层：

- `X`：该线性层的输入激活
- `Y`：该线性层的输出激活（即 `Y = Linear(X)` 的输出）

保存到：

- `{save_path}/{label}.safetensors`（若无 safetensors 则为 `.pt`）

键名格式为：

- `{module_path}.X`
- `{module_path}.Y`

其中 `module_path` 示例：`model.layers.0.mlp.gate_proj`。

### 1.2 命令行用法

```bash
python tokenwise_precision.py save_xy \
  --model_dir /path/to/hf_model \
  --save_path ./cache_xy \
  --label qwen3_veri128
```

### 1.3 参数说明（save_xy）

- `--model_dir`：HuggingFace 模型目录/名称（`AutoModelForCausalLM.from_pretrained` 可识别的路径）。
- `--save_path`：输出目录（自动创建）。
- `--label`：输出文件前缀，最终文件是 `label.safetensors` 或 `label.pt`。

代表性输入相关（你声明的默认设置已经内置）：
- `--seed`：抽样随机种子（默认 `0`）。
- `--cn`：中文样本数量（默认 `50`）。
- `--eu`：非中文（其他语言）样本数量（默认 `80`）。
- `--length`：统一长度（默认 `128` token）。

加载与数值相关：
- `--torch_dtype`：`auto|fp16|bf16|fp32`。建议：
  - 单卡推理/显存紧张：`fp16` 或 `bf16`
  - 追求更稳定的 hook 数值：`fp32`
- `--device_map`：传给 HF 的 `device_map`（例如 `auto` 或自定义映射）。
- `--save_dtype`：保存的 `X/Y` 精度：`fp16`（默认）或 `fp32`。
  - 若你担心误差放大或后续要做更精细的 tokenwise 统计，推荐 `fp32`。
- `--use_cache`：是否在 forward 时启用 `use_cache`（默认关闭）。
  - 对只抓线性层激活一般没必要开，但你若需要对齐某些实现细节可启用。

### 1.4 输出文件结构

- `label.safetensors`：包含大量 tensor：
  - `model.layers.0.self_attn.q_proj.X`: `[N, in_dim]` 或 `[B,T,in_dim]`（内部会拼接）
  - `model.layers.0.self_attn.q_proj.Y`: `[N, out_dim]` 或 `[B,T,out_dim]`
  - ...

并写入 metadata（若是 `.pt` 也会带）：

- `seed/cn/eu/length/save_dtype`

---

## 2) 功能二：无校准 tokenwise 前向输出误差（eval_fake）

### 2.1 做什么

- 从 `xy_path` 读取已缓存的原始 `X/Y`
- 对模型执行网格伪量化（多组 `rank × quant` 配置）
- 对每个目标线性层，用缓存的 `X` 计算：

  \( \tilde{Y} = X \cdot \tilde{W}^T \)

  并与缓存的原始 `Y` 比较。

### 2.2 误差指标

脚本对每个投影层每个配置输出：

- `nmse = ||Y~ - Y||^2 / ||Y||^2`
- `token_l2_mean / token_l2_max`：每个 token 的向量 L2（均值/最大）
- `token_rel_l2_mean / token_rel_l2_max`：相对 L2（除以 `||Y||`，避免不同层量纲差异）
- 以及 `sse/denom/n_tokens` 便于复算

### 2.3 命令行用法

最常用（rank×quant 组合网格）：

```bash
python tokenwise_precision.py eval_fake \
  --model_dir /path/to/hf_model \
  --xy_path ./cache_xy/qwen3_veri128.safetensors \
  --out_json ./results/tokenwise_nocalib.json \
  --ranks 8,16,32 \
  --qspecs nvfp4@128
```

更复杂的 qspecs：

```bash
python tokenwise_precision.py eval_fake \
  --model_dir /path/to/hf_model \
  --xy_path ./cache_xy/qwen3_veri128.safetensors \
  --out_json ./results/tokenwise_multiq.json \
  --ranks 16,32 \
  --qspecs nvfp4@128,fp4@128,fp4@in
```

### 2.4 参数说明（eval_fake）

- `--model_dir`：原始模型。
- `--xy_path`：`save_xy` 生成的 `label.safetensors/.pt`。
- `--out_json`：输出 JSON 路径。

网格相关：
- `--ranks`：用逗号分隔的 rank 列表，例如 `16,32`。
- `--qspecs`：用逗号分隔的量化规格列表，格式为 `fmt@blk`。
  - `fmt`：量化格式字符串（直接传给你 `fake_op` 的实现，例如 `nvfp4`、`fp4` 等）
  - `blk`：block size，可以是数字（如 `128`），也可以是字符串（如 `in`）
  - 例子：`nvfp4@128`、`fp4@in`

组合方式：
- `--grid_mode`：
  - `combined`（默认）：只跑 `rank × quant` 的笛卡尔积组合
  - `full`：额外包含 `rank-only` 与 `quant-only` 两类配置

性能/内存：
- `--chunk_tokens`：将 token 维度分块计算，避免一次性 matmul 占太多显存/内存（默认 `4096`）。
  - 若 `X` 很大（很多样本拼接），可适当调小，例如 `1024`。

加载相关：
- `--torch_dtype` / `--device_map`：同 `save_xy`。

权重恢复（重要）：
- `--no_restore`：默认 **不开**，即每个 cfg 之间会从 CPU 备份恢复原权重，保证每个配置的起点一致。
  - 如果你确定 `fake_op` 每次都会完全覆盖对应权重，且你想提速，可以加 `--no_restore`。
  - 但一般不建议关闭（尤其是 grid 很大时，避免残留影响）。

### 2.5 输出 JSON 结构

输出整体结构（推荐结构已实现）：

```json
{
  "meta": {
    "xy_path": "...",
    "xy_meta": {"seed": "0", "cn": "50", ...},
    "grid_mode": "combined",
    "ranks": [16, 32],
    "qspecs": [["nvfp4", "128"]],
    "smooth": false,
    "chunk_tokens": 4096
  },
  "by_layer": {
    "0": {
      "self_attn": {
        "by_proj": {
          "q_proj": {
            "rank=16;fmt=nvfp4;blk=128": {"nmse": 0.001, ...},
            "rank=32;fmt=nvfp4;blk=128": {"nmse": 0.0006, ...}
          },
          "k_proj": { ... }
        },
        "agg": {
          "rank=16;fmt=nvfp4;blk=128": {
            "nmse_mean_over_proj": 0.0012,
            "nmse_max_over_proj": 0.0035,
            "n_proj": 4
          }
        }
      },
      "mlp": { ... }
    }
  }
}
```

你后续做 Ranking-1 / Ranking-2 很方便：
- Ranking-1：用 `by_layer[layer][block]["agg"][cfg]["nmse_mean_over_proj"]` 排序
- Ranking-2：直接用 `by_layer[layer][block]["by_proj"][proj][cfg]["nmse"]` 全局排序

---

## 3) 功能三：有校准（SmoothQuant）tokenwise 误差（eval_smooth）

### 3.1 做什么

与 `eval_fake` 相同，但会对每个目标线性层读取其 buffer：`smooth_factor`，并对缓存 `X` 做：

- `X' = X / smooth_factor`

然后用 `X' @ W~^T` 与原始 `Y` 对比。

> 注意：脚本假设 **smooth_factor 已经挂在模型对应的线性层上**。如果某个目标层没有 `smooth_factor`，会直接报错。

### 3.2 你声明的 smooth_factor 注册方式

在外部（或 smooth 模型构造阶段）应按如下方式注册：

```python
import torch
import torch.nn as nn

def _set_or_register_buffer(mod: nn.Module, name: str, tensor: torch.Tensor):
    if name in mod._buffers:
        mod._buffers[name] = tensor
    else:
        mod.register_buffer(name, tensor)

# 只对 allowlist 的线性层挂 smooth_factor
# （脚本内部也提供了同名的 _iter_allowed_linears 逻辑）
for name, mod in target.items():
    _set_or_register_buffer(mod, "smooth_factor", s.detach())
```

### 3.3 命令行用法

```bash
python tokenwise_precision.py eval_smooth \
  --smooth_model_dir /path/to/smoothed_model \
  --xy_path ./cache_xy/qwen3_veri128.safetensors \
  --out_json ./results/tokenwise_smooth.json \
  --ranks 16,32 \
  --qspecs nvfp4@128
```

### 3.4 参数说明（eval_smooth）

与 `eval_fake` 基本一致，差异点：

- `--smooth_model_dir`：加载 smooth 模型（优先走 `smooth_util.load_smoothed_model`，否则 fallback HF）。
- 其余网格、chunk、restore 等参数同 `eval_fake`。

---

## 4) 常见填参示例

### 4.1 单机单卡（推荐）

```bash
# 1) 缓存 X/Y
python tokenwise_precision.py save_xy \
  --model_dir ./models/Qwen3-32B \
  --save_path ./cache_xy \
  --label qwen3_s0_cn50_eu80_len128 \
  --torch_dtype bf16 \
  --device_map auto

# 2) 无校准 tokenwise 评估
python tokenwise_precision.py eval_fake \
  --model_dir ./models/Qwen3-32B \
  --xy_path ./cache_xy/qwen3_s0_cn50_eu80_len128.safetensors \
  --out_json ./results/nocalib_tokenwise.json \
  --ranks 8,16,32 \
  --qspecs nvfp4@128 \
  --grid_mode combined \
  --chunk_tokens 2048 \
  --torch_dtype bf16 \
  --device_map auto
```

### 4.2 多种 quant 规格一起扫

```bash
python tokenwise_precision.py eval_fake \
  --model_dir ./models/Qwen3-32B \
  --xy_path ./cache_xy/qwen3_s0_cn50_eu80_len128.safetensors \
  --out_json ./results/nocalib_tokenwise_multiq.json \
  --ranks 16,32 \
  --qspecs nvfp4@128,fp4@128,fp4@in
```

### 4.3 full 网格（包含 rank-only / quant-only）

```bash
python tokenwise_precision.py eval_fake \
  --model_dir ./models/Qwen3-32B \
  --xy_path ./cache_xy/qwen3_s0_cn50_eu80_len128.safetensors \
  --out_json ./results/nocalib_tokenwise_fullgrid.json \
  --ranks 16,32 \
  --qspecs nvfp4@128 \
  --grid_mode full
```

---

## 5) 常见问题（FAQ）

### Q1：为什么只算 `X @ W~^T`，不跑端到端？

这是你定义的“前向输出误差评估（无校准/有校准）”的 **逐层模拟**：
- 不跑完整模型 forward，可以让每层/每配置的扫描成本低很多。
- 误差指标（尤其 `nmse`、tokenwise L2）更直接反映该层的量化敏感度。

### Q2：为什么需要 `--chunk_tokens`？

`X`/`Y` 是多条样本拼接后的 token 维度集合，可能非常大。
- 直接一次性算 `X @ W^T` 会占用大量显存/内存。
- 分块可以显著降低峰值。

### Q3：`--no_restore` 什么时候用？

默认每个 cfg 扫描前，会把目标线性层权重从 CPU 备份恢复，避免 cfg 之间相互污染。
- 若你确信 `fake_op` 每次都会把权重完整覆盖，并且不依赖历史状态，可用 `--no_restore` 提速。
- 否则建议保持默认。

### Q4：eval_smooth 报错 “has no smooth_factor buffer”

说明你加载的 smooth 模型未在目标线性层上注册 `smooth_factor`。需要：
- 在 smooth 模型生成/保存阶段把 `smooth_factor` 挂上；或
- 在加载后自行遍历 allowlist 层并注册 buffer。

---

## 6) 结果后处理建议

- Ranking-1（按层模块聚合）：
  - 用 `agg[cfg]["nmse_mean_over_proj"]` 或 `agg[cfg]["nmse_max_over_proj"]` 排序
- Ranking-2（按投影层细粒度）：
  - 用 `by_proj[proj][cfg]["nmse"]` 全局排序，找最不敏感/最敏感投影

如果你希望我顺便给一个 `analyze_tokenwise_results.py`（读 JSON 直接输出两类 Ranking、并导出 csv/markdown），我也可以直接补上。

风控评分调了权重之后，分数反而随风险下降，排查发现归一化把方向搞反了：归一化忽略方向，风险越高分数越低；归一化不钳制上下界，超界值把总分放大；权重不归一，特征数量变化时总分不可比；缺失值当零处理，缺失特征被当成最低风险；贡献分解用原始值而不是归一化后的加权值，之和与总分不符；阈值判定用开区间，边界值落空未定义；解释按贡献原始值排序而不是绝对值，负贡献被排到最后；阈值集合不校验单调，越界分段导致判定错乱；模型校验不检查权重非正，负权重反转风险方向；批量评分复用归一化结果，第二批沿用第一批的特征值。engine/risk/score.py 是核心，tests/risk/test_score.py 15个红。要修的是：按上面的顺序把 10 个缺陷逐个修掉。

## 运行方式

在仓库根目录运行：

```powershell
python -m pytest tests -q
```

当前预期输出为 `15 failed`，并且全部失败原因是 `NotImplementedError`。`tests/risk/test_score.py` 是验收契约，不可修改。

## 字段表

| 字段 | 值 |
| --- | --- |
| 业务域 | 安全 |
| 攻防方向 | 漏洞检测与攻防 |
| 任务类型 | Bugfix |
| 技术栈 | Python 3, pytest |
| 外部依赖 | 无外部依赖 |

## engine 嵌套布局

```text
sec-risk-scoring/
├── README.md
├── conftest.py
├── engine/
│   ├── __init__.py
│   └── risk/
│       ├── __init__.py
│       └── score.py          # 唯一实现文件，恰好 8 个桩
└── tests/
    └── risk/
        └── test_score.py     # 15 个严格测试，不可修改
```

`engine/risk/score.py` 中的八个函数全部是桩，初始均抛出 `NotImplementedError`。测试只通过 `engine.risk.score` 导入，不依赖其他实现模块。

## 模型契约

`model` 是普通 `dict`，包含以下键：

| 键 | 含义 |
| --- | --- |
| `features` | 特征名到规格的映射，规格含 `low`、`high`、`direction` |
| `weights` | 特征名到正权重的映射 |
| `missing` | 特征名到缺失值策略的映射，可省略 |
| `thresholds` | 动作阈值列表，元素形如 `{"max": 0.7, "action": "review"}` |

### 归一化与单调性

- `direction` 为 `up` 表示原始值越大风险越高，为 `down` 表示原始值越小风险越高。
- 归一化结果必须在 `[0.0, 1.0]` 内，低于下界钳到 0，高于上界钳到 1。
- 对风险方向为 `up` 的特征，原始值升高时其归一化分数不得下降；对 `down` 则相反。任一风险特征升高时，总分不得下降。

### 权重与贡献

- 权重先除以全部正权重之和，再做加权求和，因此归一化特征上的总分保持在 `[0.0, 1.0]`。
- `score` 返回 `{"score": float, "contributions": dict}`。
- 每个贡献等于该特征归一化值乘以其归一化权重；所有贡献之和必须等于总分，容差 `1e-9`。

### 缺失值策略

缺失值用 `None` 表示，按 `model["missing"]` 中该特征的策略填补到归一化尺度；`score` 未找到策略时默认 `"worst"`。

| 策略 | 填补值 |
| --- | --- |
| `"worst"` | `1.0` |
| `"best"` | `0.0` |
| `"neutral"` | `0.5` |
| 数字 | 该数字，限制在 `[0.0, 1.0]` |
| `{"policy": "constant", "value": x}` | `x`，限制在 `[0.0, 1.0]` |

缺失值不得在未声明策略时被当作零风险。

### 阈值边界

阈值列表按 `max` 严格递增，动作区间为左闭右开，最后一个区间右端闭合。因此分数等于中间边界时归入下一档，等于最后边界时归入最后一档；`[0.0, 1.0]` 内任何分数都必须有确定动作。

### 解释与校验

- `explain` 返回按贡献绝对值降序排列的 `(feature, contribution)` 元组列表，绝对值相同时按特征名升序，最多取 `top_k` 项。
- `validate_model` 返回问题码列表，合法模型返回空列表。问题码包括 `features_empty`、`weights_keys_mismatch`、`weights_non_positive`、`bounds_invalid`、`direction_missing`、`missing_policy_invalid`、`thresholds_invalid`、`thresholds_non_monotonic`。

### 批量评分

`risk_batch` 接收批次名到原始特征映射的字典，返回批次名到 `score` 结果的字典。每个批次必须独立归一化、独立套用缺失策略，后一批不得复用前一批的中间结果。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 安全 |
| 子方向 | 漏洞检测与攻防 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |


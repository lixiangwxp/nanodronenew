# AGENTS.md

## 目标

这个仓库的目标是让新窗口、新 agent、GPT Pro review 或人工接手时，能快速恢复当前实验状态、策略、gate、路径、checkpoint 和下一步动作。

文档服务决策，不记录流水日志。

## 读取顺序

1. 优先读 `MODEL_DEV_CURRENT.md`：当前 active、phase、gate、路径、checkpoint、tmux、next action。
2. 状态不明、上下文压缩、重大复盘、人工接手或 GPT Pro review 前，再读：
   - `Prompt.md`
   - `MODEL_DEV_HANDOFF.md`
   - `MODEL_DEV_STRATEGY.md`
3. 普通巡检只看必要状态，不要反复整篇读长期文档。

## 文档职责

- `MODEL_DEV_CURRENT.md`：当前 active 总控页。
- `Prompt.md`：长期压缩 ledger，只写影响后续决策的结论。
- `MODEL_DEV_HANDOFF.md`：详细交接摘要，不写逐分钟流水。
- `MODEL_DEV_STRATEGY.md`：策略、gate、promotion / kill 规则。

如果文件不存在，先基于 git、日志、checkpoint、summary CSV 做只读侦察，不要臆造状态。

## 实验原则

- 先确认 active，再行动。
- 先保护干净仓库，再引入改动。
- 先跑通最小闭环，再扩大实验范围。
- 不要把旧仓库、临时目录或远端 hotfix 整包合并进主仓库。
- 每次实验都要能追溯：改动类型、baseline、gate、配置、checkpoint、结果路径和结论。

## Git 规则
- 禁止 `git add .`，只逐个 add 本次相关文件。
- 不要把大 CSV、临时日志、缓存、checkpoint 默认加入 git。
- 正式训练 / eval / new best / review 必须能追到 branch、HEAD、experiment id、checkpoint、命令和 summary CSV。

## 工作模式

### Architecture Mode

改模型结构、loss、metric、rollout、data、eval、CLI、checkpoint loading 时使用。必须先本地修改、检查、commit，再同步远端正式跑。

### Tuning Mode

只改 batch size、lr、seed、epochs、patience、scheduler、loss 权重等参数时使用。可以不为失败 run commit，但要记录完整配置和 base commit。

## 远程训练

- 长训练、长测试、长 eval 必须用远端 `tmux`。
- 不要 SSH 前台裸跑长任务。
- 默认复用既有远程 worktree，不要每次新建。
- 远端运行前做只读检查：branch、HEAD、git status、worktree、tmux、进程、GPU、路径。
- 远端代码必须和本地一致；远端 hotfix 后必须同步回本地并 diff，否则结果不能作为正式结论。
- 同步方向必须明确：`local -> remote` 或 `remote -> local`。
- 禁止删除无关源码、数据集、baseline、成功 checkpoint 和汇总结果。

## 监控与自动化

- heartbeat 只做只读检查：tmux、进程、GPU、日志、checkpoint、summary CSV、`MODEL_DEV_CURRENT.md`。
- 无变化巡检不要写长期文档。
- 训练结束、eval 结束、crash/OOM/NaN、new best、gate 变化或策略变化时，才更新状态文档。
- 任务结束后删除 stale heartbeat automation。

## 协作方式

默认“主会话总控 + 分支 agent 执行/分析”。

主会话负责策略、边界、review、同步、一致性判断和最终汇报。  
分支 agent 可负责日志、GPU、tmux、checkpoint、CSV、summary、旧代码对比等。

重大策略节点或状态冲突时，不要单 agent 闷头推进，可以并行找证据后统一 review。

## 给用户命令时

给 Linux / SSH 命令要说明三件事：

1. 执行什么。
2. 为什么执行。
3. 具体命令。

不要只给一串命令。

## 结果规则

不要只凭 train / valid loss 下结论。正式结论至少说明：

- checkpoint
- eval / prediction 输出
- summary CSV
- 关键指标
- baseline
- gate 是否通过
- 下一步建议

如果只是 smoke run，必须明确说不是正式结果。

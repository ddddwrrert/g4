# G4

G4 在 G3 外面加了一层元策略存档。任务循环仍是 Harness² 上的行为对照：一条编辑、合并先跑、下一题从基础 harness 开始。Planner、Probe、Compose 的规则字符串不进入种群。

会进化的是提案策略：试哪些方向、Probe 预算多少。存档池按 [HyperAgents](https://github.com/facebookresearch/HyperAgents) 的 `score_child_prop` 抽样。权重是前沿中点上的 sigmoid，再乘 `exp(-(子代数/8)**3)`。留出题上的分数才进入这个权重。训练题上的分数只留在任务账本里。

一次 diff 的归属：

- 只改 `solution.cpp`：一阶。分数写入 `task_ledger.jsonl`，不成为元策略父代。
- 只改 `policy.json`：二阶。先过训练筛选（不低于父策略的训练分），再在留出题上打分。高于父策略的留出分才标成可选父代。
- 两侧都改：拆开，不记分。

`third_party/harness2` 是 Harness² 提交 `6868817518876de8fd170c6a8354daca273b6191`，Apache-2.0，未改。

`third_party/hyperagents/selection.py` 改编自 HyperAgents 提交 `59a68f672dfb92c74aeb7e61535d776fb36e172d` 的 `select_parent` 和 `filter_patch_by_files`，CC BY-NC-SA 4.0。

运行：

```bash
python -m g4
python -m unittest tests.test_closed_loop tests.test_meta
```

需要 Python 3.12 或 3.13。AHC 题库仍通过 `ORGANON_ROOT` 指向外部目录，不放进这个仓库。

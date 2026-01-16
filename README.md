# Linear Attention Research Playbook

> 调研当下主流/新兴的线性注意力机制，梳理可复现实验逻辑，并在自选模型结构与任务上验证既有结论，重点观察训练速度、推理速度、内存占用与模型表现。

## 🎯 Goals & Research Questions
- 将 **Softmax/Flash Attention** 作为参考基线，系统复现 **FLA Linear Attention** 与 **Gated DeltaNet** 的优势/劣势。
- 形成“一键式”实验脚本 `pt.py`，覆盖数据准备、预训练、合成任务验证、性能 Profiling 与可视化，为后续扩展提供统一入口。
- 在相同参数预算下比较不同注意力实现的 **训练吞吐、推理延迟、显存峰值、语言模型困惑度、记忆任务准确率**，验证社区中对线性注意力的既有结论。
- 输出结构化报告（CSV/图表/TensorBoard 日志），方便与新模型或新超参快速对比。

## 🗂️ Repository Highlights
| File | Description |
|------|-------------|
| `pt.py` | 主实验脚本：实现多种注意力层、语言模型骨架、本地 WikiText parquet 数据集、Copy/Associative Recall 任务、效率/内存测试与 Torch Profiler。|
| `pretrain.py`, `conti.py`, `rail.py` | 预留的其他训练脚本，可在完成主实验后做延伸。|
| `scripts/` | HPC/集群脚本模板，可参考其中写法适配自己的环境。|

> 当前阶段请以 `pt.py` 为唯一事实来源，其余文件仅作背景参考。

## 🔬 Experiment Pipeline (built into `pt.py`)
1. **环境检查**：自动检测 CUDA/FlashAttention/FLA，可在无 FlashAttention 或无 FLA 时回退至 PyTorch/自实现版本。
2. **阶段1 – 语言建模预训练**：
   - 构造字符级 WikiText parquet 数据集，自行统计词表（默认 5k）并复用于训练/验证集。
   - 针对每种注意力配置 `hidden_size / num_layers / num_heads`，保证参数量可对齐；提供自动校准函数 `calibrate_model_configs`。
   - 训练中记录 `train_loss`、`val_loss`、`val_ppl`，每个注意力机制独立 optimizer/scheduler。
3. **阶段2 – 统一测试面板**：
   - `test_training_efficiency`：多序列长度下的训练步时间、吞吐、显存峰值；
   - `test_memory_scaling`：纯注意力算子的显存扩展性曲线；
   - `test_copy_task` / `test_associative_recall`：长程记忆 & 关联记忆合成任务，输出 accuracy；
   - `test_perplexity_summary`：复用预训练模型的验证困惑度；
   - `test_model_scaling`：1M~30M 参数规模下的速度/显存趋势。
4. **阶段3 – Torch Profiler**：
   - `profile_attention`：逐序列长度 profile，统计 CUDA 时间/内存并绘制对比曲线、柱状图、log-log 复杂度曲线；
   - `profile_attention_detailed`：导出 Chrome Trace、TensorBoard、stacks.txt，便于火焰图分析。
5. **阶段4 – 报告生成**：所有结果写入 `./results/*.csv`、`./results/history.json` 以及 `./figures/*.png/.pdf`。

## 📊 Metrics We Track
| Category | Metrics |
|----------|---------|
| 训练效率 | step time (ms), throughput (token/s), gradient clip stability |
| 推理/算子性能 | CUDA kernel 时间、log-log 复杂度斜率、Chrome Trace |
| 资源占用 | GPU 峰值显存 (MB), memory scaling 斜率 |
| 模型表现 | validation perplexity, copy task accuracy, associative recall accuracy |
| 扩展性 | 参数量 vs. step time / throughput 曲线 |

这些指标共同刻画“线性注意力 = 更快 + 更省 + 保性能”的既有结论能否在我们设定的模型/数据/任务上成立。

## 🚀 Running the Study
1. **准备数据**：在 `Trainer` 初始化中设定 `data_dir` 指向包含 `train-00000-of-00001.parquet` & `validation-00000-of-00001.parquet` 的 WikiText parquet 目录。可替换成自定义文本集，只需保证 `text` 字段存在。
2. **安装依赖**：
   ```bash
   pip install torch pandas numpy matplotlib tqdm pyarrow fla flash-attn
   ```
   缺失 FLA/FlashAttention 时脚本会 fallback，但性能更低。
3. **启动主脚本**：
   ```bash
   python pt.py \
     --(可选自定义，当前脚本在 main 中直接实例化 Trainer)
   ```
   修改 `main()` 或在交互式 notebook 中 `from pt import Trainer`，可灵活控制阶段/函数。
4. **查看结果**：
   - 数据表：`results/*.csv`
   - 训练曲线/性能图：`figures/*.png|pdf`
   - Profiler：`profiler_logs/` (TensorBoard), `profiler_results/*_trace.json` (Chrome Trace), `*_stacks.txt` (speedscope)。

> 建议逐阶段运行（先预训练，确认收敛，再逐项调用测试函数），可减少显存占用与调试开销。

## 🧠 Validating Prior Conclusions
- **速度**：比较 `Softmax/Flash vs. Linear/GatedDeltaNet` 的 step time & throughput，看是否真正达到线性复杂度的优势；
- **显存**：检查 `memory scaling` 曲线斜率是否逼近 O(n)；
- **表现**：用 `val_ppl`、`copy/assoc` 合成任务评估记忆与长程依赖，核对线性注意力是否存在性能折衷；
- **推理/内核**：利用 profiler + stacks，识别瓶颈 kernel 与算子调度，验证社区报告的 kernel 加速是否在本地复现。

## 🔄 Extending the Playground
- 新增注意力：在 `Trainer.ATTN_CLASSES` 注册即可获得同样的评测套件。
- 新任务：继承 `torch.utils.data.Dataset`，加入对应的 `test_xxx` 函数并在 `plot_results` 中注册。
- 跨模型：可在 `LanguageModel` 外包装自己定义的骨干，但务必对齐参数量以保持公平对比。

## ✅ Status
- [x] 统一脚本 `pt.py` 已完成，实现上述流程；
- [x] 具备数据→训练→评测→Profiler→可视化的完整闭环；
- [ ] 还需在自定义数据/任务上实际运行以获取最终报告。

欢迎基于该 Playbook 继续迭代，补充更多线性注意力变体或系统级优化实验。

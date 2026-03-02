# 动态划分逻辑代码审计（pd_split）

结论（先说结论）：
- `pd_split` 模式不存在“运行中动态重划分层映射/迁移权重”的实现。
- `pd_split` 的层划分仅在实验初始化阶段根据策略计算一次（compute/mem/bandwidth/uniform/auto）。
- 运行期仅支持批处理参数（batch size/timeout/decode workers）调整，不会改变层到节点的映射。

## 1) 划分逻辑位置

`backend/layer_strategy.py` 的 `build_ranges` 只根据节点静态画像（算力/显存/带宽）计算区间，返回层范围；没有任何运行时反馈闭环或周期重算入口。

## 2) pd_split 的生命周期

`backend/modes.py` 中 `PDSplitExperiment.__init__` 启动时：
- 读取集群配置并筛选 prefill/decode 节点；
- 读取模型层数；
- 根据 prefill/decode 策略调用 `build_ranges` 各算一次；
- 构建 stage handle 并常驻使用。

后续请求处理 `_prefill_loop/_decode_loop` 仅消费固定的 `self.prefill_stages/self.decode_stages`，没有重建 pipeline 的调用。

## 3) API 层可变更项

`serve.py`：
- `POST /api/pd_split/batching` 只会调用 `update_batching`，更新微批与 worker 数；
- 没有 `pd_split` 的“更新节点-层映射”接口。

## 4) 仓库里“看起来像动态重划分”的唯一入口

`full_pipeline` 模式支持 `POST /api/full_pipeline/config`：
- 调用 `update_full_pipeline(nodes, strategy)` 重建 stage；
- 且要求队列与执行都为空（pipeline busy 会被拒绝）。

这属于“受控重配置”，不是运行中按负载自动迁移层。

## 5) 相关旁证

- 监控模块 `backend/monitor.py` 仅采集并上报显存/系统指标，不参与策略回写。
- 前端 `frontend/app.js` 里 `pd_split` 只能改 batching；全流水线才有配置应用按钮。


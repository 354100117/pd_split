## 数学建模：PD 推理流水线抽象分析

本文给出该项目推理流水线的抽象建模，用于论文描述系统结构、时延与吞吐分析。

### 1. 符号与系统假设

#### 1.1 基本符号

- **模型层数**：$L$
    
- **节点集合**：$\mathcal{N}$
    
- **阶段集合**：
    
    - Prefill 阶段：$\mathcal{S}_p = \{ s_1, \dots, s_{S_p} \}$
        
    - Decode 阶段：$\mathcal{S}_d = \{ s'_1, \dots, s'_{S_d} \}$
        
- **层区间**：每个阶段 $s$ 对应的层区间为 $[l_s, r_s)$。
    

#### 1.2 请求与批处理参数

- **请求 $i$**：
    
    - Prompt 长度（Token 数）：$p_i$
        
    - 生成长度：$g_i$
        
- **批处理参数**：
    
    - **Prefill**：批量大小 $B_p$，超时阈值 $\tau_p$
        
    - **Decode**：批量大小 $B_d$，超时阈值 $\tau_d$
        

#### 1.3 系统假设

1. **并行模式**：Prefill 与 Decode 均采用流水线并行（Pipeline Parallelism），阶段间同步执行。
    
2. **对齐机制**：Decode 批处理要求 `max_new_tokens` 一致；历史长度（`past_len`）通过 Padding 对齐。
    
3. **同步性**：通信与计算不显式重叠，采用同步 RPC 调用。
    

---

### 2. 分段时延模型

对于请求 $i$，总时延 $T_i$ 分解为：

$$T_i = W_p(i) + T_i^{p} + W_d(i) + T_i^{d}$$

其中：

- $W_p(i), W_d(i)$：分别为 Prefill 和 Decode 的排队等待时间。
    
- $T_i^{p}$：Prefill 阶段的执行时间。
    
- $T_i^{d}$：Decode 阶段的累计生成时间。
    

#### 2.1 Prefill 执行时间

Prefill 计算代价与 Prompt 长度 $p_i$ 近似线性：

$$T_i^{p} = \sum_{s \in \mathcal{S}_p} \left( C_s^{p}(p_i, B_p) + D_s^{p}(B_p) \right)$$

- $C_s^{p}$：阶段 $s$ 的计算时间。
    
- $D_s^{p}$：阶段间通信时间（含 RPC 开销与 Tensor 传输）。
    

#### 2.2 Decode 单步时间

每生成一个 Token，需经过所有 Decode 阶段。定义单步时间为：

$$T_{\text{step}}^{d}(B_d) = \sum_{s \in \mathcal{S}_d} \left( C_s^{d}(B_d, L_{\max}) + D_s^{d}(B_d) \right)$$

因此，请求 $i$ 的 Decode 总时间为：

$$T_i^{d} = g_i \cdot T_{\text{step}}^{d}(B_d)$$

---

### 3. Decode 批处理与 Padding 机制

由于 Batch 内请求的历史长度不同，系统通过 Padding 对齐到当前批次的最大长度 $L_{\max}$：

$$L_{\max} = \max_{i \in \mathcal{B}} (\text{past\_len}_i)$$

**Padding 额外开销比率**：

$$\text{overhead}_i = \frac{L_{\max} - \text{past\_len}_i}{L_{\max}}$$

> **注意**：当 Batch 内请求长度差异极大时，有效算力利用率会显著下降。

---

### 4. 调度与队列模型

#### 4.1 PD 分离模式 (pd_split)

建模为串联队列系统：$Q_p \rightarrow Q_d$。

- **资源分布**：Prefill 与 Decode 拥有独立算力资源。
    
- **解耦性**：Prefill 的高延迟抖动不会直接阻塞 Decode 节点的算力步进。
    

#### 4.2 全节点流水线 (full_pipeline)

建模为共享服务台的多类队列模型。Prefill 与 Decode 请求竞争同一组算力与通信资源：

$$W_d(i) = f(\lambda_p, \lambda_d, B_p, B_d)$$

- $\lambda_p, \lambda_d$ 分别为 Prefill 和 Decode 的请求到达率。
    

---

### 5. 吞吐率估计 (Throughput)

系统整体吞吐率（TPS, Tokens per second）通常受限于 Decode 阶段的步进速度：

$$\text{TPS} \approx \frac{B_d}{T_{\text{step}}^{d}(B_d)}$$

|**模式**|**吞吐特征**|
|---|---|
|**pd_split**|TPS 由专门的 Decode 集群上限决定，易于横向扩展。|
|**full_pipeline**|TPS 受 Prefill 抢占影响，但在低负载下资源利用率更高。|

---

### 6. 气泡与资源利用率分析

定义阶段 $s$ 的利用率 $U_s$：

$$U_s = \frac{T_s^{\text{busy}}}{T_s^{\text{busy}} + T_s^{\text{idle}}}$$

**空闲时间（Bubble）主要来源**：

1. **批次填充不足**：$\text{size}(\mathcal{B}) < B_d$。
    
2. **数据依赖**：等待上一个阶段的 KV Cache 传输。
    
3. **同步瓶颈**：Pipeline 中最慢阶段（Straggler）导致的同步等待。
    

---

**下一步建议：**

如果你需要将此内容放入正式论文，我建议可以**补充 Little's Law (利特尔法则)** 来推导平均队列长度与时延的关系。需要我为你增加这一部分的数学推导吗？
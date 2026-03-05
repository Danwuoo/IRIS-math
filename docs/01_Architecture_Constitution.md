# Architecture Constitution

**Document Type:** Authoritative Specification (Normative)  
**Effective date:** 2026-02-27  
**Replaces (removed on 2026-02-27):**
- System Invariants & Non-Negotiables
- What This Model Is Explicitly NOT
- Single Trunk Contract & Allowed Variations
- Routing, Gating, and Control Are Learnable

(See `docs/00_INDEX.md` for the historical path mapping.)

**Appendix (Non-normative):** Architecture design sketch (previously a separate file; inlined here on 2026-02-27).

**Authority:** If any text in this document conflicts with other non-normative notes, this document prevails.

---

## System Invariants & Non-Negotiables

**(Authoritative Specification, v2)**

### 0. Document Positioning

#### 0.1 Purpose

This document defines the immutable architectural constraints for IRIS as a
**foundation pretraining base model**.

These invariants are **task-agnostic** and **benchmark-agnostic**.

#### 0.2 Non-Goal

This document does **not** define any specific benchmark (including ARC) as a
primary optimization target.

Benchmarks are evaluation instrumentation only.

#### 0.3 Authority

Any implementation, refactor, or extension that violates this document is
architecturally invalid, even if short-term metrics improve.

---

### 1. Trunk Invariants

#### 1.1 Single Trunk

- IRIS must contain exactly one **single trunk** that serves as the primary
  parameter-bearing substrate.
- The trunk is **architecture-agnostic** at this invariant layer. It may be
  attention-based, SSM-based, hybrid, or another equivalent family.
- Introducing a second parallel network at comparable capacity (a second
  "brain") is forbidden.

#### 1.2 Trunk Computational Responsibility

- Core cognitive computation must be attributable to trunk learning dynamics,
  including representation formation, control, and the primary credit-assignment
  loop.
- Engineering scaffolding is allowed (for example tokenizer, IO adapters,
  cache, retrieval index), but scaffolding must not hard-code semantic control
  flow.

---

### 2. State IR Invariants

#### 2.1 Canonical Internal Representation

- State IR is the canonical internal representation consumed by the trunk and
  shared across levels.
- Ad-hoc internal state channels that bypass canonical State IR are forbidden.

#### 2.2 Schema Stability

- State IR schema must remain stable across tasks and data modalities.
- State IR must not be specialized for a single benchmark or benchmark-specific
  token assumptions.

#### 2.3 Controlled Evolution

- Any State IR schema change requires explicit, versioned specification updates.
- Silent token/category drift is forbidden.

---

### 3. Learnable Routing and Control Invariants

#### 3.1 Learned-by-Default Control

- Routing, gating, and control signals must be learnable and represented in
  model parameters.
- Hard-coded orchestration is not an acceptable primary decision mechanism.

#### 3.2 Guardrails vs. Semantics

- Safety gating and resource limits are allowed as engineering guardrails.
- Guardrails must not perform semantic decomposition or decide reasoning steps
  as routine policy.

---

### 4. Level Interface and Mounting Invariants

#### 4.1 Level Interfaces Must Exist

- IRIS defines Level interfaces (L0-L6 or an equivalent explicitly defined set).
- Interfaces must exist even when a level implementation is disabled in a given
  checkpoint/config.
- For disabled levels, the system must preserve:
  - IO contract
  - State naming compatibility
  - Trunk interaction points via a stub (no-op or low-capacity adapter)

This ensures levels can be remounted without breaking trunk or State IR
consistency.

#### 4.2 Optional Mounting and Configurable Attachment

- Level implementations may be disabled, replaced, or attached at different
  capacities.
- Any mounted implementation must satisfy the corresponding Level contract,
  including observability and credit-assignment compatibility.

---

### 5. Credit Assignment and Failure Recovery Invariants

#### 5.1 Attributable Credit Assignment

- The system must support attributable credit assignment for output and behavior.
- Attribution must be traceable to trunk contributions and, when mounted,
  per-level contributions.

#### 5.2 Learned Recovery

- If failure-recovery behaviors exist (retry, reflection, repair, self-check),
  they must be part of learned policy.
- Hard-coded recovery flowcharts are forbidden as default behavior.

---

### 6. Status of Benchmarks and Tools

#### 6.1 Benchmarks

- Benchmarks (including ARC) are **evaluation instrumentation** in this
  specification.
- Benchmarks are not architecture invariants.
- Benchmarks are not the sole source of training objectives.

#### 6.2 Tools

- Toolchains (solver, search, external programs) may be used for data
  generation, monitoring, and evaluation.
- Toolchains must not replace trunk-learned reasoning or control.

---

### 7. Explicit Non-Negotiability Clause

If there is a tradeoff between architectural compliance and short-term
performance or convenience, architectural compliance takes precedence.

---

### 8. Summary

IRIS must always preserve:

- one and only one primary trunk
- architecture-agnostic trunk family at invariant level
- canonical, benchmark-agnostic State IR
- learnable routing and control
- persistent level interfaces with optional mounting
- attributable credit assignment and learned failure recovery
- benchmark/tooling as instrumentation, not intelligence substrate

These are invariants, not suggestions.

---

**End of Document**

---

## What This Model Is Explicitly **NOT**

**Document Type:** Explicit Non-Goals
**Scope:** System-level architectural and philosophical exclusions
**Status:** Normative (non-negotiable)
**Audience:** Researchers, implementers, and any agent modifying or extending the system

This document enumerates *explicit non-goals* of the model. These exclusions are intentional and foundational. Any design, optimization, refactor, or extension that drifts toward the prohibited categories below is considered a violation of the model’s core philosophy and must be rejected, even if it appears to improve short-term performance, efficiency, or simplicity.

This document exists to prevent silent architectural collapse via “reasonable” engineering shortcuts.

---

### 1. This Model Is **NOT** a Tool-First Agent

#### 1.1 No External Tools as Primary Reasoning Substrate

The model is **not** designed around tools (APIs, solvers, executors, planners) as the primary locus of intelligence.

Specifically:

* The system does **not** assume tools are more reliable than internal reasoning.
* The system does **not** offload core cognition to tools.
* The system does **not** treat tools as oracles.

Tools, if present, are:

* Optional
* Peripheral
* Invoked only through learned, weight-based routing

Any architecture where:

* “Thinking” is reduced to deciding *which tool to call*, or
* The trunk becomes a thin dispatcher around tools

is explicitly out of scope.

---

### 2. This Model Is **NOT** a Symbolic-First or DSL-Centric System

#### 2.1 No Pure Symbolic Executors

The model explicitly rejects architectures where:

* Programs are executed by a fully hand-written DSL interpreter.
* Correctness is guaranteed by symbolic rules rather than learned representations.
* Neural components merely rank, select, or wrap symbolic executions.

All program execution **must** involve learned neural primitives.
Symbolic structure may exist, but **semantic authority cannot reside outside the weights**.

#### 2.2 No “Neural Proposal + Symbolic Executor” Split

A common failure mode is:

> “Neural network proposes → symbolic system executes → neural network scores.”

This model **explicitly forbids** that separation of responsibility.

If execution semantics live outside the model weights, credit assignment collapses. This is unacceptable.

---

### 3. This Model Is **NOT** an If-Else System in Disguise

#### 3.1 No Hard-Coded Control Logic Masquerading as Architecture

The following are explicitly disallowed:

* Hand-written routing trees
* Deterministic phase pipelines
* Rule-based level invocation
* Hard-coded termination conditions
* Search depth, beam size, or rollout length fixed by code logic

Any decision that *appears* to be “control flow” must ultimately be:

* Parameterized
* Learnable
* Expressible via weights (even if approximated during MVP)

If the system’s behavior can be fully understood by reading control code alone, the design has failed.

---

### 4. This Model Is **NOT** a Flat End-to-End Network

#### 4.1 No Single Loss That Erases Layer Responsibility

The system explicitly rejects:

* Monolithic end-to-end loss without structured credit assignment
* Architectures where intermediate layers have no semantic obligations
* Training regimes that treat all internal structure as incidental

The layered design is not cosmetic.
Each level has **semantic responsibility**, and failure must be attributable across levels.

Any proposal that:

* “Simplifies” training by collapsing losses, or
* Treats intermediate modules as interchangeable

is out of scope.

---

### 5. This Model Is **NOT** Optimized for Human-Readable Programs

#### 5.1 No Requirement for Interpretability via Programs

The system does **not** aim to produce:

* Human-readable code
* Clean DSL programs
* Minimal or elegant symbolic traces

Programs, macros, and abstractions are:

* Internal latent structures
* Optimized for learnability and credit assignment
* Free to be redundant, messy, or opaque

Human interpretability is not a design constraint and must not shape architectural decisions.

---

### 6. This Model Is **NOT** a Planner-Only or Search-Only System

#### 6.1 No Delegation of Intelligence Solely to Search

Search is a tool, not the intelligence itself.

The model explicitly rejects:

* Treating search as the primary problem solver
* Using brute-force enumeration with weak heuristics
* Relying on search depth to compensate for representational weakness

Search, rollout, and expansion exist **only** as guided processes under learned control.
If increasing search budget is the main path to improvement, the architecture is misaligned.

---

### 7. This Model Is **NOT** a Memory-First Retrieval System

#### 7.1 No Retrieval as a Substitute for Reasoning

The system is not designed as:

* A nearest-neighbor problem solver
* A case-based reasoning engine
* A retrieval-augmented system where memory dominates inference

Memory:

* Assists reasoning
* Does not replace it
* Must be gated, fused, and written by learned mechanisms

Any architecture where retrieval alone can solve most tasks indicates a failure of abstraction learning.

---

### 8. This Model Is **NOT** an Engineering-Optimized Minimal System

#### 8.1 No Architecture Shaped Primarily by Convenience

The system explicitly rejects design choices motivated primarily by:

* Ease of implementation
* Familiar frameworks
* Existing libraries
* Short-term performance benchmarks

Examples of forbidden rationales:

* “This is simpler to code.”
* “This is how existing agents do it.”
* “This removes the need for another head.”

Architectural integrity takes precedence over engineering convenience.

---

### 9. This Model Is **NOT** Guaranteed to Be Stable Without Learning

#### 9.1 No Expectation of Hand-Tuned Optimality

The model is **not** expected to:

* Work optimally with frozen routing
* Behave sensibly under fixed heuristics
* Be robust without training pressure

Instability prior to learning is acceptable.
What matters is that instability is **learnable**, not manually patched.

---

### 10. Summary of Explicit Non-Goals

This model is explicitly **not**:

* A tool-centric agent
* A symbolic executor with neural wrappers
* A rule-based controller
* A flat end-to-end network
* A human-readable program synthesizer
* A brute-force planner
* A retrieval-dominant system
* An engineering-minimal design
* A hand-stable system without learning

Any modification that moves the system toward any of the above categories must be rejected, regardless of apparent gains.

---

### 11. Enforcement Clause

If a future agent, contributor, or automated system proposes a change that:

* Violates any exclusion in this document, or
* Introduces ambiguity about these non-goals

that proposal must be considered **architecturally invalid by default**.

Justification must prove *non-violation*, not utility.

---

**End of Document**

---

## Single Trunk Contract & Allowed Variations

### 0. Scope 與權威層級

- 本文件定義 IRIS 的 **single trunk** 必要條件、可允許的變體、以及違規判準。
- 本文件是架構層合約：**不依賴任何特定 benchmark**；benchmark 僅屬評估工具，不得反向定義 trunk 結構。
- 本文件不規定 trunk 必須是 attention / SSM / hybrid；僅規定它們必須滿足的行為與接口。

---

### 1. 定義

#### 1.1 Trunk（主幹）

- Trunk = 主要承載模型能力的參數集合，負責：
    - 將輸入映射到 State IR（或其等價內部表徵）
    - 執行控制、路由、信用分配（credit）相關的主要學習動態
    - 生成輸出（或生成可被 head 解碼的內部狀態）

#### 1.2 「第二個 trunk」的判準（Multi-trunk 反模式）

以下任一條成立，即視為**第二個 trunk**（違規），除非被明確標註為「低容量 adapter」並通過容量門檻（見 1.3）：

- **參數量級判準**：存在一個獨立模組，其參數量或 FLOPs 長期穩定地達 trunk 的顯著比例（建議門檻：≥ 25%）。
- **獨立優化判準**：該模組有獨立 optimizer state/學習率排程，且其學習對核心能力至關重要。
- **語義閉環判準**：該模組形成一個能自行完成「理解→推理→決策→輸出」的閉環，而 trunk 只扮演 IO 或轉接。
- **競爭路由判準**：路由在兩個大模組間做主決策，且任何一方都可在多數步驟中繞過另一方完成主要運算。

> 這些判準的目的：禁止「雙腦分裂」與不可追責的能力外包。

#### 1.3 低容量模組（Allowed Peripheral Capacity）

- 允許存在 peripheral modules（例如 tokenizer、IO adapters、memory index、輕量 head、監控器），但它們必須：
    - 不形成語義閉環
    - 不取代 trunk 的推理/控制決策
    - capacity 明顯低於 trunk（建議：≤ 5–10% 參數量或推理 FLOPs，取嚴者）

---

### 2. Single Trunk 核心不變量

#### 2.1 單一主幹責任

- 所有「語義層」決策（分解、控制、路由、信用分配策略、修復策略）必須在 trunk 的 learned dynamics 中可見、可訓練、可歸因。
- 禁止把語義決策硬編碼在外圍管線（例如規則式 planner、固定流程圖、手寫 search policy）中。

#### 2.2 Trunk 與 State IR 的一致性

- Trunk 必須以 State IR（或其等價 schema）作為主要內部狀態表示（canonical internal representation）。
- 允許在 trunk 內部有多個 latent spaces，但必須能映射回 canonical State IR（至少在接口點上）。

#### 2.3 可觀測性（Observability）

- Trunk 的關鍵狀態必須可被 instrumentation 觀測（例如：中間 state、control logits、routing gate、信用分配信號）。
- 若出於效率採用壓縮或 checkpointing，仍須提供可還原/可抽樣的觀測鉤子。

---

### 3. Allowed Variations（允許的 trunk family）

以下任一類均允許，只要滿足第 2 節不變量與第 4 節接口要求：

#### 3.1 Attention-based

- 全注意力、稀疏注意力、線性注意力、MoE-attention 等
- 允許長上下文策略（滑窗、分塊、外部 KV 壓縮），但不得形成第二個語義閉環。

#### 3.2 SSM/Sequence Model-based

- 任意 SSM/Hyena/Mamba-like family
- 允許 state cache、selective scan 變體、chunked recurrence

#### 3.3 Hybrid

- Attention + SSM 混合、分層 block、交替堆疊
- 關鍵限制：混合仍然是一個 trunk（共享訓練目標/主狀態、無獨立大腦）

#### 3.4 Retrieval-augmented（僅作為外圍、非第二大腦）

- 允許 retrieval index / vector store 作為外部 memory
- 必須滿足：
    - retrieval 不具語義閉環
    - retrieval policy（何時查、查什麼、怎麼用）主要由 trunk learned 控制

---

### 4. Interfaces（trunk 必須提供的對外接口）

#### 4.1 IO 接口

- `encode(inputs) -> StateIR/LatentState`
- `step(state, inputs/control) -> state'`
- `decode(state) -> outputs` 或 `produce_logits(state)`

（具體函式名可不同，但語義要等價：編碼、狀態更新、解碼/輸出。）

#### 4.2 Control/Routing 接口（若系統定義了 learned control）

- Trunk 必須輸出可訓練的 control signals（例如 gate/logits/latent commands）
- 並允許在訓練/評估時抽樣、干預、或記錄這些信號（for credit assignment）

#### 4.3 Level Interfaces 的挂載點（對齊你的決策 2）

- Trunk 必須提供 level interface 的掛載點（即使某些 level 實作關閉，也保留 stub）
- stub 必須：
    - 不破壞 State IR 流
    - 不引入新的語義閉環
    - 對 observability 一致

---

### 5. Prohibited Patterns（明確禁止）

#### 5.1 Dual-core / Two-brain

- 任何形式的「大 encoder + 大 solver」並行，互相可替代完成主要推理

#### 5.2 Hard-coded semantic control flow

- 規則式 planner 主導步驟、硬編排 search policy、固定推理流程圖
- 允許工程級限制（資源上限、安全 gating），但不得承擔語義分解與決策

#### 5.3 Benchmark-shaped trunk

- 為單一 benchmark 直接特化 trunk 結構（例如固定動作空間、固定步數、固定 grid-specific pipeline）
- 可針對 benchmark 加 heads/adapters，但 trunk contract 不得被 benchmark 反向塑形

---

### 6. Compliance Checklist（驗收清單）

在 PR 或設計 review 時，至少回答：

1. 本變更是否引入第二個 trunk？依 1.2 判準逐條否定。
2. peripheral modules 的容量是否低於門檻？是否可能形成語義閉環？
3. control/routing 是否仍為 learned？是否被硬編排取代？
4. trunk 是否仍能產生/維持 canonical State IR？
5. Level interfaces 的挂載點是否仍存在（就算關閉實作）？
6. 是否出現 benchmark-shaped trunk 的特化？

---

### 7. Rationale

- Single trunk 的目的：避免能力與責任分裂，確保 credit assignment 與 failure recovery 可歸因、可學習、可擴展。
- Architecture-agnostic 的目的：允許你在 scaling/吞吐/長上下文等工程權衡下切換 trunk family，而不破壞系統合約。

---

## Routing, Gating, and Control Are Learnable

**(Normative Contract Document)**

### 1. Purpose and Scope

This document defines the **non-negotiable philosophy and technical constraints** governing routing, gating, and control mechanisms in the system. Its purpose is to prevent the erosion of the model’s layered learning structure through hard-coded logic, heuristic shortcuts, or implicit procedural control.

This contract applies to **all inter-Level invocation decisions**, **intra-Level execution control**, and **cross-module fusion behaviors**, regardless of whether such decisions appear “engineering-related” or “performance-motivated.”

---

### 2. Core Principle

> **All routing, gating, and control decisions that affect model behavior MUST be represented in learnable parameters and participate in training, credit assignment, and calibration.**

Control is not an external orchestration layer.
Control is a **first-class learned capability**.

---

### 3. Definitions

#### 3.1 Routing

Routing refers to decisions that determine:

* Which Levels are invoked (e.g., L1 rollout vs. L2 program induction),
* Which candidate states, programs, or memories are expanded or pruned,
* Which outputs are selected when multiple candidates exist.

#### 3.2 Gating

Gating refers to continuous or discrete-valued signals that:

* Modulate contribution strength (e.g., soft enable/disable),
* Control residual fusion, memory read/write, macro updates,
* Influence budget allocation, search depth, or termination.

#### 3.3 Control

Control refers to any mechanism that:

* Alters computational flow, iteration count, or resource allocation,
* Determines stopping conditions or retry behavior,
* Influences exploration vs. exploitation trade-offs.

---

### 4. Mandatory Learnability Requirements

#### 4.1 No Behavioral Control Outside Weights

The following are **forbidden** as primary decision mechanisms:

* Hard-coded if/else logic that selects Levels, tools, or execution paths,
* Rule-based thresholds (including fixed confidence cutoffs),
* Hand-tuned heuristics that bypass learned signals,
* Deterministic controllers that do not expose gradients or learning signals.

All such logic, if temporarily unavoidable, **MUST** be explicitly labeled as *technical debt* (see Section 7).

---

#### 4.2 Routers Are Not Conditionals

Routers **MUST NOT** degenerate into:

* Static dispatch tables,
* Deterministic pipelines,
* Task-type switches.

A Router is defined as a **parameterized function** producing:

* Soft or probabilistic routing decisions,
* Scores, logits, or gates that can be trained, calibrated, and overridden by learning.

---

### 5. Required Learnable Control Modules

The system **MUST** include learnable parameters for at least the following control surfaces:

#### 5.1 Level Invocation Routing

* Decides whether and how strongly each Level (L1–L5) is invoked.
* Implemented via soft gates or scored selectors.
* Even when a Level is effectively “off,” its gate **must exist** and remain learnable.

#### 5.2 State Fusion Routing

* Governs how outputs from rollout, program execution, memory retrieval, and macro abstraction are merged back into the State IR.
* Fusion **must not** be fixed summation or concatenation.
* At minimum, gated residual or FiLM-style modulation is required.

#### 5.3 Output Selection Routing

* Resolves multiple candidate solutions, programs, or states.
* Selection **must** be learned (e.g., reranking, voting with learned weights).
* Deterministic tie-breaking is prohibited.

---

### 6. Separation of Concerns: What Control Is *Not*

#### 6.1 Control Is Not Scheduling

* Execution order, batching, or hardware-level optimizations may exist,
  but **must not encode semantic decisions** about reasoning strategy.

#### 6.2 Control Is Not DSL Semantics

* Program execution semantics may be partially structured,
  but **selection, matching, and applicability** of primitives must be learned.

#### 6.3 Control Is Not Human-Interpretable Policy

* The system does not optimize for readability or symbolic clarity.
* Learned control may be opaque; this is acceptable and expected.

---

### 7. Hard Control as Explicit Technical Debt

Some hard control may exist temporarily due to tooling or infrastructure limitations (e.g., maximum loop counts, safety caps).

Such mechanisms **MUST** satisfy all of the following:

1. Explicitly documented as *temporary technical debt*, including the intended learned replacement and a removal criterion.
2. Isolated so they can be replaced by learned counterparts.
3. Non-binding under intended operation: hard control may only function as a guardrail (e.g., safety caps), not as routine policy (e.g., fixed beam size, fixed rollout depth, fixed termination rules).
4. Do not encode task-specific heuristics.
5. Do not silently bias credit assignment across Levels.

Failure to label hard control as technical debt constitutes a **contract violation**.

---

### 8. Credit Assignment Compatibility

All routing and gating signals **MUST** be compatible with:

* Backpropagation or surrogate learning signals,
* Cross-Level credit routing (e.g., L6 → L3 → L2/L1),
* Calibration and uncertainty estimation.

Any control decision that cannot, in principle, receive learning feedback is **disallowed**.

---

### 9. Non-Negotiable Constraints Summary

* Routing ≠ if/else
* Gating ≠ fixed thresholds
* Control ≠ external orchestration
* Efficiency optimizations ≠ behavioral logic

Violations of these constraints undermine the layered architecture and are considered **system-level failures**, not implementation details.

---

### 10. Design Intent (Non-Normative)

The long-term objective is a system where:

* Strategy emerges from learned signals,
* Resource allocation is adaptive and self-calibrating,
* Failures improve future control policies rather than being patched around.

This document exists to ensure that such emergence remains structurally possible.

---

**End of Contract**

---

## Appendix: Architecture Design Sketch (Non-normative)

This appendix is included for convenience. If any conflict exists, the normative sections above prevail.

---

## 模型架構設計

> 注意：若與任何 **(Normative Contract / Authoritative Specification)** 文件衝突，以合約文件為準。
> 對齊宣告：本文件中所有 MVP 固定值、硬 gate、預設策略，若影響行為，均屬於 **Routing, Gating, and Control Are Learnable** 所定義之 *temporary technical debt*：必須標註、隔離、可替換，且在 intended operation 下不得成為例行 policy（只能作 guardrail）。

### 1) 整體權重拓撲：Single Trunk 作為唯一大容量，Level Interfaces 以 Adapter/Head 形式掛載

#### 1.1 參數分塊（最終要打包的權重集合）

[

\Theta =

\Theta_{\text{embed+IR}} \cup

\Theta_{\text{single_trunk}} \cup

\bigcup_{k=0}^{6}\Theta_{k} \cup

\Theta_{\text{routers}}

]

- **(\Theta_{\text{embed+IR}})**：State IR / Program IR 相關 embedding、type embedding、pos/time embedding、以及 tokenization projection。
- **(\Theta_{\text{single_trunk}})**：主幹層（唯一的「大網路」；可為 attention/SSM/hybrid 等家族）。
- **(\Theta_{k})**：Level k 的 head / operator / adapter 權重（實作可關閉，但接口與 stub 必須存在）。
- **(\Theta_{\text{routers}})**：跨層路由與融合（gating / mixture / termination），避免策略落入手寫邏輯。

> 這個拓撲的核心好處：參數主要集中在 trunk；各 Level 只需薄 head + 小 adapter 即可「在權重裡」而不膨脹。
> 

---

### 2) 統一 State IR（必要：否則各層權重無法乾淨組合）

需要一個固定的狀態表示（不談任務類型，只談形狀/語義）：

#### 2.1 Token 集合（所有 tokens 經同一 trunk）

- **Object tokens**：(O \in \mathbb{R}^{N_o \times d})
- **Relation tokens**：(R \in \mathbb{R}^{N_r \times d})
- **Event tokens**：(X \in \mathbb{R}^{N_x \times d})
- **Task token**：(T \in \mathbb{R}^{1 \times d})
- **Global token**：(G \in \mathbb{R}^{1 \times d})
- **Macro tokens**：(M \in \mathbb{R}^{N_m \times d})

拼接後得到序列：

[

Z = [T; G; O; R; X; M] \in \mathbb{R}^{L \times d}

]

其中 (L = 2 + N_o + N_r + N_x (+ N_m))。

> Single trunk 的使用方式，就是把這個 (Z) 當作可學序列/狀態流處理；需加入 type embedding（object/relation/event/…）以維持 token 類型可辨識性。
> 

#### 2.2 Program IR 與 State IR 的邊界（硬性）

（對齊：State IR Canonical Spec、State IR Examples & Edge Cases、Level 2 Contract）

- **Program IR tokens**（例如 {P_i}）不屬於 State IR，且不得（即使暫時）拼接/注入到 (Z) 序列中。
- Program execution 的影響若要回到 State IR，只能以 **既有 State IR tokens 的 embedding 更新**（例如 (ΔZ)、gated residual、FiLM）或 **標量訊號** 的形式表達；不得把 Program IR tokens 當成 State IR token。

---

### 3) Single Trunk 具體組合方式（架構無關）

#### 3.1 Trunk 結構（建議：多層 trunk block + 小型 cross-token mixing）

為了兼顧 token 類型交互與效率，建議採用一個**最小混合**設計；具體核心可由 attention、SSM 或 hybrid 實作：

- **每層 block：**
    1. **Trunk core 層**（主計算；attention/SSM/hybrid 任一）
    2. **Token-mixer（輕量）**：可用極小的 gating 或低秩 mixing（不是全注意力）
    3. **RMSNorm + residual**

目標：在不引入第二大腦的前提下，仍能讓 object/relation/event 在同層互相影響。

#### 3.2 位置/時間處理

- 對靜態輸入，你需要的是「結構位置」，不是時間序列。
- 對互動/迭代任務，你需要 event-time encoding。

所以建議：

- **type embedding**（token 類型）
- **structure embedding**（例如 object 的幾何/拓撲位置、relation 的端點類型）
- **time embedding（event time）**：僅對 event/macro tokens 啟用

這些全是 (\Theta_{\text{embed+IR}}) 的一部分。

---

### 4) 各 Level 模組清單（接口必存在，實作可關閉）與 I/O

以下每個 Level 的模組都以「吃 trunk 輸出的 token 表徵」為主，避免另外訓練一套大網路。

#### Level 0：感知→對象/關係（接口必存在，實作可關閉）

**L0-1 Objectizer Head（權重）**

- Input：raw/grid/符號特徵（轉成 patch/cell token 也行）
- Output：初始 object tokens (O_0)
- 形式：小 encoder + slot 提取（或 set encoder）

**L0-2 Relation Inducer（權重）**

- Input：(O_0)
- Output：relation tokens (R_0)（或 adjacency-embedded tokens）

**L0-3 Eventizer（權重）**

- Input：前後狀態差分或互動訊號
- Output：event tokens (X_0)

> L0 的本質：把原始輸入轉成 State IR 的 token 集合（之後由 trunk 統一 contextualize）。
> 

---

#### Level 1：世界模型 dynamics（接口必存在，實作可關閉）

**L1-1 Dynamics Operator（權重）**

- Input：contextualized (Z_t)（trunk 輸出）與 action/tool token（若有）
- Output：(\hat{Z}_{t+1}) 或 (\Delta Z)

**L1-2 Constraint/Energy Head（權重）**

- Input：候選狀態 (\hat{Z})
- Output：constraint score / energy（可用於 pruning 或校正）

**L1-3 Uncertainty Head（權重）**

- Input：rollout 過程中 trunk 表徵
- Output：不確定性估計（標量或 token-wise）

> 若任務為靜態題型，L1 可以弱化到「可選 rollout」；但接口與 stub 仍需存在（可被 controller 置零使用）。
> 

---

#### Level 2：程序歸納（接口必存在，且不能是純手寫執行器）

這層是程序歸納的核心，你需要「proposal + executor + scorer」。

**L2-1 Program IR Embedding（權重）**

- program token / AST node embedding，供 proposal、memory、scorer 共用

**L2-2 Program Proposal Head（權重）**

- Input：(Z)（含 task/global/object…）
- Output：K 個 program 候選 {P_i}（Program IR tokens / sketch；不屬於 State IR，不得進入 (Z)）

**L2-3 Neural Primitive Executor（權重）**

要避免純 deterministic DSL 解譯器「不在權重裡」。最低要求是：primitive 的核心匹配/選擇要可學。

- Input：program + (Z)
- Output：對 State IR 的影響以 (ΔZ)/gated residual/FiLM 回寫到既有 tokens（形成 (Z')）；另可輸出候選解/子目標 token（不屬於 (Z)）
- 建議做法：primitive 是 learned operator（例如對 objects 的 soft select、對 relations 的 learned match、對 transforms 的 parameter head），控制流上限可先硬（guardrail；若為手寫 hard control，必須明確標註為 temporary technical debt 且可移除）。

**L2-4 Program Scorer/Value（權重）**

- Input：program + 執行後表徵
- Output：分數（用於 search / rerank）

---

#### Level 3：Meta-search / 資源控制（接口必存在，實作可關閉）

**L3-1 Budget Controller（權重）**

- Input：(Z)、L1 uncertainty、L6 confidence
- Output：((k_{\text{retrieve}}, d_{\text{search}}, h_{\text{rollout}}, \text{tool_on/off}, \text{stop}))

**L3-2 Node Expansion Scorer（權重）**

- Input：候選 program/state 節點表徵
- Output：擴展優先級、剪枝分數

**L3-3 Termination Head（權重）**

- Input：當前最佳候選 + 信心 + 成本特徵
- Output：停止/繼續（以及可選的“再試一次”策略）

> 這些 head 是「效率」的權重化來源；否則你會把策略寫死在程式裡。
> 

---

#### Level 4：Program / Concept Memory（接口必存在，實作可關閉）

你可以用外部存儲，但**讀寫策略必須由權重決定**。

**L4-1 Memory Key Encoder（權重）**

- Input：(Z)
- Output：query embedding (q)

**L4-2 Memory Read Fusion（權重）**

- Input：檢索到的 memory items（program/concept embeddings）+ (Z)
- Output：融合後 (Z')（通常是對 (G) 或 (T) 做 gated residual）

**L4-3 Write Gate（權重）**

- Input：當前解題過程表徵（program trace / verifier 結果）
- Output：是否寫入、寫入什麼類型（program/概念/片段）

**L4-4 Consolidation Gate（權重）**

- Input：新條目 vs 舊條目相似度表徵
- Output：merge / new / ignore

---

#### Level 5：Abstraction Management（接口必存在，實作可關閉）

**L5-1 Macro Selector（權重）**

- Input：(Z)（尤其 events）
- Output：macro update 的 gate、粒度選擇（micro vs macro）

**L5-2 Macro Updater（權重）**

- Input：一段 event/object 變化摘要
- Output：macro tokens (M) 或更新 (G^{macro})

> MVP 階段可以對 macro 粒度給強 prior（讓 gate 偏向固定），但 selector/updater 的權重仍要存在且可學；不得以手寫規則強制，若暫時存在硬規則，必須標註為 temporary technical debt（guardrail）。
> 

---

#### Level 6：Self-Monitoring / Meta-Credit（接口必存在，實作可關閉）

**L6-1 Verifier Head（權重）**

- Input：候選解/候選 program 執行後表徵
- Output：validity score、違反類型 logits（可多頭）

**L6-2 Confidence/Calibration（權重）**

- Input：verifier、controller 狀態、trunk 表徵
- Output：confidence（供 Level 3 使用）

**L6-3 Credit Router（權重）**

- Input：失敗/低信心時的診斷特徵
- Output：對 Level 0/1/2/4 的 credit routing / 診斷偏好分佈（供 Level 3 決策；不得直接設定 beam/depth/k 等 budget）

---

### 5) 跨層「路由/融合」權重：避免把控制寫死

至少需要三組 Router/Fusion 權重（獨立打包為 (\Theta_{\text{routers}})）：

1. **Level Invocation Router（權重）**
- 把 “要不要用 L1/L2/L4/L5” 變成 soft gate（由 controller 產生，但仍可有一層 router 做穩定化）
1. **State Fusion Router（權重）**
- 把 memory read、program execution、rollout 結果融合回 (G)/(T)/部分 object tokens（gated residual / FiLM；不得注入 Program IR tokens）
1. **Output Selection Router（權重）**
- 當多候選解存在：如何聚合、如何投票/選擇（learned reranker）

---

### 6) 權重共享策略

#### 6.1 預設建議：Trunk 占 70–90% 參數，Heads/Adapters 占 10–30%

- **Trunk**：L 層 trunk blocks（你可從 8–16 層起跳做 MVP）
- **每層插 adapter**：不同 Level 的 head 不再建獨立深網，而是：
    - 在 trunk 的少數層插入 **Level-specific adapter**（小 MLP 或低秩）
    - head 本身只做輸出映射（projection + 小 scoring MLP）

#### 6.2 Adapter 放置策略（實用且可控）

- 在 trunk 的 **前 1/3**：偏 L0/L1（表徵與 dynamics）
- 在 trunk 的 **中 1/3**：偏 L2（程序歸納）
- 在 trunk 的 **後 1/3**：偏 L3/L6（控制與驗證）

這讓同一 trunk 同時支援「表徵—推理—控制」，且參數不膨脹。

---

### 7) 混合精度：按模組分區

做法是「精度 map」隨權重包一起發布。建議分成三個 precision domain：

#### Domain A（BF16/FP16）：主表徵傳播

- (\Theta_{\text{single_trunk}})
- L0 的大部分 encoder/projection
- 大部分融合層（非排序決策）

#### Domain B（BF16 + FP32 accumulate）：長鏈條/rollout 穩定

- L1 dynamics state update（尤其多步 rollout 累積）
- L3 value/scoring 的核心累積（避免排序抖動）

#### Domain C（FP32 或局部 FP32）：離散邊界決策

- L6 verifier logits（accept/reject 類）
- L3 termination（stop/continue）
- L4 retrieval ranking 的最終打分（top-k 邊界很敏感）
- 任何 “if/loop/branch 的 gate” 最終 logits（避免 FP16 的臨界翻轉）

> 這個精度分區本身也算系統的一部分，應該作為 config 隨權重一起版本化。
> 

---

### 8) 最終的「模組圖」資料流（

以一次推理迭代（one reasoning cycle）為單位：

1. **L0**：raw → (O,R,X)（State IR tokens）
2. **Embed+Type/Time**：得到 (Z)
3. **Trunk**：(Z \rightarrow \tilde{Z})
4. **L4 read（受 L3 gate）**：(\tilde{Z} \rightarrow \tilde{Z}')
5. **L2 propose（受 L3 gate）**：(\tilde{Z}' \rightarrow {P_i}_{i=1..K})
6. **L1 rollout（受 L3 gate）**：(\tilde{Z}' \rightarrow \hat{Z}^{(h)})
7. **L2 execute + L2 score（受 L3 node scoring）**
8. **L6 verify + calibrate**：產生 validity/confidence
9. **L3 termination/budget update**：決定停止或下一輪（以及調整 k/d/h/tool）
10. **L4 write/consolidate（受 gate）**

> 註：{P_i} 為 Program IR tokens（不屬於 State IR）；executor 的影響僅能以 (ΔZ)/gated residual/FiLM 形式回寫到既有 State IR tokens，不能把 program tokens 注入 (Z)。

每一步都有對應接口；即便某些步驟在某次迭代被 gate 掉，接口與 stub 仍存在於系統註冊表中，符合接口優先約束。

---

### 9) 最小權重包清單

建議將權重以「模組註冊表」方式打包，至少包含：

1. **state_ir_embed/**：type/pos/time embeddings、IR projections
2. **trunk/**：block_0 … block_{L-1}（含 mixer、norm）
3. **level0/**：objectizer、relation_inducer、eventizer
4. **level1/**：dynamics、constraint_head、uncertainty_head
5. **level2/**：program_embed、proposal、executor_primitives、program_scorer
6. **level3/**：budget_controller、node_scorer、termination
7. **level4/**：key_encoder、read_fusion、write_gate、consolidation
8. **level5/**：macro_selector、macro_updater（pattern_summarizer（non-executable）可選）
9. **level6/**：verifier、calibration、credit_router
10. **routers/**：invocation_router、state_fusion_router、output_router
11. **config/**：d、各 token 上限、precision map、head dims、gating temperature 等

---

### 10) 先定的 3 個數值規格

#### 非規範註記（Non-normative）

All numeric caps in Section 10 are non-normative MVP defaults and must not be interpreted as architectural constraints.

### 10-1) 全系統 token hidden dim：**d**

#### 建議決策（MVP）

- **d = 512**

### 10-2) 各類 token 上限：(N_o, N_r, N_x, N_m)

#### MVP hard cap

| Token 類型 | 建議上限 | 說明 |
| --- | --- | --- |
| Object (N_o) | **32** | 多數 MVP 任務幾乎夠用；agent/數學可先 objectize 成有限集合 |
| Relation (N_r) | **64** | 通常 ≥ O×2 才不會卡圖結構 |
| Event (N_x) | **16** | MVP 階段只保留「關鍵變化」 |
| Macro (N_m) | **8** | abstraction management 的最小可用集 |

→ 總長度（不含 padding）約：

[

L \approx 2 + 32 + 64 + 16 + 8 = 122

]

### 10-3) Program IR token 長度上限：**Lp** 與候選數 **K**

#### 決策（MVP）

- **Program token 長度上限 Lp = 32**
- **Proposal beam / 候選數 K = 8**

---

### 11) Training Profile Link (Non-normative)

This architecture note is contract-aligned but not the source of truth for single-card training hyperparameters.

For the active single-H100 3B training profile, use:

- `docs/09_Training_Profile_SingleH100_3B.md`
- `docs/09_Training_Profile_SingleH100_3B.md`
- `docs/08_Training_Run_Governance.md`

For segment/apply/resume behavior, use:

- `docs/08_Training_Run_Governance.md`

For data mixture policy, use:

- `docs/07_Data_Mixture_and_Ingestion.md`

This section is documentation linkage only and does not change architecture contracts.

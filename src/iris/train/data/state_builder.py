from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ...schema import (
    AdjudicationState,
    ApplicabilityAudit,
    Branch,
    BudgetState,
    ConstraintRelation,
    ControlAction,
    ControlState,
    LemmaBinding,
    ProblemFrame,
    RequiredOutput,
    ScopeRef,
    StateIR,
    StrategyCandidate,
    Subgoal,
    SymbolEntry,
)


def _token_vector(token_id: int, hidden_dim: int) -> np.ndarray:
    index = np.arange(1, hidden_dim + 1, dtype=np.float32)
    token = float(int(token_id) + 1)
    return (
        np.sin(token * index * 0.013).astype(np.float32)
        + np.cos((token + 2.0) * index * 0.007).astype(np.float32)
    )


def _chunk_means(embeddings: np.ndarray, chunk_count: int) -> np.ndarray:
    if embeddings.shape[0] <= 0 or chunk_count <= 0:
        return np.zeros((0, embeddings.shape[1]), dtype=np.float32)
    chunk_count = min(chunk_count, embeddings.shape[0])
    chunks = np.array_split(embeddings, chunk_count)
    means = [np.mean(chunk, axis=0) for chunk in chunks if chunk.size > 0]
    if not means:
        return np.zeros((0, embeddings.shape[1]), dtype=np.float32)
    return np.stack(means, axis=0).astype(np.float32)


def _g_vector(token_ids: Sequence[int], embeddings: np.ndarray) -> np.ndarray:
    token_count = float(len(token_ids))
    unique_ratio = float(len(set(int(token) for token in token_ids))) / float(max(len(token_ids), 1))
    mean_abs = float(np.mean(np.abs(embeddings))) if embeddings.size > 0 else 0.0
    std = float(np.std(embeddings)) if embeddings.size > 0 else 0.0
    features = np.asarray([token_count, unique_ratio, mean_abs, std], dtype=np.float32)
    repeats = int(np.ceil(float(embeddings.shape[1]) / float(features.shape[0])))
    tiled = np.tile(features, repeats)[: embeddings.shape[1]]
    return tiled.reshape(1, embeddings.shape[1]).astype(np.float32)


def text_to_state_ir(
    *,
    text: str,
    tokenizer: Any,
    hidden_dim: int,
    max_input_tokens: int = 256,
) -> StateIR:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        token_ids = [0]
    token_ids = [int(token) for token in token_ids[: int(max(max_input_tokens, 1))]]

    embeddings = np.stack([_token_vector(token, hidden_dim) for token in token_ids], axis=0).astype(np.float32)

    t_rows = min(8, embeddings.shape[0])
    t_section = np.mean(embeddings[:t_rows], axis=0, keepdims=True).astype(np.float32)
    g_section = _g_vector(token_ids, embeddings)

    o_section = _chunk_means(embeddings, chunk_count=4)
    if o_section.shape[0] > 1:
        relation_rows = [o_section[idx + 1] - o_section[idx] for idx in range(o_section.shape[0] - 1)]
        r_section = np.stack(relation_rows, axis=0).astype(np.float32)
    else:
        r_section = np.zeros((0, hidden_dim), dtype=np.float32)

    if embeddings.shape[0] > 1:
        delta = embeddings[1:] - embeddings[:-1]
        x_section = _chunk_means(delta, chunk_count=2)
    else:
        x_section = np.zeros((0, hidden_dim), dtype=np.float32)

    m_section = np.mean(embeddings, axis=0, keepdims=True).astype(np.float32)

    problem_scope = ScopeRef(scope_kind="problem_global", scope_id="scope-problem")
    branch_scope = ScopeRef(
        scope_kind="branch_local",
        scope_id="branch-0",
        parent_scope_id=problem_scope.scope_id,
    )

    symbols = tuple(
        SymbolEntry(
            sy_id=f"sy-{index}",
            surface_form=f"sym_{index}",
            entity_kind="symbol",
            scope_ref=problem_scope,
            binding_state="unresolved",
            type_status="unknown",
            vector=row,
        )
        for index, row in enumerate(o_section)
    )
    constraints = tuple(
        ConstraintRelation(
            cg_id=f"cg-{index}",
            relation_type="dependency",
            arguments=(f"chunk_{index}", f"chunk_{index + 1}"),
            relation_status="candidate",
            vector=row,
        )
        for index, row in enumerate(r_section)
    )

    frontier = [
        Branch(
            branch_id="branch-0",
            branch_status="active",
            local_scope_ref=branch_scope,
            strategy_family="text_structuring",
            summary="bootstrap frontier branch",
            vector=(x_section[0] if x_section.shape[0] > 0 else g_section[0]),
        )
    ]
    for index, row in enumerate(x_section):
        frontier.append(
            Subgoal(
                subgoal_id=f"subgoal-{index}",
                branch_id="branch-0",
                goal_kind="interpret_segment",
                target_payload=f"segment_{index}",
                goal_status="candidate",
                vector=row,
            )
        )
    if not x_section.shape[0]:
        frontier.append(
            StrategyCandidate(
                strategy_id="strategy-0",
                branch_id="branch-0",
                strategy_family="fallback_parse",
                candidate_status="candidate",
                score=0.0,
                vector=g_section[0],
            )
        )

    memory = tuple(
        LemmaBinding(
            lm_id="lm-0",
            memory_kind="pattern",
            source_ref="text_builder",
            claim_signature="chunk_mean_pattern",
            binding_map={},
            applicability_audit=ApplicabilityAudit(
                audit_status="unchecked",
                required_conditions=(),
            ),
            retrieval_signal=0.0,
            vector=m_section[0],
        )
        for _ in range(1 if m_section.shape[0] > 0 else 0)
    )

    return StateIR(
        PF=ProblemFrame(
            task_type="interpret",
            target_spec=f"text_len:{len(text)}",
            required_output=RequiredOutput(
                output_kind="structured_state",
                answer_channel="structured_object",
                formality_level="informal",
            ),
            problem_assumptions=(),
            domain_tags=("text",),
            source_anchor_refs=(),
            frame_status="active",
            vector=t_section[0],
        ),
        SY=symbols,
        CG=constraints,
        FR=tuple(frontier),
        LM=memory,
        VS=(),
        CS=ControlState(
            selected_action=ControlAction(
                action_id="action-continue",
                action_type="continue",
                action_status="selected",
            ),
            budget_state=BudgetState(
                global_step_budget_remaining=max(int(max_input_tokens) - len(token_ids), 0)
            ),
            runtime_status="in_progress",
            uncertainty_state="estimated",
            escalation_state="inactive",
            adjudication_state=AdjudicationState(
                task_adjudication_policy_id="task-family-proof-natural-language-default-v1",
                adjudication_status="pending",
            ),
            vector=g_section[0],
        ),
    )

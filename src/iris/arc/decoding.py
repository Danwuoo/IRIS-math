from __future__ import annotations

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    denom = float(np.sum(exp))
    if denom <= 0.0:
        return np.full_like(logits, fill_value=1.0 / float(logits.shape[0]))
    return exp / denom


def _ensure_color_projection(seq_w: np.ndarray, ctrl_w: np.ndarray) -> np.ndarray:
    hidden_dim = int(seq_w.shape[0])
    if seq_w.shape[1] >= 10:
        projection = seq_w[:, :10]
    else:
        projection = np.zeros((hidden_dim, 10), dtype=np.float32)
        projection[:, : seq_w.shape[1]] = seq_w
        remain = 10 - seq_w.shape[1]
        if remain > 0 and ctrl_w.shape[1] > 0:
            tiled = np.tile(ctrl_w[:, : min(ctrl_w.shape[1], remain)], int(np.ceil(remain / float(ctrl_w.shape[1]))))
            projection[:, seq_w.shape[1] : 10] = tiled[:, :remain]
    return projection.astype(np.float32)


class ArcDecoder:
    def __init__(self, seq_w: np.ndarray, ctrl_w: np.ndarray, ctrl_b: np.ndarray) -> None:
        self.color_w = _ensure_color_projection(seq_w=np.asarray(seq_w), ctrl_w=np.asarray(ctrl_w))
        if ctrl_b.shape[0] >= 10:
            self.color_b = np.asarray(ctrl_b[:10], dtype=np.float32)
        else:
            self.color_b = np.zeros((10,), dtype=np.float32)
            self.color_b[: ctrl_b.shape[0]] = np.asarray(ctrl_b, dtype=np.float32)
        self.row_bias = np.asarray(self.color_w[0], dtype=np.float32)
        self.col_bias = np.asarray(self.color_w[1 % self.color_w.shape[0]], dtype=np.float32)

    @classmethod
    def from_model_params(cls, trunk_params: dict) -> "ArcDecoder":
        return cls(
            seq_w=np.asarray(trunk_params["seq_w"], dtype=np.float32),
            ctrl_w=np.asarray(trunk_params["ctrl_w"], dtype=np.float32),
            ctrl_b=np.asarray(trunk_params["ctrl_b"], dtype=np.float32),
        )

    def decode_grid(
        self,
        sequence: np.ndarray,
        *,
        output_shape: tuple[int, int],
    ) -> tuple[list[list[int]], float]:
        rows, cols = output_shape
        rows = max(int(rows), 1)
        cols = max(int(cols), 1)
        grid: list[list[int]] = []
        conf_values: list[float] = []
        token_count = int(sequence.shape[0])
        for row_idx in range(rows):
            row_values: list[int] = []
            for col_idx in range(cols):
                token_idx = (row_idx * cols + col_idx) % max(token_count, 1)
                token = np.asarray(sequence[token_idx], dtype=np.float32)
                logits = token @ self.color_w + self.color_b
                logits = logits + (row_idx * 0.03) * self.row_bias + (col_idx * 0.03) * self.col_bias
                probs = _softmax(logits.astype(np.float64))
                color = int(np.argmax(probs))
                row_values.append(color)
                conf_values.append(float(np.max(probs)))
            grid.append(row_values)
        confidence = float(np.mean(conf_values)) if conf_values else 0.0
        return grid, confidence


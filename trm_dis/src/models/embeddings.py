"""
Embedding layer for football episode sequences
Combines categorical and continuous features
"""
import torch
import torch.nn as nn


class EpisodeEmbedding(nn.Module):
    """
    Embeds action sequences into d_model dimensions

    Combines:
    - Categorical embeddings (action type, result type, is_home)
    - Continuous projections (start_x, start_y, time, padding_flag)
    """

    def __init__(
        self,
        d_model: int,
        n_action_types: int = 26,
        n_result_types: int = 9,  # 8 + NaN
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model

        # Categorical embeddings (each gets d_model // 4 dimensions)
        emb_dim = d_model // 4
        self.type_emb = nn.Embedding(n_action_types, emb_dim)
        self.result_emb = nn.Embedding(n_result_types, emb_dim)
        self.home_emb = nn.Embedding(2, emb_dim)  # Binary: home or away

        # Continuous projection (4 features → d_model // 4)
        self.cont_proj = nn.Linear(4, emb_dim)

        # Fusion layer: 4 * emb_dim → d_model
        self.fusion = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        continuous: torch.Tensor,  # [B, seq_len, 4]
        categorical: torch.Tensor   # [B, seq_len, 3] - (type_id, result_id, is_home)
    ) -> torch.Tensor:
        """
        Embed episode sequences

        Args:
            continuous: [B, seq_len, 4] - (start_x, start_y, time_norm, padding_flag)
            categorical: [B, seq_len, 3] - (type_id, result_id, is_home)

        Returns:
            embeddings: [B, seq_len, d_model]
        """
        B, seq_len, _ = continuous.shape

        # Categorical embeddings
        type_ids = categorical[:, :, 0]  # [B, seq_len]
        result_ids = categorical[:, :, 1]
        is_home = categorical[:, :, 2]

        type_emb = self.type_emb(type_ids)      # [B, seq_len, emb_dim]
        result_emb = self.result_emb(result_ids)
        home_emb = self.home_emb(is_home.long())

        # Continuous projection
        cont_emb = self.cont_proj(continuous)  # [B, seq_len, emb_dim]

        # Concatenate all embeddings
        combined = torch.cat([type_emb, result_emb, home_emb, cont_emb], dim=-1)
        # [B, seq_len, d_model]

        # Fusion and normalization
        fused = self.fusion(combined)
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)

        return fused

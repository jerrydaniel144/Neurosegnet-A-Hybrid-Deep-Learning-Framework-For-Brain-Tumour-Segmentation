import torch
import torch.nn.functional as F
import numpy as np


class AttentionRollout:
    def __init__(self, model, head_fusion='mean', discard_ratio=0.0):
        # Initializes the AttentionRollout explainer.
       
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attentions = []

        # Register hooks on transformer blocks
        for block in self._get_transformer_blocks():
            block.attn.register_forward_hook(self._save_attention)

    def _get_transformer_blocks(self):
        #Get transformer blocks that contain self-attention.
        
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'blocks'):
            return self.model.transformer.blocks
        else:
            raise AttributeError("Model does not have accessible transformer blocks")

    def _save_attention(self, module, input, output):
        """
        Hook to save attention maps during forward pass.
        """
        attn = output[1] if isinstance(output, tuple) else output
        self.attentions.append(attn.detach().cpu())

    def compute_rollout_attention(self):
        """
        Computes the attention rollout matrix by multiplying attention maps across layers.
        Returns:
            torch.Tensor: shape (batch, tokens, tokens)
        """
        result = torch.eye(self.attentions[0].size(-1))  # Identity matrix

        with torch.no_grad():
            for attn in self.attentions:
                if self.head_fusion == 'mean':
                    attn_fused = attn.mean(dim=1)
                elif self.head_fusion == 'max':
                    attn_fused = attn.max(dim=1)[0]
                else:
                    raise ValueError("head_fusion must be 'mean' or 'max'")

                # Apply discard mask
                if self.discard_ratio > 0:
                    batch_size, tokens, _ = attn_fused.size()
                    flat = attn_fused.view(batch_size, -1)
                    topk = int(flat.size(1) * (1 - self.discard_ratio))
                    _, indices = flat.topk(topk, dim=-1)
                    mask = torch.zeros_like(flat).scatter_(-1, indices, 1).view_as(attn_fused)
                    attn_fused = attn_fused * mask

                # Normalize attention
                attn_fused = attn_fused / (attn_fused.sum(dim=-1, keepdim=True) + 1e-6)

                result = torch.matmul(attn_fused, result)

        return result  # (batch_size, tokens, tokens)

    def reset(self):
        """
        Reset stored attention maps.
        """
        self.attentions = []
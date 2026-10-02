"""Component ablations of MetaDC-INR for the Scientific Reports revision (R1.4 / R2.1).

This module does NOT modify model.py. It imports the shared primitives from it, so
SEBlock, PositionalEncoding and the sampler are literally the same code, and defines one
class whose `ablate` argument selects a component ablation.

With ablate=None the module graph, parameter names and forward arithmetic are identical to
model.InRetouchNR, so meta_model_ft.pth loads with strict=True and outputs match bit for bit.

    None            reference model (identical to model.InRetouchNR)
    'no_global'     FiLM conditioned on the local context field only
    'no_local'      FiLM conditioned on the broadcast global descriptor only
    'naive_coords'  raw (dx,dy,x,y,R,G,B) replaces the 107-d Fourier encoding
    'single_scale'  caller passes x_ctx=None, so one 13x13 patch feeds both paths
    'concat_cond'   context concatenated to the MLP input instead of FiLM modulation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import SEBlock, PositionalEncoding  # shared, not reimplemented

ABLATIONS = (None, 'no_global', 'no_local', 'naive_coords', 'single_scale', 'concat_cond')


class InRetouchNRAblate(nn.Module):
    def __init__(self, hidden_dim=128, ablate=None):
        super(InRetouchNRAblate, self).__init__()
        assert ablate in ABLATIONS, f"unknown ablate={ablate!r}; expected one of {ABLATIONS}"
        self.hidden_dim = hidden_dim
        self.ablate = ablate

        if ablate != 'no_local':
            self.local_context = nn.Sequential(
                nn.Conv2d(3, hidden_dim // 2, 3, padding=2, dilation=2, padding_mode='reflect'),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, padding=4, dilation=4, padding_mode='reflect'),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, padding=8, dilation=8, padding_mode='reflect'),
                SEBlock(hidden_dim // 2),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_dim // 2, hidden_dim, 1)
            )

        if ablate != 'no_global':
            self.global_context = nn.Sequential(
                nn.AdaptiveAvgPool2d((8, 8)),
                nn.Flatten(),
                nn.Linear(3 * 8 * 8, hidden_dim * 2),
                nn.SiLU(inplace=True),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )

        # num_frequencies=0 leaves freq_bands empty, so PositionalEncoding returns the raw
        # input and the Gaussian dampening term is never reached.
        nf_rel, nf_abs, nf_col = (0, 0, 0) if ablate == 'naive_coords' else (10, 3, 8)
        self.rel_coord_pe = PositionalEncoding(num_frequencies=nf_rel, include_input=True)
        self.abs_coord_pe = PositionalEncoding(num_frequencies=nf_abs, include_input=True)
        self.color_pe = PositionalEncoding(num_frequencies=nf_col, include_input=True)

        self.coord_dim = (2 + 4 * nf_rel) + (2 + 4 * nf_abs)   # 42 + 14 = 56 by default
        self.color_dim = 3 + 6 * nf_col                        # 51 by default
        self.struct_dim = self.coord_dim + self.color_dim      # 107 by default

        n_ctx = 1 if ablate in ('no_local', 'no_global') else 2
        self.cond_dim = hidden_dim * n_ctx
        self.film_params_dim = (hidden_dim * 2 * 2) + (hidden_dim * 2)
        if ablate != 'concat_cond':
            self.film_gen = nn.Sequential(
                nn.Linear(self.cond_dim, hidden_dim * 2),
                nn.SiLU(inplace=True),
                nn.Linear(hidden_dim * 2, self.film_params_dim)
            )

        self.mlp_in_dim = self.struct_dim + (self.cond_dim if ablate == 'concat_cond' else 0)
        self.mlp_layer1 = nn.Linear(self.mlp_in_dim, hidden_dim * 2)
        self.mlp_layer2 = nn.Linear(hidden_dim * 2, hidden_dim)

        # DUAL-PATH HEAD: 12 (Matrix) + 3 (Detail Residual) = 15
        self.mlp_head = nn.Linear(hidden_dim, 15)

        self._initialize_weights()

    def _initialize_weights(self):
        # Deliberately byte-identical to model.InRetouchNR._initialize_weights, INCLUDING the
        # bias indices 0/4/8. Those do not produce an identity affine map (row-major (3,4)
        # reshape sends index k to M[k//4, k%4], so they fill the first column and map
        # [R,G,B,1] -> [R,R,R]); identity would need 0/5/10. Preserved on purpose: the
        # reference model A1 was meta-trained from this starting point, and changing it here
        # would add a second difference between A1 and the ablation rows.
        if hasattr(self, 'film_gen'):
            nn.init.zeros_(self.film_gen[-1].weight)
            nn.init.zeros_(self.film_gen[-1].bias)
        nn.init.zeros_(self.mlp_head.weight)
        with torch.no_grad():
            self.mlp_head.bias.fill_(0)
            self.mlp_head.bias[0] = 1.0  # m11 (as in model.py)
            self.mlp_head.bias[4] = 1.0  # m22 (as in model.py)
            self.mlp_head.bias[8] = 1.0  # m33 (as in model.py)

    def forward(self, x, x_ctx=None, global_image=None):
        B, C, H, W = x.shape
        x_ctx = x_ctx if x_ctx is not None else x

        parts = []
        if self.ablate != 'no_local':
            ctx_feat_all = self.local_context(x_ctx)
            _, cc, ch, cw = ctx_feat_all.shape
            sh, sw = (ch - H) // 2, (cw - W) // 2
            parts.append(ctx_feat_all[:, :, sh:sh + H, sw:sw + W])

        if self.ablate != 'no_global':
            g_img = global_image if global_image is not None else x_ctx
            global_feat = self.global_context(g_img)
            parts.append(global_feat.view(-1, self.hidden_dim, 1, 1).expand(B, -1, H, W))

        if hasattr(self, 'current_offsets') and self.current_offsets is not None:
            rel_feat = self.rel_coord_pe(self.current_offsets * 2.0)
            abs_feat = self.abs_coord_pe(self.current_abs_coords, pixel_size=0.02)
        else:
            dummy_off = torch.zeros(B, 2, H, W, device=x.device)
            rel_feat = self.rel_coord_pe(dummy_off)
            ys = torch.linspace(-1, 1, H, device=x.device)
            xs = torch.linspace(-1, 1, W, device=x.device)
            gy, gx = torch.meshgrid(ys, xs, indexing='ij')
            grid_abs = torch.stack([gx, gy], dim=-1).unsqueeze(0).permute(0, 3, 1, 2).expand(B, -1, -1, -1)
            abs_feat = self.abs_coord_pe(grid_abs, pixel_size=0.02)

        coord_feat = torch.cat([rel_feat, abs_feat], dim=1)
        color_feat = self.color_pe(x)

        mod_input = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        mod_flat = mod_input.permute(0, 2, 3, 1).reshape(-1, self.cond_dim)

        struct_feat = torch.cat([coord_feat, color_feat], dim=1)
        h = struct_feat.permute(0, 2, 3, 1).reshape(-1, self.struct_dim)

        if self.ablate == 'concat_cond':
            h = torch.cat([h, mod_flat], dim=1)
            h = F.silu(self.mlp_layer1(h))
            h = F.silu(self.mlp_layer2(h))
        else:
            film_params = self.film_gen(mod_flat)
            g1 = film_params[:, :self.hidden_dim * 2]
            b1 = film_params[:, self.hidden_dim * 2: self.hidden_dim * 4]
            h = self.mlp_layer1(h)
            h = h * (1.0 + g1) + b1
            h = F.silu(h)

            g2 = film_params[:, self.hidden_dim * 4: self.hidden_dim * 5]
            b2 = film_params[:, self.hidden_dim * 5:]
            h = self.mlp_layer2(h)
            h = h * (1.0 + g2) + b2
            h = F.silu(h)

        head_out = self.mlp_head(h)
        matrix_flat = head_out[:, :12]
        detail_flat = head_out[:, 12:]

        M = matrix_flat.view(B, H, W, 3, 4)
        ones = torch.ones(B, 1, H, W, device=x.device)
        x_h = torch.cat([x, ones], dim=1).permute(0, 2, 3, 1).unsqueeze(-1)
        out_matrix = torch.matmul(M, x_h).squeeze(-1).permute(0, 3, 1, 2)

        detail_res = detail_flat.view(B, H, W, 3).permute(0, 3, 1, 2)

        final_out = out_matrix + torch.tanh(detail_res) * 0.1

        return torch.clamp(final_out, 0, 1)

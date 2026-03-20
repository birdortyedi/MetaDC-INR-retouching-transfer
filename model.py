import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class PositionalEncoding(nn.Module):
    def __init__(self, num_frequencies=10, include_input=True):
        super(PositionalEncoding, self).__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.register_buffer('freq_bands', 2**torch.linspace(0, num_frequencies - 1, num_frequencies))

    def forward(self, x, pixel_size=None):
        out = [x] if self.include_input else []
        for freq in self.freq_bands:
            xf = x * (freq * math.pi)
            if pixel_size is not None:
                k = freq * math.pi
                var = (pixel_size**2) / 12.0
                damp = torch.exp(-0.5 * (k**2) * var)
                out.append(torch.sin(xf) * damp)
                out.append(torch.cos(xf) * damp)
            else:
                out.append(torch.sin(xf))
                out.append(torch.cos(xf))
        
        if x.dim() == 4:
            return torch.cat(out, dim=1)
        else:
            return torch.cat(out, dim=-1)


class InRetouchNR(nn.Module):
    def __init__(self, hidden_dim=128):
        super(InRetouchNR, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.local_context = nn.Sequential(
            nn.Conv2d(3, hidden_dim // 2, 3, padding=2, dilation=2, padding_mode='reflect'), # RF 5
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, padding=4, dilation=4, padding_mode='reflect'), # RF 13
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, padding=8, dilation=8, padding_mode='reflect'), # RF 29
            SEBlock(hidden_dim // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 1)
        )
        
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, hidden_dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.rel_coord_pe = PositionalEncoding(num_frequencies=10, include_input=True)  # 42
        self.abs_coord_pe = PositionalEncoding(num_frequencies=3, include_input=True)   # 14
        self.color_pe = PositionalEncoding(num_frequencies=8, include_input=True)      # 51
        
        self.coord_dim = 42 + 14  # 56
        self.color_dim = 51
        
        self.cond_dim = hidden_dim + hidden_dim  # 256
        self.film_params_dim = (hidden_dim * 2 * 2) + (hidden_dim * 2)
        self.film_gen = nn.Sequential(
            nn.Linear(self.cond_dim, hidden_dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim * 2, self.film_params_dim)
        )
        
        self.mlp_in_dim = self.coord_dim + self.color_dim  # 107
        self.mlp_layer1 = nn.Linear(self.mlp_in_dim, hidden_dim * 2)
        self.mlp_layer2 = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # DUAL-PATH HEAD: 12 (Matrix) + 3 (Detail Residual) = 15
        self.mlp_head = nn.Linear(hidden_dim, 15) 
        
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)
        nn.init.zeros_(self.mlp_head.weight)
        with torch.no_grad():
            self.mlp_head.bias.fill_(0)
            # Matrix component starts at Identity
            self.mlp_head.bias[0] = 1.0  # m11
            self.mlp_head.bias[4] = 1.0  # m22
            self.mlp_head.bias[8] = 1.0  # m33
            # Detail components (12, 13, 14) stay at 0.0

    def forward(self, x, x_ctx=None, global_image=None):
        B, C, H, W = x.shape
        x_ctx = x_ctx if x_ctx is not None else x
        
        # Local context
        ctx_feat_all = self.local_context(x_ctx)
        _, cc, ch, cw = ctx_feat_all.shape
        sh, sw = (ch - H) // 2, (cw - W) // 2
        local_feat = ctx_feat_all[:, :, sh:sh+H, sw:sw+W]
        
        # Global context
        g_img = global_image if global_image is not None else x_ctx
        global_feat = self.global_context(g_img) 
        global_feat = global_feat.view(-1, self.hidden_dim, 1, 1).expand(B, -1, H, W)
        
        # Hybrid Coordinates
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
        
        # FiLM Modulation
        mod_input = torch.cat([local_feat, global_feat], dim=1)
        mod_flat = mod_input.permute(0, 2, 3, 1).reshape(-1, self.cond_dim)
        film_params = self.film_gen(mod_flat)
        
        # Modulated Structure MLP
        struct_feat = torch.cat([coord_feat, color_feat], dim=1)
        h = struct_feat.permute(0, 2, 3, 1).reshape(-1, self.mlp_in_dim)
        
        g1 = film_params[:, :self.hidden_dim*2]
        b1 = film_params[:, self.hidden_dim*2 : self.hidden_dim*4]
        h = self.mlp_layer1(h)
        h = h * (1.0 + g1) + b1
        h = F.silu(h)
        
        g2 = film_params[:, self.hidden_dim*4 : self.hidden_dim*5]
        b2 = film_params[:, self.hidden_dim*5 : ]
        h = self.mlp_layer2(h)
        h = h * (1.0 + g2) + b2
        h = F.silu(h)
        
        # 6. DUAL-PATH OUTPUT
        head_out = self.mlp_head(h)
        matrix_flat = head_out[:, :12]
        detail_flat = head_out[:, 12:]
        
        # Path A: Matrix transformation
        M = matrix_flat.view(B, H, W, 3, 4)
        ones = torch.ones(B, 1, H, W, device=x.device)
        x_h = torch.cat([x, ones], dim=1).permute(0, 2, 3, 1).unsqueeze(-1)
        out_matrix = torch.matmul(M, x_h).squeeze(-1).permute(0, 3, 1, 2)
        
        # Path B: Detail Residual
        detail_res = detail_flat.view(B, H, W, 3).permute(0, 3, 1, 2)
        
        final_out = out_matrix + torch.tanh(detail_res) * 0.1
        
        return torch.clamp(final_out, 0, 1)


def get_subpixel_sampling_windows(image_tensor, batch_size=484, window_size=13, centers=None):
    B, C, H, W = image_tensor.shape
    device = image_tensor.device
    unit_h, unit_w = 2.0 / H, 2.0 / W
    if centers is None:
        y_c = torch.rand(batch_size, device=device) * 1.8 - 0.9
        x_c = torch.rand(batch_size, device=device) * 1.8 - 0.9
        centers = torch.stack([x_c, y_c], dim=-1)
    else:
        x_c, y_c = centers[:, 0], centers[:, 1]
    extent_h, extent_w = ((window_size - 1) / 2.0) * unit_h, ((window_size - 1) / 2.0) * unit_w
    grid_y = torch.linspace(-extent_h, extent_h, window_size, device=device)
    grid_x = torch.linspace(-extent_w, extent_w, window_size, device=device)
    gy, gx = torch.meshgrid(grid_y, grid_x, indexing='ij')
    abs_y = y_c.view(batch_size, 1, 1) + gy.view(1, window_size, window_size)
    abs_x = x_c.view(batch_size, 1, 1) + gx.view(1, window_size, window_size)
    grid_abs = torch.stack([abs_x, abs_y], dim=-1)
    y_idx = torch.round((y_c + 1.0) / unit_h - 0.5)
    x_idx = torch.round((x_c + 1.0) / unit_w - 0.5)
    dy, dx = (y_c + 1.0) / unit_h - 0.5 - y_idx, (x_c + 1.0) / unit_w - 0.5 - x_idx
    offsets = torch.stack([dx, dy], dim=-1)
    pixel_y_lin = torch.linspace(- (window_size // 2), (window_size // 2), window_size, device=device)
    pixel_x_lin = torch.linspace(- (window_size // 2), (window_size // 2), window_size, device=device)
    gy_p, gx_p = torch.meshgrid(pixel_y_lin, pixel_x_lin, indexing='ij')
    idx_y, idx_x = y_idx.view(batch_size, 1, 1) + gy_p.view(1, window_size, window_size), x_idx.view(batch_size, 1, 1) + gx_p.view(1, window_size, window_size)
    norm_y, norm_x = (idx_y + 0.5) * unit_h - 1.0, (idx_x + 0.5) * unit_w - 1.0
    grid_sharp = torch.stack([norm_x, norm_y], dim=-1)
    patches_sharp = F.grid_sample(image_tensor.expand(batch_size, -1, -1, -1), grid_sharp, mode='nearest', padding_mode='reflection', align_corners=False)
    patches_smooth = F.grid_sample(image_tensor.expand(batch_size, -1, -1, -1), grid_abs, mode='bilinear', padding_mode='reflection', align_corners=False)
    
    return patches_sharp, patches_smooth, offsets, grid_abs.permute(0, 3, 1, 2), centers

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

MIN_NUM_PATCHES = 16


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads,
                                                dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(
                    dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

# class DecoderTransformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=None):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 Residual(PreNorm(dim, Attention(dim, heads=heads,
#                                                 dim_head=dim_head, dropout=dropout))),
#                 # Residual(PreNorm(dim, FeedForward(
#                 #     dim, mlp_dim, dropout=dropout)))
#             ]))

#     def forward(self, x, mask=None):
#         for attn, ff in self.layers:
#             x = attn(x, mask=mask)
#             x = ff(x)
#         return x


class ViT3D(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool='cls', channels=1, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert all([each_dimension % patch_size ==
                    0 for each_dimension in image_size])
        num_patches = (image_size[0] // patch_size) * \
            (image_size[1] // patch_size)*(image_size[2] // patch_size)
        patch_dim = channels * patch_size ** 3
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {
            'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_size = patch_size
        self.dim = dim

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        # self.pool = pool
        # self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )

    def forward(self, img, mask=None):
        p = self.patch_size

        x = rearrange(
            img, 'b c (x p1) (y p2) (z p3) -> b (x y z) (p1 p2 p3 c)', p1=p, p2=p, p3=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)
        # import pdb;pdb.set_trace()

        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # x = self.to_latent(x)
        return x #self.mlp_head(x)


# # class MaeViT3D(nn.Module):
#     """ Masked Autoencoder with VisionTransformer backbone
#     """
#     def __init__(self, 
#                 # encoder:
#                 image_size,
#                 patch_size=16,
#                 channels=1,
#                 dim=1024, 
#                 depth=24, 
#                 heads=16,
#                 dim_head=64,
#                 mlp_dim=2048,
#                 # dropout=0.1,
#                 # emb_dropout=0.1,
#                 # decoder:
#                 decoder_dim=512, 
#                 decoder_depth=8, 
#                 decoder_heads=16,
#                 decoder_dim_head=64,
#                 mlp_ratio=4., 
#                 norm_layer=nn.LayerNorm,
#                 norm_pix_loss=False):
#         super().__init__()


#         # MAE encoder specifics
#         num_patches = (image_size[0] // patch_size)*(image_size[1] // patch_size)*(image_size[2] // patch_size)
#         patch_dim = channels * patch_size ** 3

#         self.patch_size = patch_size
#         self.dim = dim

#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim), requires_grad=False)
#         self.patch_to_embedding = nn.Linear(patch_dim, dim)
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
#         # self.dropout = nn.Dropout(emb_dropout)

#         self.transformer = Transformer(
#             dim, 
#             depth, 
#             heads, 
#             dim_head, 
#             mlp_dim, 
#             # dropout
#             )

#         # MAE decoder specifics
#         self.decoder_embed = nn.Linear(dim, decoder_dim, bias=True)
#         self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
#         self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim), requires_grad=False)

#         self.decoder_transformer = DecoderTransformer(
#             decoder_dim, 
#             decoder_depth, 
#             decoder_heads, 
#             decoder_dim_head, 
#             # mlp_dim, 
#             # dropout
#             )

#         # self.decoder_blocks = nn.ModuleList([
#         #     Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
#         #     for i in range(decoder_depth)])

#         self.decoder_norm = norm_layer(decoder_dim)
#         self.decoder_pred = nn.Linear(decoder_dim, patch_dim, bias=True) # decoder to patch
#         # --------------------------------------------------------------------------

#         self.norm_pix_loss = norm_pix_loss

#         self.initialize_weights()

#     def initialize_weights(self):
#         # initialization
#         # initialize (and freeze) pos_embed by sin-cos embedding
#         pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

#         decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
#         self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

#         # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
#         w = self.patch_embed.proj.weight.data
#         torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

#         # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
#         torch.nn.init.normal_(self.cls_token, std=.02)
#         torch.nn.init.normal_(self.mask_token, std=.02)

#         # initialize nn.Linear and nn.LayerNorm
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def patchify(self, imgs):
#         """
#         imgs: (N, 3, H, W)
#         x: (N, L, patch_size**2 *3)
#         """
#         p = self.patch_embed.patch_size[0]
#         assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

#         h = w = imgs.shape[2] // p
#         x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
#         x = torch.einsum('nchpwq->nhwpqc', x)
#         x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
#         return x

#     def unpatchify(self, x):
#         """
#         x: (N, L, patch_size**2 *3)
#         imgs: (N, 3, H, W)
#         """
#         p = self.patch_embed.patch_size[0]
#         h = w = int(x.shape[1]**.5)
#         assert h * w == x.shape[1]
        
#         x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
#         x = torch.einsum('nhwpqc->nchpwq', x)
#         imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
#         return imgs

#     def random_masking(self, x, mask_ratio):
#         """
#         Perform per-sample random masking by per-sample shuffling.
#         Per-sample shuffling is done by argsort random noise.
#         x: [N, L, D], sequence
#         """
#         N, L, D = x.shape  # batch, length, dim
#         len_keep = int(L * (1 - mask_ratio))
        
#         noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
#         # sort noise for each sample
#         ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
#         ids_restore = torch.argsort(ids_shuffle, dim=1)

#         # keep the first subset
#         ids_keep = ids_shuffle[:, :len_keep]
#         x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

#         # generate the binary mask: 0 is keep, 1 is remove
#         mask = torch.ones([N, L], device=x.device)
#         mask[:, :len_keep] = 0
#         # unshuffle to get the binary mask
#         mask = torch.gather(mask, dim=1, index=ids_restore)

#         return x_masked, mask, ids_restore

#     def forward_encoder(self, x, mask_ratio):
#         # embed patches
#         x = self.patch_embed(x)

#         # add pos embed w/o cls token
#         x = x + self.pos_embed[:, 1:, :]

#         # masking: length -> length * mask_ratio
#         x, mask, ids_restore = self.random_masking(x, mask_ratio)

#         # append cls token
#         cls_token = self.cls_token + self.pos_embed[:, :1, :]
#         cls_tokens = cls_token.expand(x.shape[0], -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)

#         # apply Transformer blocks
#         for blk in self.blocks:
#             x = blk(x)
#         x = self.norm(x)

#         return x, mask, ids_restore

#     def forward_decoder(self, x, ids_restore):
#         # embed tokens
#         x = self.decoder_embed(x)

#         # append mask tokens to sequence
#         mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
#         x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
#         x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
#         x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

#         # add pos embed
#         x = x + self.decoder_pos_embed

#         # apply Transformer blocks
#         for blk in self.decoder_blocks:
#             x = blk(x)
#         x = self.decoder_norm(x)

#         # predictor projection
#         x = self.decoder_pred(x)

#         # remove cls token
#         x = x[:, 1:, :]

#         return x

#     def forward_loss(self, imgs, pred, mask):
#         """
#         imgs: [N, 3, H, W]
#         pred: [N, L, p*p*3]
#         mask: [N, L], 0 is keep, 1 is remove, 
#         """
#         target = self.patchify(imgs)
#         if self.norm_pix_loss:
#             mean = target.mean(dim=-1, keepdim=True)
#             var = target.var(dim=-1, keepdim=True)
#             target = (target - mean) / (var + 1.e-6)**.5

#         loss = (pred - target) ** 2
#         loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

#         loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
#         return loss

#     def forward(self, imgs, mask_ratio=0.75):
#         latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
#         pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
#         loss = self.forward_loss(imgs, pred, mask)
#         return loss, pred, mask
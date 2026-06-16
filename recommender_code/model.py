import torch
import torch.nn as nn
from recbole.model.layers import TransformerEncoder


class PlanarFlow(nn.Module):
    """
    Planar normalizing flow: z = z_0 + u * tanh(w^T z_0 + b)
    Enables multi-modal posterior q(z|x) from base Gaussian q_0(z_0|x).
    Log-det: log |1 + u^T * (1 - tanh^2) * w|
    """

    def __init__(self, latent_dim, init_scale=0.001):
        super().__init__()
        self.w = nn.Parameter(torch.randn(latent_dim) * init_scale)
        self.u = nn.Parameter(torch.randn(latent_dim) * init_scale)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, z_0):
        """
        Args:
            z_0: [..., latent_dim]

        Returns:
            z: [..., latent_dim]
            log_det: [...]  (log |det dz/dz_0|)
        """

        act = torch.sum(z_0 * self.w, dim=-1, keepdim=True) + self.b
        h = torch.tanh(act)
        h_prime = 1.0 - h ** 2
        z = z_0 + self.u * h
        psi = h_prime * self.w
        det = 1.0 + torch.sum(self.u * psi, dim=-1)
        log_det = torch.log(torch.abs(det) + 1e-8)

        return z, log_det


class PlanarFlowStack(nn.Module):
    """Stack of K Planar Flow layers for more expressive posterior."""

    def __init__(self, latent_dim, num_flows=2, flow_init_scale=0.001):
        super().__init__()
        self.flows = nn.ModuleList(
            [PlanarFlow(latent_dim, init_scale=flow_init_scale)
             for _ in range(num_flows)]
        )

    def forward(self, z_0):
        z = z_0
        log_det_total = 0.0

        for flow in self.flows:
            z, log_det = flow(z)
            log_det_total = log_det_total + log_det

        return z, log_det_total


class RadialFlow(nn.Module):
    """
    Radial normalizing flow: z = z_0 + beta * h(alpha, r) * (z_0 - z_ref)
    where r = ||z_0 - z_ref||, h(alpha, r) = 1 / (alpha + r).
    Enables radial contractions/expansions around z_ref.
    Log-det: (d-1)*log(1 + beta*h) + log(1 + beta*alpha*h^2).
    """

    def __init__(self, latent_dim, init_scale=0.001):
        super().__init__()
        self.latent_dim = latent_dim
        # z_ref: reference point (center of radial transform)
        self.z_ref = nn.Parameter(torch.zeros(latent_dim))
        # alpha > 0 for invertibility (enforced via softplus)
        self.alpha_raw = nn.Parameter(torch.tensor(1.0))
        # beta: strength of radial push (small init near identity)
        self.beta = nn.Parameter(torch.tensor(init_scale))

    def forward(self, z_0):
        """
        Args:
            z_0: [..., latent_dim]

        Returns:
            z: [..., latent_dim]
            log_det: [...]  (log |det dz/dz_0|)
        """

        alpha = torch.nn.functional.softplus(self.alpha_raw) + 1e-8
        diff = z_0 - self.z_ref
        r = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-8)
        h = 1.0 / (alpha + r)
        z = z_0 + self.beta * h * diff

        # log |det J| = (d-1)*log(1 + beta*h) + log(1 + beta*alpha*h^2)
        h_squeezed = h.squeeze(-1)
        term1 = (self.latent_dim - 1) * torch.log(
            torch.abs(1.0 + self.beta * h_squeezed) + 1e-8)
        term2 = torch.log(
            torch.abs(1.0 + self.beta * alpha * (h_squeezed ** 2)) + 1e-8)
        log_det = term1 + term2

        return z, log_det


class RadialFlowStack(nn.Module):
    """Stack of K Radial Flow layers for more expressive posterior."""

    def __init__(self, latent_dim, num_flows=2, flow_init_scale=0.001):
        super().__init__()
        self.flows = nn.ModuleList(
            [RadialFlow(latent_dim, init_scale=flow_init_scale)
             for _ in range(num_flows)]
        )

    def forward(self, z_0):
        z = z_0
        log_det_total = 0.0

        for flow in self.flows:
            z, log_det = flow(z)
            log_det_total = log_det_total + log_det

        return z, log_det_total


class ConditionalFusion(nn.Module):
    """
    Key idea:
    - Incorporate semantic information as a condition to guide the generation of item embeddings.
    - Use a VAE structure (Conditional VAE) to model the latent variable z.
    - Produce condition-aware embeddings at the item embedding stage.

    Pipeline:
    1. Conditional Encoder: item_id_emb + item_semantic_emb → z_mean, z_logvar
    2. Reparameterization: z = reparameterize(z_mean, z_logvar)
    3. Conditional Decoder: z + item_semantic_emb → enhanced_item_emb
    """

    def __init__(self, item_dim, semantic_dim, latent_dim,
                 num_flows=2, flow_init_scale=0.001, flow_type='planar'):
        super().__init__()
        self.latent_dim = latent_dim

        # Conditional Encoder: item_id_emb + item_semantic_emb → z_mean,
        # z_logvar
        self.encoder = nn.Sequential(
            nn.Linear(item_dim + semantic_dim, item_dim * 2),
            nn.LayerNorm(item_dim * 2),
            nn.GELU(),
            nn.Linear(item_dim * 2, latent_dim * 2)
        )

        # Posterior flow: transforms Gaussian q_0(z_0|x) to flexible q(z|x)
        self.num_flows = num_flows
        self.flow_type = flow_type

        if num_flows > 0:
            if flow_type == 'radial':
                self.flow = RadialFlowStack(
                    latent_dim, num_flows=num_flows, flow_init_scale=flow_init_scale)

            else:
                self.flow = PlanarFlowStack(
                    latent_dim, num_flows=num_flows, flow_init_scale=flow_init_scale)

        else:
            self.flow = None

        # Conditional Decoder: z + item_semantic_emb → enhanced_item_emb
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + semantic_dim, item_dim * 2),
            nn.LayerNorm(item_dim * 2),
            nn.GELU(),
            nn.Linear(item_dim * 2, item_dim)
        )

        self._init_weights()

    def _init_weights(self):
        modules = [self.encoder, self.decoder]
        if self.flow is not None:
            modules.append(self.flow)

        for module in modules:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=1.414)

                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0.0)
                    nn.init.constant_(m.weight, 1.0)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, item_id_emb, item_semantic_emb, training=True):
        """
        Forward propagation (CVAE conditional fusion).

        Args:
            item_id_emb: [B, L, item_dim] or [N, item_dim] - item embeddings from ID lookup (collaborative signal only)
            item_semantic_emb: [B, L, semantic_dim] or [N, semantic_dim] - item semantic embeddings (condition), from LLM
            training: bool - whether to use reparameterization (only during training)

        Returns:
            enhanced_item_emb: [B, L, item_dim] or [N, item_dim] - condition-generated enhanced item embeddings
            kl_loss: optional, KL divergence loss (only computed during training)
        """

        # Concatenate item_id_emb and item_semantic_emb as encoder input
        encoder_input = torch.cat([item_id_emb, item_semantic_emb], dim=-1)
        encoder_output = self.encoder(encoder_input)
        z_mean, z_logvar = torch.chunk(encoder_output, 2, dim=-1)

        if training:
            # Base posterior: z_0 ~ q_0(z_0|x) = N(z_mean, exp(z_logvar))
            z_0 = self.reparameterize(z_mean, z_logvar)
            if self.flow is not None:
                z, log_det = self.flow(z_0)
            else:
                z = z_0
                log_det = 0.0

        else:
            if self.flow is not None:
                z, _ = self.flow(z_mean)
            else:
                z = z_mean

        # Concatenate z and item_semantic_emb as decoder input
        decoder_input = torch.cat([z, item_semantic_emb], dim=-1)
        enhanced_item_emb = self.decoder(decoder_input)

        if training:
            if self.flow is not None:
                # KL = E[log q_0(z_0) - log_det - log p(z)] with posterior flow
                var = torch.exp(z_logvar)
                log_q0 = -0.5 * torch.sum(
                    z_logvar + (z_0 - z_mean) ** 2 / var, dim=-1)
                log_pz = -0.5 * torch.sum(z ** 2, dim=-1)
                kl_per_sample = log_q0 - log_det - log_pz
                kl_loss = kl_per_sample.mean()

            else:
                # Original: KL(q_0 || p) for Gaussian posterior
                kl_loss = -0.5 * torch.sum(
                    1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=-1)
                kl_loss = kl_loss.mean()

            return enhanced_item_emb, kl_loss

        else:
            return enhanced_item_emb, None


class TransformerRec(nn.Module):
    def __init__(self, config):
        super(TransformerRec, self).__init__()
        # Configuration
        self.attn_dropout_prob = config.attn_dropout_prob
        self.hidden_act = config.hidden_act
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range
        self.inner_size = config.inner_size
        # register_buffer: item_semantic_emb moves with model.to(device).
        self.register_buffer('item_semantic_emb', config.item_semantic_emb)
        self.beta = getattr(config, 'beta', 0.01)
        self.latent_dim = config.latent_dim
        self.layer_norm_eps = config.layer_norm_eps
        self.max_seq_len = config.max_seq_len
        self.num_heads = getattr(config, 'num_heads', 2)
        self.num_items = config.item_num
        self.num_layers = getattr(config, 'num_layers', 2)

        # Neural network modules
        self.cvae_fusion_module = ConditionalFusion(
            item_dim=self.hidden_size,
            semantic_dim=self.hidden_size,
            latent_dim=self.latent_dim,
            num_flows=getattr(config, 'num_flows', 2),
            flow_init_scale=getattr(config, 'flow_init_scale', 0.001),
            flow_type=getattr(config, 'flow_type', 'planar')
        )
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.item_embedding = nn.Embedding(
            self.num_items + 1, self.hidden_size, padding_idx=0)
        self.LayerNorm = nn.LayerNorm(
            self.hidden_size, eps=self.layer_norm_eps)
        self.loss_fct = nn.CrossEntropyLoss()
        self.position_embedding = nn.Embedding(
            self.max_seq_len, self.hidden_size)

        # Project item semantic embedding to hidden_size dimensions
        semantic_input_dim = self.item_semantic_emb.shape[1]
        self.semantic_projection = nn.Linear(
            semantic_input_dim, self.hidden_size)

        self.trm_encoder = TransformerEncoder(
            n_layers=self.num_layers,
            n_heads=self.num_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.apply(self._init_weights)

    def set_beta(self, beta):
        # Set KL loss coefficient (beta in beta-CVAE) for annealing
        self.beta = beta

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def forward(self, item_seq, return_kl_loss=False):
        """
        Args:
            item_seq: [B, L] - item sequence
            return_kl_loss: bool - whether to return KL loss (for loss calculation)

        Returns:
            output: [B, H] - sequence representation
            kl_loss: optional, KL divergence loss
        """

        position_ids = torch.arange(item_seq.size(
            1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_id_emb = self.item_embedding(item_seq)
        # Contiguous after indexing; non-contiguous views can break
        # nn.Linear/cuBLAS.
        item_semantic_emb = self.item_semantic_emb[item_seq]
        if not item_semantic_emb.is_contiguous():
            item_semantic_emb = item_semantic_emb.contiguous()
        item_semantic_emb = self.semantic_projection(item_semantic_emb)

        enhanced_item_emb, kl_loss = self.cvae_fusion_module(
            item_id_emb, item_semantic_emb, training=self.training)

        input_emb = enhanced_item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = output[:, -1, :]

        if return_kl_loss:
            return output, kl_loss
        else:
            return output

    def _get_all_enhanced_item_embeddings(self, device):
        """
        Get enhanced embeddings for all items using CVAE conditional fusion in inference mode.

        Args:
            device: torch.device - device to create tensors on

        Returns:
            enhanced_all_item_emb: [n_items, hidden_size] - enhanced item embeddings
        """

        all_item_ids = torch.arange(1, self.num_items + 1, device=device)
        all_item_id_emb = self.item_embedding(all_item_ids)
        # Same: contiguous after indexing before Linear.
        all_item_semantic_emb = self.item_semantic_emb[all_item_ids]
        if not all_item_semantic_emb.is_contiguous():
            all_item_semantic_emb = all_item_semantic_emb.contiguous()
        all_item_semantic_emb = self.semantic_projection(all_item_semantic_emb)

        # Expand dimensions to match batch processing (inference mode)
        all_item_id_emb_expanded = all_item_id_emb.unsqueeze(0)
        all_item_semantic_emb_expanded = all_item_semantic_emb.unsqueeze(0)

        enhanced_all_item_emb, _ = self.cvae_fusion_module(
            all_item_id_emb_expanded, all_item_semantic_emb_expanded, training=False)
        enhanced_all_item_emb = enhanced_all_item_emb.squeeze(0)

        return enhanced_all_item_emb

    def calculate_loss(self, item_seq, target_items):
        seq_output, kl_loss = self.forward(item_seq, return_kl_loss=True)
        # Get enhanced embeddings for all items (for computing logits)
        enhanced_all_item_emb = self._get_all_enhanced_item_embeddings(
            item_seq.device)
        # Both operands contiguous before large matmul; avoids cuBLAS edge
        # cases.
        seq_output = seq_output.contiguous()
        enhanced_all_item_emb = enhanced_all_item_emb.contiguous()
        logits = torch.matmul(
            seq_output, enhanced_all_item_emb.transpose(0, 1))
        target_items_0idx = target_items - 1
        rec_loss = self.loss_fct(logits, target_items_0idx)
        weighted_kl_loss = self.beta * kl_loss
        total_loss = rec_loss + weighted_kl_loss

        return total_loss, rec_loss, weighted_kl_loss, seq_output

    def full_sort_predict(self, item_seq):
        seq_output = self.forward(item_seq)
        # Get enhanced embeddings for all items (for computing scores)
        enhanced_all_item_emb = self._get_all_enhanced_item_embeddings(
            item_seq.device)
        # Both operands contiguous before large matmul.
        seq_output = seq_output.contiguous()
        enhanced_all_item_emb = enhanced_all_item_emb.contiguous()
        scores = torch.matmul(
            seq_output, enhanced_all_item_emb.transpose(0, 1))

        return scores

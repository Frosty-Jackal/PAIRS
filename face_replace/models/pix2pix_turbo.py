import math
import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from diffusers.utils import is_xformers_available
from peft import LoraConfig
from transformers import AutoTokenizer, CLIPTextModel

from face_replace.configs.train_config import ModelConfig
from face_replace.models.attn_processors import SharedAttnProcessor, register_attention_processor, register_attention_processor_kv_unet, \
    AttnProcessor
from face_replace.models.model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd
from face_replace.models.unet_2d_condition.unet import UNet2DConditionModel

import time


# ═══════════════════════════════════════════════════════════════════
# Set-based Identity Encoding (Idea 3)
# ═══════════════════════════════════════════════════════════════════

class SetIdentityEncoder(nn.Module):
    """PMA-based set identity encoder.

    Flattens reference ResBlock features across K reference images into an
    unordered set of tokens, compresses them via cross-attention with M
    learnable inducing points, then produces a per-channel delta [B, C, 1, 1]
    for ADD-injection into the generation UNet at the corresponding layer.

    Parameters
    ----------
    feat_dim : int
        Number of channels at this UNet scale (e.g. 1280, 640, 320).
    num_inducing : int
        Number of learnable inducing points M (default 32).
    num_heads : int
        Number of attention heads for the cross-attention (default 8).
    """

    def __init__(self, feat_dim: int, num_inducing: int = 32, num_heads: int = 8):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_inducing = num_inducing

        # Learnable inducing points S ∈ R^{M × C}
        self.inducing = nn.Parameter(torch.randn(num_inducing, feat_dim) * 0.02)

        # PMA: cross-attention — S queries, flattened tokens as keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # ── Learned aggregation over M inducing points ──────────────
        # mean() discards the specialisation that inducing points learn.
        # Instead: a lightweight gating network learns to weight each
        # inducing point's contribution before summing.
        self.aggregator = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.GELU(),
            nn.Linear(feat_dim // 4, 1),
        )

        # Zero-initialised 1×1 conv for stable training (cf. ControlNet)
        self.zero_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, ref_feats):
        """
        Parameters
        ----------
        ref_feats : list of K tensors
            Each tensor has shape [B, C, H, W] from one reference image.
            K = max_conditioning_images (e.g. 3).

        Returns
        -------
        delta : torch.Tensor
            [B, C, 1, 1] — broadcastable identity delta for ADD injection.
        """
        B = ref_feats[0].shape[0]

        # Step 1: flatten each ref feature map → [B, HW, C]
        tokens_list = [f.flatten(2).transpose(1, 2) for f in ref_feats]
        # Step 2: concatenate across reference images → [B, K×HW, C]
        tokens = torch.cat(tokens_list, dim=1)  # [B, K*HW, C]

        # Step 3: PMA — inducing points attend to the unordered token set
        S = self.inducing.unsqueeze(0).expand(B, -1, -1)  # [B, M, C]
        Z, _ = self.cross_attn(S, tokens, tokens)          # [B, M, C]

        # Step 4: learned weighted aggregation → [B, C]
        # Each inducing point learns to detect a different identity
        # attribute; the aggregator learns how important each one is.
        weights = self.aggregator(Z).sigmoid()  # [B, M, 1]
        z_id = (Z * weights).sum(dim=1)              # [B, C]

        # Step 5: zero-initialised 1×1 conv → [B, C, 1, 1]
        delta = self.zero_conv(z_id.unsqueeze(-1).unsqueeze(-1))  # [B, C, 1, 1]
        return delta


MODEL_NAME = 'stabilityai/sd-turbo'


class Pix2Pix_Turbo(torch.nn.Module):

    def __init__(self,
                 pretrained_name: str = None,
                 pretrained_path: str = None,
                 lora_rank_unet: int = 8,
                 lora_rank_vae: int = 4,
                 condition_on_face_embeds: bool = False,
                 concat_mask_and_landmarks: bool = False,
                 save_self_attentions: bool = False,
                 train_reference_networks: bool = False,
                 cfg: ModelConfig = None):
        super().__init__()

        self.cfg = cfg

        # ── Set-encoding state ──────────────────────────────────────
        self._ref_up_block_features: list = []     # captured from Ref UNet
        self._set_deltas: list = []                # computed deltas for Gen UNet
        self._ref_hook_handles: list = []          # forward-hook handles on Ref UNet
        self._gen_hook_handles: list = []          # forward-hook handles on Gen UNet
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
        self.sched = make_1step_sched(model_name=MODEL_NAME)

        self.train_vae = self.cfg.train_vae
        self.train_only_vae_encoder = self.cfg.train_only_vae_encoder

        # vae = AutoencoderTiny.from_pretrained("madebyollin/taesd")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)

        # add the skip connection convs
        if self.cfg.use_shortcuts:
            vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
            vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
            vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
            vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
            vae.decoder.ignore_skip = False
        else:
            vae.decoder.ignore_skip = True

        unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")

        original_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

        original_unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")

        b1 = 1.4
        b2 = 1.6
        s1 = 0.9
        s2 = 0.2

        unet.enable_freeu(s1, s2, b1, b2)
        original_unet.enable_freeu(s1, s2, b1, b2)

        unet.to("cuda")
        vae.to("cuda")

        original_unet.to("cuda")
        original_vae.to("cuda")

        self.unet, self.vae, self.original_unet, self.original_vae = unet, vae, original_unet, original_vae

        self.train_reference_networks = train_reference_networks

        # We can use xformers here
        if is_xformers_available():
            self.original_unet.enable_xformers_memory_efficient_attention()

        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([999], device="cuda").long()

        self._init_models(pretrained_path=pretrained_path,
                          pretrained_name=pretrained_name,
                          lora_rank_vae=lora_rank_vae,
                          lora_rank_unet=lora_rank_unet)

        self.condition_on_face_embeds = condition_on_face_embeds

        self.text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder").cuda()
        self.text_encoder.requires_grad_(False)

        register_attention_processor_kv_unet(self.original_unet)
        register_attention_processor(self.unet, cfg=cfg, save_self_attentions=save_self_attentions)

        # ── Set-based Identity Encoding ─────────────────────────────
        self._init_set_encoders(cfg)

        prompt = "A high-quality photo of a person; professional, 8k"
        caption_tokens = self.tokenizer(prompt,
                                        max_length=self.tokenizer.model_max_length,
                                        padding="max_length",
                                        truncation=True,
                                        return_tensors="pt").input_ids.cuda()
        self.caption_enc = self.text_encoder(caption_tokens)[0]
        self.noise_timesteps = [249, 499, 749]

    def _init_models(self,
                     pretrained_path: str = None,
                     pretrained_name: str = None,
                     lora_rank_vae: int = 4,
                     lora_rank_unet: int = 8):
        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"],
                                          init_lora_weights="gaussian",
                                          target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"],
                                         init_lora_weights="gaussian",
                                         target_modules=sd["vae_lora_target_modules"])
            self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = self.vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            self.vae.load_state_dict(_sd_vae)
            self.unet.add_adapter(unet_lora_config)
            _sd_unet = self.unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            self.unet.load_state_dict(_sd_unet)
            # Load the weights for the original unet and vae
            _sd_original_unet = self.original_unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_original_unet[k] = sd["state_dict_unet"][k]
            self.unet.load_state_dict(_sd_original_unet)
            _sd_original_vae = self.original_vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_original_vae[k] = sd["state_dict_vae"][k]
            self.original_vae.load_state_dict(_sd_original_vae)

        elif pretrained_name is None and pretrained_path is None:
            if self.cfg.use_shortcuts:
                print("Initializing model with random weights")
                torch.nn.init.constant_(self.vae.decoder.skip_conv_1.weight, 1e-5)
                torch.nn.init.constant_(self.vae.decoder.skip_conv_2.weight, 1e-5)
                torch.nn.init.constant_(self.vae.decoder.skip_conv_3.weight, 1e-5)
                torch.nn.init.constant_(self.vae.decoder.skip_conv_4.weight, 1e-5)
            
            if self.train_vae:
                target_modules_vae = [
                    "conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                    "to_k", "to_q", "to_v", "to_out.0",
                ]
                if self.cfg.use_shortcuts:
                    target_modules_vae.extend(["skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4"])

                vae_lora_config = LoraConfig(r=lora_rank_vae,
                                            lora_alpha=lora_rank_vae // 2,
                                            init_lora_weights="gaussian",
                                            target_modules=target_modules_vae)
                self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
                self.target_modules_vae = target_modules_vae
                if self.train_reference_networks:
                    original_vae_lora_config = LoraConfig(r=16,
                                                          lora_alpha=8,
                                                          init_lora_weights="gaussian",
                                                          target_modules=target_modules_vae)
                    self.original_vae.add_adapter(original_vae_lora_config, adapter_name="vae_skip")

            target_modules_unet = [
                "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
            ]
            unet_lora_config = LoraConfig(r=lora_rank_unet,
                                          lora_alpha=lora_rank_unet // 2,
                                          init_lora_weights="gaussian",
                                          target_modules=target_modules_unet)
            self.unet.add_adapter(unet_lora_config)
            self.lora_rank_unet = lora_rank_unet
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_unet = target_modules_unet
            if self.train_reference_networks:
                original_unet_lora_config = LoraConfig(r=16,
                                                       lora_alpha=8,
                                                       init_lora_weights="gaussian",
                                                       target_modules=target_modules_unet)
                self.original_unet.add_adapter(original_unet_lora_config)

    # ═══════════════════════════════════════════════════════════════
    # Set-based Identity Encoding — setup, hooks, helpers
    # ═══════════════════════════════════════════════════════════════

    def _init_set_encoders(self, cfg: ModelConfig):
        """Create SetIdentityEncoder modules and register forward hooks.

        Hooks on the **Reference UNet** up_blocks capture intermediate
        features during ``get_conditioning_keys_values``.

        Hooks on the **Generation UNet** up_blocks inject the computed
        deltas during the main forward pass.

        The up_block indices and per-scale channel counts are read from
        the actual UNet config, so they are always correct for whichever
        SD variant is loaded.
        """
        if not getattr(cfg, 'use_set_encoding', False):
            self.set_encoders = nn.ModuleList()
            return

        # ── determine which up_blocks to hook ──────────────────────
        # By default hook all up blocks for multi-scale identity injection.
        default_indices = list(range(len(self.original_unet.up_blocks)))
        block_indices = getattr(cfg, 'set_encoding_up_block_indices', None)
        if block_indices is None:
            block_indices = default_indices
        elif len(block_indices) == 0:
            self.set_encoders = nn.ModuleList()
            return

        num_inducing = getattr(cfg, 'set_encoding_num_inducing', 32)

        # Read channel counts from the actual model
        reversed_channels = list(reversed(self.original_unet.config.block_out_channels))
        # reversed_channels[i] is the *output* channel of up_blocks[i]

        self._set_block_indices = block_indices

        encoders = []
        for idx in block_indices:
            feat_dim = reversed_channels[idx]
            enc = SetIdentityEncoder(
                feat_dim=feat_dim,
                num_inducing=num_inducing,
            )
            encoders.append(enc)

        self.set_encoders = nn.ModuleList(encoders)

        # ── register forward hooks ─────────────────────────────────
        self._register_ref_hooks(block_indices)
        self._register_gen_hooks(block_indices)

        print(f"[SetIdentityEncoder] Hooking up_blocks {block_indices} "
              f"with channel dims {[reversed_channels[i] for i in block_indices]}, "
              f"M={num_inducing}")

    def _register_ref_hooks(self, block_indices):
        """Capture up_block outputs from the Reference UNet."""
        self._remove_hooks(self._ref_hook_handles)
        self._ref_up_block_features = [None] * len(self.original_unet.up_blocks)

        def _make_capture_hook(blk_idx):
            def hook(module, input, output):
                self._ref_up_block_features[blk_idx] = output.detach()
            return hook

        for idx in block_indices:
            handle = self.original_unet.up_blocks[idx].register_forward_hook(
                _make_capture_hook(idx)
            )
            self._ref_hook_handles.append(handle)

    def _register_gen_hooks(self, block_indices):
        """Inject deltas into the Generation UNet up_block outputs."""
        self._remove_hooks(self._gen_hook_handles)
        self._set_deltas = [None] * len(self.unet.up_blocks)

        def _make_inject_hook(blk_idx):
            def hook(module, input, output):
                delta = self._set_deltas[blk_idx]
                if delta is not None:
                    # delta is [B, C, 1, 1]; broadcast-add to [B, C, H, W]
                    return output + delta.to(dtype=output.dtype)
                return output
            return hook

        for idx in block_indices:
            handle = self.unet.up_blocks[idx].register_forward_hook(
                _make_inject_hook(idx)
            )
            self._gen_hook_handles.append(handle)

    @staticmethod
    def _remove_hooks(handle_list):
        for h in handle_list:
            h.remove()
        handle_list.clear()

    def _compute_set_deltas(self, batch_size: int):
        """Run PMA on captured reference features to get per-scale deltas.

        Called once per forward pass, after ``get_conditioning_keys_values``
        has populated ``self._ref_up_block_features``.

        Parameters
        ----------
        batch_size : int
            Original batch size B (before K-reference flattening).
        """
        if not self.set_encoders:
            self._set_deltas = [None] * len(self.unet.up_blocks)
            return

        for enc_idx, blk_idx in enumerate(self._set_block_indices):
            raw = self._ref_up_block_features[blk_idx]
            if raw is None:
                self._set_deltas[blk_idx] = None
                continue

            # raw shape: [B*K, C, H, W]  →  reshape to per-sample list
            C = raw.shape[1]
            K = raw.shape[0] // batch_size  # number of ref images per sample

            ref_feats = []
            for k in range(K):
                ref_feats.append(raw[k::K].contiguous())  # [B, C, H, W]

            delta = self.set_encoders[enc_idx](ref_feats)  # [B, C, 1, 1]
            self._set_deltas[blk_idx] = delta

    def _clear_ref_features(self):
        self._ref_up_block_features = [None] * len(self.original_unet.up_blocks)

    # ═══════════════════════════════════════════════════════════════
    # End Set-encoding helpers
    # ═══════════════════════════════════════════════════════════════

    def set_eval(self):
        self.unet.eval()
        self.original_unet.eval()
        self.vae.eval()
        self.original_vae.eval()
        self.unet.requires_grad_(False)
        self.original_unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.original_vae.requires_grad_(False)
        if self.set_encoders:
            self.set_encoders.eval()

    def set_train(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)

        if self.train_vae:
            self.vae.train()
            for n, _p in self.vae.named_parameters():
                if "lora" in n:
                    _p.requires_grad = True
        else:
            self.vae.eval()
            for n, _p in self.vae.named_parameters():
                _p.requires_grad = False

        # Set the new cross-attention layers to trainable
        if self.condition_on_face_embeds:
            for n, _p in self.unet.named_parameters():
                if "_face_embed" in n or "face_projection" in n:
                    _p.requires_grad = True
        
        # Freeze the weights of the original UNet and VAE
        if self.train_reference_networks:
            self.original_unet.train()
            for n, _p in self.original_unet.named_parameters():
                if "lora" in n:
                    _p.requires_grad = True
            self.original_unet.conv_in.requires_grad_(True)
            self.original_vae.train()
            for n, _p in self.original_vae.named_parameters():
                if "lora" in n:
                    _p.requires_grad = True
        else:
            self.original_unet.eval()
            for n, _p in self.original_unet.named_parameters():
                _p.requires_grad = False
            self.original_vae.eval()
            for n, _p in self.original_vae.named_parameters():
                _p.requires_grad = False

        # ── Set-encoding parameters always trainable ──────────────
        if self.set_encoders:
            self.set_encoders.train()
            for enc in self.set_encoders:
                for _p in enc.parameters():
                    _p.requires_grad = True

    def get_conditioning_keys_values(self, conditioning_images, valid_indices):
        # Extract the keys and values from the conditioning images and use this to inject into the unet
        cond = conditioning_images.reshape(-1, 3, 512, 512)
        encoded_condition = self.original_vae.encode(cond).latent_dist.sample() * self.vae.config.scaling_factor

        t = torch.tensor([1], device="cuda")
        noise = torch.randn_like(encoded_condition)
        timesteps = t.long().repeat(encoded_condition.shape[0])
        noisy_encoded_condition = self.sched.add_noise(encoded_condition, noise, timesteps)
        model_input = self.sched.scale_model_input(noisy_encoded_condition, timesteps)

        extended_caption_enc = self.caption_enc.repeat(model_input.shape[0], 1, 1)

        # ── Clear stored ref features so hooks capture fresh ones ──
        self._clear_ref_features()

        model_pred_condition = self.original_unet(model_input,
                                                  t,
                                                  encoder_hidden_states=extended_caption_enc).sample

        # Get all the keys and values from the forward pass
        self_attn_processors = [p for p in self.original_unet.attn_processors.values() if type(p) in [AttnProcessor]]
        keys = [p.keys for p in self_attn_processors]
        values = [p.values for p in self_attn_processors]

        # Split then back to be back to the batch dimension
        keys_ = [k.reshape(-1, conditioning_images[0].shape[0], k.shape[1], k.shape[2]) for k in keys]
        values_ = [v.reshape(-1, conditioning_images[0].shape[0], v.shape[1], v.shape[2]) for v in values]

        # Using the valid_indices, we can zero out the invalid keys and values so we don't use them
        for k, v in zip(keys_, values_):
            for sample_idx in range(k.shape[0]):
                idx = valid_indices[sample_idx]  # zero out the entries greater than valid_idx
                k[sample_idx, idx:] = 0
                v[sample_idx, idx:] = 0

        for p in self_attn_processors: p.reset()

        x_denoised = self.sched.step(model_pred_condition, t, noisy_encoded_condition, return_dict=True).pred_original_sample
        output_image_conditions = (self.original_vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        return keys_, values_, output_image_conditions

    def forward(self, c_t,
                face_embeds=None,
                conditioning_images: torch.Tensor = None,
                valid_indices: torch.Tensor = None,
                mask: torch.Tensor = None,
                return_self_attention_maps: bool = False):

        # print("------------")
        # start_time = time.time()

        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor

        # print(f"VAE Encoding: {time.time() - start_time}")
        # start_time = time.time()

        # Extract the keys and values from the conditioning images and use this to inject into the unet
        if conditioning_images is not None and self.cfg.use_shared_attention:
            keys_, values_, output_image_conditions = self.get_conditioning_keys_values(conditioning_images, valid_indices)
        else:
            keys_, values_ = None, None
            output_image_conditions = None

        # print(f"Get Keys and Values: {time.time() - start_time}")
        # start_time = time.time()

        step = np.random.choice(self.noise_timesteps, 1)[0]
        t = torch.tensor([step], device="cuda")
        noise = torch.randn_like(encoded_control)
        timesteps = t.long().repeat(encoded_control.shape[0])
        noisy_encoded_condition = self.sched.add_noise(encoded_control, noise, timesteps)
        model_input = self.sched.scale_model_input(noisy_encoded_condition, timesteps)

        # print(f"Preprocessing: {time.time() - start_time}")
        # start_time = time.time()

        # ── Compute set-encoding deltas before Gen UNet forward ──
        if conditioning_images is not None and self.set_encoders:
            B = c_t.shape[0]
            self._compute_set_deltas(B)

        if self.condition_on_face_embeds and face_embeds is not None:
            model_pred = self.unet(model_input,
                                    t,
                                    encoder_hidden_states=face_embeds,
                                    cross_attention_kwargs={'ref_keys': keys_, 'ref_values': values_}).sample
        else:
            extended_caption_enc = self.caption_enc.repeat(model_input.shape[0], 1, 1)
            model_pred = self.unet(model_input,
                                    t,
                                    encoder_hidden_states=extended_caption_enc,
                                    cross_attention_kwargs={'ref_keys': keys_, 'ref_values': values_}).sample
        
        # print(f"UNet: {time.time() - start_time}")
        # start_time = time.time()

        x_denoised = self.sched.step(model_pred, t, noisy_encoded_condition, return_dict=True).pred_original_sample
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        # print(f"Post-processing + VAE Decode: {time.time() - start_time}")
        # start_time = time.time()

        if return_self_attention_maps:
            shared_attn_maps = []
            selected_indices = []
            for p in self.unet.attn_processors.values():
                if type(p) == SharedAttnProcessor and p.self_attn_idx is not None and p.attention_probs is not None:
                    shared_attn_maps.append(p.attention_probs)
                    selected_indices.append(p.self_attn_idx)
            ref_k_list = [keys_[idx] for idx in selected_indices] if selected_indices else []
            return output_image, output_image_conditions, shared_attn_maps, ref_k_list, selected_indices
        else:
            return output_image, output_image_conditions, None, None, None

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae if self.train_vae else None
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        # ── Set-encoding weights ──────────────────────────────────
        if self.set_encoders:
            sd["state_dict_set_encoders"] = {
                f"set_encoder_{i}": enc.state_dict()
                for i, enc in enumerate(self.set_encoders)
            }
            sd["set_block_indices"] = self._set_block_indices
        torch.save(sd, outf)

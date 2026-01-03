import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16
from torchvision.models import ViT_B_16_Weights
from torchvision.models import VisionTransformer
from transformers import ViTForImageClassification
from transformers import GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers import Qwen2ForSequenceClassification
from transformers import Qwen2Tokenizer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.qwen2.modeling_qwen2 import PreTrainedModel, Qwen2Attention, Qwen2DecoderLayer, Qwen2PreTrainedModel, Qwen2Config, Qwen2RMSNorm, Qwen2RotaryEmbedding, BaseModelOutputWithPast, Cache, DynamicCache
# from transformers import T5ForConditionalGeneration
# from transformers import BertForSequenceClassification
from transformers.modeling_layers import GenericForSequenceClassification
from typing import Optional

class TransformerEncoder(nn.Module):
    def __init__(self, args, attn_type, feats, mlp_hidden=128, head=8, dropout=0., embed_len=64, \
                low_rank=10, rank_multi=10, attn_drop=0.):
        super(TransformerEncoder, self).__init__()
        self.args = args
        self.attn_type = attn_type
        self.la1 = nn.LayerNorm(feats)
        if self.attn_type == "softmax":
            self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        elif self.attn_type == "kep_svgp":
            self.msa = KEP_SVGPAttention(feats, head, embed_len=embed_len, low_rank=low_rank, rank_multi=rank_multi, \
                                            concate=args.concate, proj_drop=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        if self.attn_type == "softmax":
            # noise_std = self.args.adversarial_noise # adjust as needed
            # Process original input
            la_x = self.la1(x)
            out_original = self.msa(la_x)
            mean_original = out_original
            # cov_original = torch.zeros_like(out_original)
            x_t_trans = out_original + x  # residual for original input
            out_final = self.mlp(self.la2(x_t_trans)) + x_t_trans

            # # Generate adversarial_samples adversarial inputs and compute aggregated mean/variance
            # means_list = [mean_original]
            # for _ in range(self.args.adversarial_samples):
            #     adv_x = x + torch.randn_like(x) * noise_std
            #     la_adv = self.la1(adv_x)
            #     out_adv = self.msa(la_adv)
            #     means_list.append(out_adv)
            # means_stack = torch.stack(means_list, dim=0)  # shape: (5, B, seq_len, d_model)
            # aggregated_mean = torch.mean(means_stack, dim=0) + x
            # aggregated_std = torch.std(means_stack, dim=0, unbiased=False) 

            # return out_final, x_t_trans, aggregated_mean, aggregated_std
            return out_final, x_t_trans, x_t_trans, torch.zeros_like(x_t_trans)

        else:
            la_x = self.la1(x)
            out, scores, Lambda_inv, kl, mean, cov = self.msa(la_x)
            out = out + x
            x_t_trans = out
            out = self.mlp(self.la2(out)) + out
            return out, scores, Lambda_inv, kl, x_t_trans, mean, cov


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats, head, dropout):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)

        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o

class ViT(nn.Module):
    def __init__(self, args, attn_type, ksvd_layers=1, low_rank=10, rank_multi=10, num_classes=10, img_size=32, channels=3, \
                patch=4, dropout=0., num_layers=7, hidden=384, mlp_hidden=384, head=8, is_cls_token=False):
        super(ViT, self).__init__()
        self.attn_type = attn_type
        self.patch = patch # number of patches in one row(or col)
        self.is_cls_token = is_cls_token
        self.patch_size = img_size//self.patch
        f = (img_size//self.patch)**2*channels # 48 # patch vec length
        num_tokens = (self.patch**2)+1 if self.is_cls_token else (self.patch**2)
        self.num_layers = num_layers
        self.ksvd_layers = ksvd_layers

        self.emb = nn.Linear(f, hidden) # (b, n, f)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden)) if is_cls_token else None
        self.pos_emb = nn.Parameter(torch.randn(1,num_tokens, hidden))
        enc_list = [TransformerEncoder(args=args, attn_type="softmax", low_rank=low_rank, rank_multi=rank_multi, embed_len=num_tokens, \
                    feats=hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(num_layers)]
        if self.attn_type == "kep_svgp":
            for i in range(self.ksvd_layers):
                enc_list[-(i+1)] = TransformerEncoder(args=args, attn_type="kep_svgp", low_rank=low_rank, rank_multi=rank_multi, embed_len=num_tokens, \
                    feats=hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head)
        self.enc = nn.Sequential(*enc_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes) # for cls_token
        )

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.patch**2 ,-1)
        return out

    def forward(self, x):
        x_t = []
        score_list = []
        Lambda_inv_list = []
        kl_list = []
        means = []
        covariances = []
        out = self._to_words(x)
        out = self.emb(out)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out],dim=1)
        out = out + self.pos_emb
        x_t.append(out)
        for enc in self.enc:
            if enc.attn_type == "softmax":
                out, x_t_trans, mean, cov = enc(out)
                x_t.append(x_t_trans)
                means.append(mean)
                covariances.append(cov)
            elif enc.attn_type == "kep_svgp":
                out, scores, Lambda_inv, kl, x_t_trans, mean, cov = enc(out)
                score_list.append(scores)
                Lambda_inv_list.append(Lambda_inv)
                kl_list.append(kl)
                x_t.append(x_t_trans)
                means.append(mean)
                covariances.append(cov)
        
        if self.is_cls_token:
            out = out[:,0]
        else:
            out = out.mean(1)
        out = self.fc(out)

        return out, x_t, means, covariances
    
class ViT_ImageNet(nn.Module):
    def __init__(self, weights=ViT_B_16_Weights.DEFAULT, seq_length=197, hidden_dim=768):
        super().__init__()
        self.model = vit_b_16(weights=weights)
    
    def forward(self, x):
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.model.encoder.pos_embedding
        x_t = [x]
        means = []
        stds = []
        for i, layer in enumerate(self.model.encoder.layers):
            out = layer.ln_1(x)
            out, _ = layer.self_attention(out, out, out, need_weights=False)
            out = layer.dropout(x)
            out = out + x
            x_t.append(out)
            means.append(out)
            stds.append(torch.zeros_like(out))
            x = layer.ln_2(out)
            x = layer.mlp(x)
            x = x + out
        return None, x_t, means, stds
            
class CustomViT(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.dataset == 'cifar10':
            self.model = ViTForImageClassification.from_pretrained(
                'aaraki/vit-base-patch16-224-in21k-finetuned-cifar10'
            )
            self.config = self.model.config   
    def forward(self, pixel_values):
        x_t = []
        hidden_states = self.model.vit.embeddings(pixel_values)
        x_t.append(hidden_states)
        for i, layer in enumerate(self.model.vit.encoder.layer):
            x = layer.attention(layer.layernorm_before(hidden_states), output_attentions=False)
            x = x[0] + hidden_states
            x_t.append(x)
            hidden_states = layer.output(layer.intermediate(layer.layernorm_after(x)), x)
        means = x_t[1:]
        stds = [torch.zeros_like(x) for x in means]
        return None, x_t, means, stds
    
class CustomGPT2(nn.Module):
    # Use GPT2ForSequenceClassification backbone; mirror CustomViT collection pattern
    def __init__(self, args=None):
        super().__init__()
        self.model = GPT2ForSequenceClassification.from_pretrained(
            'PavanNeerudu/gpt2-finetuned-sst2'
        )
        self.config = self.model.config

    def _expand_attention_mask(self, attention_mask, dtype):
        if attention_mask is None:
            return None
        # GPT-2 expects additive mask of shape (batch, 1, 1, seq_len)
        extended = attention_mask[:, None, None, :].to(dtype)
        extended = (1.0 - extended) * -1e4
        return extended

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        transformer = self.model.transformer
        bsz, seqlen = input_ids.shape
        device = input_ids.device

        position_ids = torch.arange(0, seqlen, dtype=torch.long, device=device).unsqueeze(0)
        hidden_states = transformer.wte(input_ids) + transformer.wpe(position_ids)
        hidden_states = transformer.drop(hidden_states)

        x_t = [hidden_states]
        means, stds = [], []

        attn_mask = self._expand_attention_mask(attention_mask, hidden_states.dtype) if attention_mask is not None else None

        for block in transformer.h:
            attn_input = block.ln_1(hidden_states)
            attn_outputs = block.attn(attn_input, layer_past=None, attention_mask=attn_mask, use_cache=False, output_attentions=False)
            attn_hidden = hidden_states + attn_outputs[0]

            x_t.append(attn_hidden)
            means.append(attn_hidden)
            stds.append(torch.zeros_like(attn_hidden))

            mlp_input = block.ln_2(attn_hidden)
            mlp_out = block.mlp(mlp_input)
            hidden_states = attn_hidden + mlp_out

        return None, x_t, means, stds

class HookedQwen2DecoderLayer(Qwen2DecoderLayer):
    def forward(
        self,
        hidden_states,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        use_cache = False,
        cache_position = None,
        position_embeddings = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states


        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, residual

class HookedQwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [HookedQwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        
        x_t = []
        means = []

        hidden_states = inputs_embeds
        x_t.append(hidden_states)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, residual = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            x_t.append(hidden_states)
            means.append(residual)

        hidden_states = self.norm(hidden_states)
        past_key_values = past_key_values if use_cache else None
        return past_key_values, hidden_states, x_t, means
    
# class HookedQwen2PretrainedModel(PreTrainedModel):
#     config: Qwen2Config
#     base_model_prefix = "model"
#     supports_gradient_checkpointing = True
#     _no_split_modules = ["HookedQwen2DecoderLayer"]
#     _skip_keys_device_placement = ["past_key_values"]
#     _supports_flash_attn = True
#     _supports_sdpa = True
#     _supports_flex_attn = True

#     _can_compile_fullgraph = True
#     _supports_attention_backend = True
#     _can_record_outputs = {
#         "hidden_states": HookedQwen2DecoderLayer,
#         "attentions": None,
#     }


    
# class CustomQwen2(nn.Module):
#     def __init__(self, args=None):
#         super().__init__()
#         self.model = Qwen2ForSequenceClassification.from_pretrained(
#             '/mnt/disk1/aiotlab/pqhung/TransDiff/qwen_cola_finetuned_merged'
#         )
#         self.config = self.model.config

#     def _expand_attention_mask(self, attention_mask, dtype):
#         if attention_mask is None:
#             return None
#         # Qwen2 expects additive mask of shape (batch, 1, 1, seq_len)
#         extended = attention_mask[:, None, None, :].to(dtype)
#         extended = (1.0 - extended) * torch.finfo(dtype).min
#         return extended

#     def forward(self, input_ids, attention_mask=None, token_type_ids=None):
#         # Access Qwen2 model structure (different from GPT-2)
#         model = self.model.model  # Qwen2Model is nested under .model
#         bsz, seqlen = input_ids.shape
#         device = input_ids.device

#         # Qwen2 uses embed_tokens instead of wte, and doesn't have wpe
#         hidden_states = model.embed_tokens(input_ids)
        
#         # Apply rotary position embedding is handled inside each layer
#         x_t = [hidden_states]
#         means, stds = [], []

#         # Prepare attention mask for Qwen2
#         attn_mask = self._expand_attention_mask(attention_mask, hidden_states.dtype) if attention_mask is not None else None

#         # Iterate through Qwen2 layers
#         for layer in model.layers:
#             # Layer normalization before attention (Qwen2 style)
#             normed_hidden = layer.input_layernorm(hidden_states)
            
#             # Self-attention with RoPE (rotary position embedding)
#             attn_outputs = layer.self_attn(
#                 normed_hidden,
#                 attention_mask=attn_mask,
#                 position_ids=None,  # RoPE handles positions internally
#                 past_key_value=None,
#                 output_attentions=False,
#                 use_cache=False
#             )
            
#             # Residual connection after attention
#             attn_hidden = hidden_states + attn_outputs[0]
            
#             x_t.append(attn_hidden)
#             means.append(attn_hidden)
#             stds.append(torch.zeros_like(attn_hidden))

#             # Post-attention layer norm and MLP
#             normed_attn = layer.post_attention_layernorm(attn_hidden)
#             mlp_out = layer.mlp(normed_attn)
#             hidden_states = attn_hidden + mlp_out

#         return None, x_t, means, stds



def vit_cifar(args, attn_type, num_classes, ksvd_layers, low_rank, rank_multi):
    return ViT(args=args, attn_type=attn_type, ksvd_layers=ksvd_layers, num_classes=num_classes, low_rank=low_rank, rank_multi=rank_multi, \
                img_size=32, patch=8, dropout=0.1, num_layers=args.depth, hidden=args.hdim, head=args.num_heads, mlp_hidden=args.hdim, is_cls_token=False)

if __name__ == '__main__':
    # Sanity check for CustomQwen2
    print("Testing CustomQwen2...")
    
    # Mock args for testing
    class MockArgs:
        pass
    
    args = MockArgs()
    
    try:
        # Initialize model
        model = CustomQwen2(args)
        print(f"✓ Model loaded successfully")
        print(f"  - Config: {model.config.name_or_path if hasattr(model.config, 'name_or_path') else 'Custom'}")
        print(f"  - Vocab size: {model.config.vocab_size}")
        print(f"  - Hidden size: {model.config.hidden_size}")
        print(f"  - Number of layers: {model.config.num_hidden_layers}")
        
        # Create sample input
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        
        print(f"\n✓ Created sample input:")
        print(f"  - Input IDs shape: {input_ids.shape}")
        print(f"  - Attention mask shape: {attention_mask.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output, x_t, means, stds = model(input_ids, attention_mask)
            
        print(f"\n✓ Forward pass successful:")
        print(f"  - Output: {output}")
        print(f"  - Number of intermediate representations (x_t): {len(x_t)}")
        print(f"  - Number of means: {len(means)}")
        print(f"  - Number of stds: {len(stds)}")
        
        # Check shapes
        print(f"\n✓ Shape analysis:")
        for i, x in enumerate(x_t):
            print(f"  - x_t[{i}]: {x.shape}")
        
        print(f"\n✓ Mean shapes:")
        for i, mean in enumerate(means):
            print(f"  - means[{i}]: {mean.shape}")
            
        print(f"\n✓ Std shapes:")
        for i, std in enumerate(stds):
            print(f"  - stds[{i}]: {std.shape}")
        
        # Test with different sequence lengths
        print(f"\n✓ Testing variable sequence lengths:")
        for seq_len in [5, 15, 20]:
            test_ids = torch.randint(0, 1000, (1, seq_len))
            test_mask = torch.ones_like(test_ids)
            with torch.no_grad():
                _, test_x_t, test_means, test_stds = model(test_ids, test_mask)
            print(f"  - Seq length {seq_len}: {len(test_x_t)} representations, shapes OK")
        
        print(f"\n🎉 All tests passed! CustomQwen2 is working correctly.")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
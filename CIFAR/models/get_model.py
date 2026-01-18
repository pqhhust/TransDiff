import models.vit_cifar
import models.diffusion
import models.q_distribution
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models.vision_transformer import ViT_B_16_Weights
import torch.distributed as dist
from transformers import AutoModelForImageClassification, ViTForImageClassification
# from models.q_distribution import HookedQwen2Model
from transformers import Qwen2Config
from transformers import Qwen2ForSequenceClassification
from transformers import Qwen2Tokenizer
from transformers import Qwen2Model

def get_model(model_name, nb_cls, logger, args):
    if model_name == "vit_image_net":
        # net = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).cuda()
        net = AutoModelForImageClassification.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
    if model_name == "q_distribution":
        net = models.q_distribution.vit_cifar(args=args, attn_type=args.attn_type, num_classes=nb_cls, ksvd_layers=args.ksvd_layers, low_rank=args.low_rank, rank_multi=args.rank_multi).cuda()
    if model_name == "q_distribution_imagenet":
        # net = models.q_distribution.ViT_ImageNet().cuda()
        net = models.q_distribution.CustomViT(args).cuda()
    if model_name == "q_distribution_text":
        config = Qwen2Config.from_pretrained(args.pretrained_dir)
        tokenizer = Qwen2Tokenizer.from_pretrained(args.pretrained_dir)
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Ensure we have a valid pad token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        pretrained_model = Qwen2Model.from_pretrained(args.pretrained_dir)
        net = models.q_distribution.HookedQwen2Model(config).cuda()
        net.load_state_dict(pretrained_model.state_dict(), strict=False)
        # Set pad token id in model config BEFORE resizing
        net.config.pad_token_id = tokenizer.pad_token_id
        
        # Resize token embeddings if needed (this must be done AFTER setting pad_token_id)
        if len(tokenizer) != net.config.vocab_size:
            # logger.info(f"Resizing token embeddings from {net.config.vocab_size} to {len(tokenizer)}")
            net.resize_token_embeddings(len(tokenizer))
            net.config.vocab_size = len(tokenizer)
    if model_name == "vit_cifar":
        net = models.vit_cifar.vit_cifar(args=args, attn_type=args.attn_type, num_classes=nb_cls, ksvd_layers=args.ksvd_layers, low_rank=args.low_rank, rank_multi=args.rank_multi).cuda()
    if model_name == "diffusion":
        args, config = args
        if args.backbone == 'mlp':
            net = models.diffusion.Diffusion_MLP(args=args, d_model=args.hdim, hdim1=args.mlp_hdim1, hdim2=args.mlp_hdim2, hdim3=args.mlp_hdim3, hdim4=args.mlp_hdim4, dropout=args.mlp_dropout, clip=args.clip, ViT_depth=args.depth)
        if args.backbone == 'unet1d':
            net = models.diffusion.Diffusion_UNet1D()
        if args.backbone == 'transformer':
            net = models.diffusion.Diffusion_Transformer(d_model=args.hdim, depth=args.trans_depth, num_heads=args.trans_num_heads, mlp_ratio=args.trans_mlp_ratio, dropout=args.trans_dropout, ViT_depth=args.depth, nb_cls=args.nb_cls, CONFIG=config)
        if args.backbone == 'mlp_mixer':
            net = models.diffusion.Diffusion_MLPMixer()
        if args.backbone == 'lstm' or args.backbone == 'gru':
            net = models.diffusion.Diffusion_RNN(args=args, rnn_hidden=args.rnn_hidden, rnn_num_layers=args.rnn_num_layers, dropout=args.rnn_dropout, ViT_depth=args.depth, low_dim=args.rnn_low_dim)
    if model_name == "diffusion_text":
        args, config = args
        net = models.diffusion.Diffusion_Transformer_Text(
            d_model=args.hdim,
            depth=args.trans_depth,
            num_heads=args.trans_num_heads,
            mlp_ratio=args.trans_mlp_ratio,
            dropout=args.trans_dropout,
            ViT_depth=args.depth,
            nb_cls=args.nb_cls,
            CONFIG=config,
            from_layer=args.from_layer,
            to_layer=args.to_layer,
        )
        base_model = Qwen2ForSequenceClassification.from_pretrained(args.pretrained_dir)
        net.score.load_state_dict(base_model.score.state_dict())
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        msg = 'Using {} ...'.format(model_name)
        logger.info(msg)
    return net
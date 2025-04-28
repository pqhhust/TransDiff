import models.vit_cola
import models.q_distribution
import models.diffusion
import models.sgpa
# import models.svdkl

def get_model(model_name, vocab_size, logger, args):
    if model_name == "svdkl":
        feature_extractor = models.svdkl.vit_cola(args=args, vocab_size=vocab_size, attn_type=args.attn_type, ksvd_layers=args.ksvd_layers, low_rank=args.low_rank, rank_multi=args.rank_multi)
        net = models.svdkl.DKLModel(feature_extractor, num_dim=args.hdim)
    if model_name == "q_distribution":
        net = models.q_distribution.vit_cola(args=args, vocab_size=vocab_size, attn_type=args.attn_type, ksvd_layers=args.ksvd_layers, low_rank=args.low_rank, rank_multi=args.rank_multi)
    if model_name == "vit_cola":
        if args.attn_type == "sgpa":
            net = models.sgpa.Transformer(device='cuda', vocab_size=vocab_size, depth=args.depth, max_len=100, embdim=128, num_class=args.num_classes, hdim=args.hdim, num_heads=args.num_heads, sample_size=1, jitter=1e-7, drop_rate=0.1, keys_len=5, kernel_type='exponential', flag_sgp=True)
        else:
            net = models.vit_cola.vit_cola(args=args, vocab_size=vocab_size, attn_type=args.attn_type, ksvd_layers=args.ksvd_layers, low_rank=args.low_rank, rank_multi=args.rank_multi)
    if model_name == "diffusion":
        if args.backbone == 'transformer':
            net = models.diffusion.Diffusion_Transformer(args=args, vocab_size=vocab_size, d_model=args.hdim, depth=args.trans_depth, num_heads=args.trans_num_heads, mlp_ratio=args.trans_mlp_ratio, dropout=args.trans_dropout, ViT_depth=args.depth, nb_cls=args.num_classes)
        if args.backbone == 'mlp':
            net = models.diffusion.Diffusion_MLP(args=args, vocab_size=vocab_size, d_model=args.hdim, hdim1=args.mlp_hdim1, hdim2=args.mlp_hdim2, hdim3=args.mlp_hdim3, hdim4=args.mlp_hdim4, dropout=args.mlp_dropout, clip=args.clip, ViT_depth=args.depth, nb_cls=args.num_classes)
    msg = 'Using {} ...'.format(model_name)
    logger.info(msg)
    return net
import models.vit_cifar
# import models.diffusion
import models.q_distribution
import models.sgpa
import models.cgpt
import models.scgpt
import models.svdkl
def get_model(model_name, nb_cls, logger, args):
    if model_name == "svdkl":
        feature_extractor = models.svdkl.vit_cifar(args=args, attn_type=args.attn_type, num_classes=nb_cls, ksvd_layers=args.ksvd_layers, low_rank=args.low_rank, rank_multi=args.rank_multi).cuda()
        net = models.svdkl.DKLModel(feature_extractor, num_dim=args.hdim)
    if model_name == "q_distribution":
        net = models.q_distribution.vit_cifar(args=args, attn_type=args.attn_type, num_classes=nb_cls, ksvd_layers=args.ksvd_layers, low_rank=args.low_rank, rank_multi=args.rank_multi).cuda()
    if model_name == "vit_cifar" or model_name == 'temperature_scaling' or model_name == 'mc_dropout' or model_name == 'kflla':
        if args.attn_type == "sgpa":
            net = models.sgpa.ViT(device=f'cuda:{args.gpu}', depth=args.depth, patch_size=4, in_channels=3, max_len = 64, num_class=args.nb_cls, hdim=args.hdim, num_heads=args.num_heads, sample_size=1, jitter=1e-6, drop_rate=0.1, keys_len=16, kernel_type='ard', flag_sgp=True).cuda()
        elif args.attn_type == "cgpt":
            net = models.cgpt.ViT(device=f'cuda:{args.gpu}', depth=args.depth, patch_size=4, in_channels=3, max_len = 64, num_class=args.nb_cls, hdim=args.hdim, num_heads=args.num_heads, sample_size=1, jitter=1e-6, drop_rate=0.1, keys_len=16, kernel_type='std', flag_cgp=True).cuda()
        elif args.attn_type == "scgpt":
            net = models.scgpt.ViT(device=f'cuda:{args.gpu}', depth=args.depth, patch_size=4, in_channels=3, max_len = 64, num_class=args.nb_cls, hdim=args.hdim, num_heads=args.num_heads, sample_size=1, jitter=1e-6, noise=0.1, drop_rate=0.1, keys_len=16, kernel_type='std', flag_cgp=True).cuda()
        else:
            net = models.vit_cifar.vit_cifar(args=args, attn_type=args.attn_type, num_classes=nb_cls, ksvd_layers=args.ksvd_layers, low_rank=args.low_rank, rank_multi=args.rank_multi).cuda()
    if model_name == "diffusion":
        pass
        # if args.backbone == 'mlp':
        #     net = models.diffusion.Diffusion_MLP(args=args, d_model=args.hdim, hdim1=args.mlp_hdim1, hdim2=args.mlp_hdim2, hdim3=args.mlp_hdim3, hdim4=args.mlp_hdim4, dropout=args.mlp_dropout, clip=args.clip, ViT_depth=args.depth)
        # if args.backbone == 'unet1d':
        #     net = models.diffusion.Diffusion_UNet1D()
        # if args.backbone == 'transformer':
        #     net = models.diffusion.Diffusion_Transformer(d_model=args.hdim, depth=args.trans_depth, num_heads=args.trans_num_heads, mlp_ratio=args.trans_mlp_ratio, dropout=args.trans_dropout, ViT_depth=args.depth, nb_cls=args.nb_cls)
        # if args.backbone == 'mlp_mixer':
        #     net = models.diffusion.Diffusion_MLPMixer()
        # if args.backbone == 'lstm' or args.backbone == 'gru':
        #     net = models.diffusion.Diffusion_RNN(args=args, rnn_hidden=args.rnn_hidden, rnn_num_layers=args.rnn_num_layers, dropout=args.rnn_dropout, ViT_depth=args.depth, low_dim=args.rnn_low_dim)
    if model_name == 'vit_cifar_teacher':
        net = models.vit_cifar.vit_cifar_teacher(args=args, attn_type=args.attn_type, num_classes=nb_cls, ksvd_layers=args.ksvd_layers, low_rank=args.low_rank, rank_multi=args.rank_multi).cuda()
    msg = 'Using {} ...'.format(model_name)
    logger.info(msg)
    return net
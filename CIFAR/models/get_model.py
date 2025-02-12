import models.vit_cifar
import models.diffusion
import models.q_distribution

def get_model(model_name, nb_cls, logger, args):
    if model_name == "q_distribution":
        net = models.q_distribution.vit_cifar(args=args, attn_type=args.attn_type, num_classes=nb_cls, ksvd_layers=args.ksvd_layers, low_rank=args.low_rank, rank_multi=args.rank_multi).cuda()
    if model_name == "vit_cifar":
        net = models.vit_cifar.vit_cifar(args=args, attn_type=args.attn_type, num_classes=nb_cls, ksvd_layers=args.ksvd_layers, low_rank=args.low_rank, rank_multi=args.rank_multi).cuda()
    if model_name == "diffusion":
        if args.backbone == 'mlp':
            net = models.diffusion.Diffusion_MLP(args=args, d_model=args.hdim, hdim1=args.mlp_hdim1, hdim2=args.mlp_hdim2, hdim3=args.mlp_hdim3, hdim4=args.mlp_hdim4, dropout=args.mlp_dropout, clip=args.clip, ViT_depth=args.depth)
        if args.backbone == 'unet1d':
            net = models.diffusion.Diffusion_UNet1D()
        if args.backbone == 'transformer':
            net = models.diffusion.Diffusion_Transformer()
        if args.backbone == 'mlp_mixer':
            net = models.diffusion.Diffusion_MLPMixer()
        if args.backbone == 'lstm' or args.backbone == 'gru':
            net = models.diffusion.Diffusion_RNN(args=args, rnn_hidden=args.rnn_hidden, rnn_num_layers=args.rnn_num_layers, dropout=args.rnn_dropout, ViT_depth=args.depth, low_dim=args.rnn_low_dim)
    msg = 'Using {} ...'.format(model_name)
    logger.info(msg)
    return net
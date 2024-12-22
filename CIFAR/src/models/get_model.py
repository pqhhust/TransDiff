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
            net = models.diffusion.Diffusion_MLP()
        if args.backbone == 'unet1d':
            net = models.diffusion.Diffusion_UNet1D()
        if args.backbone == 'transformer':
            net = models.diffusion.Diffusion_Transformer()
    msg = 'Using {} ...'.format(model_name)
    logger.info(msg)
    return net
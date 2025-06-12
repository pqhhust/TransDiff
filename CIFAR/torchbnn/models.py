import numpy as np
import torch
import torchsde
from torch import nn
from torchbnn import diffeq_layers, utils
from torchbnn.DiT import DiT


# TODO:
class YNetWithSplit(nn.Module):
    def __init__(self, *blocks):
        # Or create the blocks in side this function given the input size and other hparams
        # Each block has a split module at the end, which does t1, t2 = t.chunk(dim=1, chunks=2).
        pass

    def forward(self, x):
        zs = []
        net = x
        for block in self.blocks:
            z1, z2 = block(net)
            zs.append(z2)
            net = z1
        return zs  # Or cat along non-batch dimension.


def make_y_net(input_size,
               blocks=(2, 2, 2),
               activation="softplus",
               verbose=False,
               explicit_params=True,
               hidden_width=384,
               aug_dim=0):
    _input_size = (input_size[0] + aug_dim,) + input_size[1:]
    layers = []
    for i, num_blocks in enumerate(blocks, 1):
        for j in range(1, num_blocks + 1):
            layers.extend(
                diffeq_layers.make_ode_k3_block_layers(
                    input_size=_input_size,
                    activation=activation,
                    last_activation=i < len(blocks) or j < num_blocks,
                    hidden_width=hidden_width
                )
            )
            if verbose:
                if i == 1:
                    print(f"y_net (augmented) input size: {_input_size}")
                layers.append(diffeq_layers.Print(name=f"group: {i}, block: {j}"))
        if i < len(blocks):
            layers.append(diffeq_layers.ConvDownsample(_input_size))
            _input_size = _input_size[0] * 4, _input_size[1] // 2, _input_size[2] // 2

    y_net = diffeq_layers.DiffEqSequential(*layers, explicit_params=explicit_params)
    # return augmented input size b/c y net should have same input / output
    return y_net, _input_size


def make_w_net(in_features, hidden_sizes=(1, 64, 1), activation="softplus", inhomogeneous=True):
    activation = utils.select_activation(activation)
    all_sizes = (in_features,) + tuple(hidden_sizes) + (in_features,)

    if inhomogeneous:
        layers = []
        for i, (in_size, out_size) in enumerate(zip(all_sizes[:-1], all_sizes[1:]), 1):
            layers.append(diffeq_layers.Linear(in_size, out_size))
            if i + 1 < len(all_sizes):
                layers.append(diffeq_layers.DiffEqWrapper(activation()))
            else:  # Last layer needs zero initialization.
                nn.init.zeros_(layers[-1].weight)
                nn.init.zeros_(layers[-1].bias)
        return diffeq_layers.DiffEqSequential(*layers, explicit_params=False)
    else:
        layers = []
        for i, (in_size, out_size) in enumerate(zip(all_sizes[:-1], all_sizes[1:]), 1):
            layers.append(nn.Linear(in_size, out_size))
            if i + 1 < len(all_sizes):
                layers.append(activation())
            else:  # Last layer needs zero initialization.
                nn.init.zeros_(layers[-1].weight)
                nn.init.zeros_(layers[-1].bias)
        return diffeq_layers.DiffEqWrapper(nn.Sequential(*layers))


class BaselineYNet(nn.Module):
    def __init__(self, input_size=(3, 32, 32), num_classes=10, activation="softplus", residual=False, hidden_width=384,
                 aug=0):
        super(BaselineYNet, self).__init__()
        y_net, output_size = make_y_net(
            input_size=input_size, explicit_params=False, activation=activation, hidden_width=hidden_width)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(np.prod(output_size)) + aug, num_classes)
        )
        self.y_net = y_net
        self.residual = residual

    def forward(self, y, *args, **kwargs):
        t = y.new_tensor(0.)
        outs = self.y_net(t, y).flatten(start_dim=1)
        if self.residual:
            outs += y.flatten(start_dim=1)
        return self.projection(outs), torch.tensor(0., device=y.device)


# TODO: add STL
class SDENet(torchsde.SDEStratonovich):
    def __init__(self,
                 input_size=(3, 32, 32),
                 blocks=(2, 2, 2),
                 weight_network_sizes=(1, 1024, 1),
                 num_classes=10,
                 activation="softplus",
                 verbose=False,
                 inhomogeneous=True,
                 sigma=0.1,
                 hidden_width=384,
                 aug_dim=0,
                 img_size=32,
                 patch=8,
                 is_cls_token=None):
        super(SDENet, self).__init__(noise_type="diagonal")
        self.input_size = input_size
        # self.aug_input_size = (aug_dim + input_size[0], *input_size[1:])
        # self.aug_zeros_size = (aug_dim, *input_size[1:])
        # self.register_buffer('aug_zeros', torch.zeros(size=(1, *self.aug_zeros_size)))

        ## Embedding layers
        self.patch = patch # number of patches in one row(or col)
        self.is_cls_token = is_cls_token
        self.patch_size = img_size//self.patch
        f = (img_size//self.patch)**2*input_size[0]
        num_tokens = (self.patch**2)+1 if self.is_cls_token else (self.patch**2)
        self.emb = nn.Linear(f, hidden_width) # (b, n, f)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_width)) if is_cls_token else None
        self.pos_emb = nn.Parameter(torch.randn(1,num_tokens, hidden_width))

        # Create network evolving state.
        self.y_net = diffeq_layers.make_ode_dit(depth=1, hidden_size=hidden_width, num_heads=12, mlp_ratio=1.0)
        self.output_size = (num_tokens, hidden_width)
        # self.y_net, self.output_size = make_y_net(
        #     input_size=input_size,
        #     blocks=blocks,
        #     activation=activation,
        #     verbose=verbose,
        #     hidden_width=hidden_width,
        #     aug_dim=aug_dim
        # )


        # Create network evolving weights.
        initial_params = self.y_net.make_initial_params()  # w0.
        flat_initial_params, unravel_params = utils.ravel_pytree(initial_params)
        self.flat_initial_params = nn.Parameter(flat_initial_params, requires_grad=True)
        self.params_size = flat_initial_params.numel()
        for i in range(10):
            print(initial_params[1][i].size())
        print(f"initial_params ({self.params_size}): {flat_initial_params.shape}")
        self.unravel_params = unravel_params
        self.w_net = make_w_net(
            in_features=self.params_size,
            hidden_sizes=weight_network_sizes,
            activation="tanh",
            inhomogeneous=inhomogeneous
        )

        # Final projection layer.
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_width),
            nn.Linear(hidden_width, num_classes)
        )

        self.register_buffer('ts', torch.tensor([0., 1.]))
        self.sigma = sigma
        self.nfe = 0

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.patch**2 ,-1)
        return out

    def f(self, t, y: torch.Tensor):
        input_y = y
        self.nfe += 1
        # print('y_numel:', y.numel())
        # print('y_shape:', y.size())
        y, w, _ = y.split(split_size=(y.numel() - self.params_size - 1, self.params_size, 1), dim=1) # params_size: 606408
        # print('y_reshape:', y.reshape(-1, *self.output_size).size())
        fy = self.y_net(t, y.reshape((-1, *self.output_size)), self.unravel_params(w.squeeze(0))).reshape(-1).unsqueeze(0) 
        nn = self.w_net(t, w)
        fw = nn - w  # hardcoded OU prior on weights w
        fl = (nn ** 2).sum(dim=1, keepdim=True) / (self.sigma ** 2)
        # print(fy.shape, fw.shape, fl.shape)
        # print(input_y.shape)
        assert input_y.shape == torch.cat((fy, fw, fl),dim=1).shape, f"Want: {input_y.shape} Got: {torch.cat((fy, fw, fl)).shape}. Check nblocks for dataset divisibility.\n"
        return torch.cat((fy, fw, fl),dim=1)

    def g(self, t, y):
        self.nfe += 1
        gy = torch.zeros(size=(y.numel() - self.params_size - 1,), device=y.device)
        gw = torch.full(size=(self.params_size,), fill_value=self.sigma, device=y.device)
        gl = torch.tensor([0.], device=y.device)
        # print('y_shape:', y.shape, 'gy_shape:', gy.shape, 'gw_shape:', gw.shape, 'gl_shape:', gl.shape)
        return torch.cat((gy, gw, gl)).unsqueeze(0)

    def make_initial_params(self):
        return self.y_net.make_initial_params()

    def forward(self, y, adjoint=False, dt=0.02, adaptive=False, adjoint_adaptive=False, method="midpoint", rtol=1e-4, atol=1e-3):
        # Note: This works correctly, as long as we are requesting the nfe after each gradient update.
        #  There are obviously cleaner ways to achieve this.
        self.nfe = 0    
        y = self._to_words(y)
        y = self.emb(y)
        if self.is_cls_token:
            y = torch.cat([self.cls_token.repeat(y.size(0), 1, 1), y], dim=1)
        y = y + self.pos_emb
        # print('y_shape_before_fed:', y.shape)
        sdeint = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint
        # if self.aug_zeros.numel() > 0:  # Add zero channels.
        #     aug_zeros = self.aug_zeros.expand(y.shape[0], *self.aug_zeros_size)
        #     y = torch.cat((y, aug_zeros), dim=1) # 235200
        aug_y = torch.cat((y.reshape(-1), self.flat_initial_params, torch.tensor([0.], device=y.device))) # 841609: (235200, 606408, 1)
        aug_y = aug_y[None]
        # print('aug_y_shape:', aug_y.shape)
        bm = torchsde.BrownianInterval(
            t0=self.ts[0], t1=self.ts[-1], size=aug_y.shape, dtype=aug_y.dtype, device=aug_y.device,
            cache_size=45 if adjoint else 30  # If not adjoint, don't really need to cache.
        )
        if adjoint_adaptive:
            _, aug_y1 = sdeint(self, aug_y, self.ts, bm=bm, method=method, dt=dt, adaptive=adaptive, adjoint_adaptive=adjoint_adaptive, rtol=rtol, atol=atol)
        else:
            _, aug_y1 = sdeint(self, aug_y, self.ts, bm=bm, method=method, dt=dt, adaptive=adaptive, rtol=rtol, atol=atol)
        # print('aug_y1_shape:', aug_y1.shape, 'y_numel:', y.numel())
        # print('y1_shape:', (aug_y1[:y.numel()]).shape)
        y1 = aug_y1[0, :y.numel()].reshape(y.size())
        logits = self.projection(y1.mean(1))
        logqp = .5 * aug_y1[0, -1]
        # return y, y1, logits, logqp
        return logits, logqp

    def zero_grad(self) -> None:
        for p in self.parameters(): p.grad = None


if __name__ == "__main__":
    batch_size = 2
    input_size = c, h, w = 3, 32, 32
    sde = SDENet(inhomogeneous=False, input_size=input_size, aug_dim=1)
    sde.ts = torch.tensor([0., 1e-9])  # t0 can't be equal to t1 due to torchsde internal checks, set t1 to be tiny.

    y0 = torch.randn(batch_size, c, h, w)
    y1 = sde(y0)
    torch.testing.assert_close(y0, y1)
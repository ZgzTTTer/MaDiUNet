import torch
from ptflops import get_model_complexity_info
from models.unet import VanillaUNet, ResUNet, MambaUNet, DenseUNet, SwinUNet

class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.dummy_t = torch.tensor([0], dtype=torch.long)

    def forward(self, x):
        device = next(self.model.parameters()).device
        t = self.dummy_t.to(device)
        return self.model(x.to(device), t)

def get_model_profile(model, input_shape):
    device = torch.device("cuda:0")
    model = WrappedModel(model.to(device)).eval()
    with torch.cuda.device(device):
        macs, params = get_model_complexity_info(
            model, input_shape, as_strings=False,
            print_per_layer_stat=False, verbose=False
        )
        flops = macs / 1e9
        params_m = params / 1e6
    return params_m, flops

def build_model(model_class, args, in_channels, out_channels):
    model = model_class(
        args.T, args.ch, args.ch_mult, args.attn,
        args.num_res_blocks, args.dropout,
        in_channels=in_channels, out_channels=out_channels
    )
    return model

class Args:
    T = 1000
    ch = 128
    ch_mult = [1, 2, 3, 4]
    attn = [2]
    num_res_blocks = 2
    dropout = 0.3

if __name__ == "__main__":
    args = Args()
    in_channels = 2
    out_channels = 1
    input_size = (in_channels, 256, 256)

    model_classes = {
        "ResUNet": ResUNet,
        "MambaUNet": MambaUNet,
        "VanillaUNet": VanillaUNet,
        "DenseUNet": DenseUNet,
        "SwinUNet": SwinUNet,
    }

    print(f"{'Model':<15}{'Params (M)':<15}{'FLOPs (G)':<15}")
    print("-" * 45)
    for name, cls in model_classes.items():
        model = build_model(cls, args, in_channels, out_channels)
        try:
            params, flops = get_model_profile(model, input_size)
            print(f"{name:<15}{params:<15.2f}{flops:<15.2f}")
        except Exception as e:
            print(f"{name:<15}Error: {str(e)}")

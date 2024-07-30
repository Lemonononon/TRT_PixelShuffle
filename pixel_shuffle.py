from torch2trt import tensorrt_converter,get_arg,add_missing_trt_tensors

@tensorrt_converter('torch.nn.PixelShuffle.forward')
def convert_PixelShuffle(ctx):
    input = ctx.method_args[1]
    module = ctx.method_args[0]
    scale_factor =  module.upscale_factor

    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]


    output = ctx.method_return

    batch_size, in_channels, in_height, in_width = input.shape

    print("in_channels:", batch_size, in_channels, in_height, in_width)

    assert scale_factor >= 1

    out_channels = in_channels // (scale_factor * scale_factor)
    out_height = in_height * scale_factor
    out_width = in_width * scale_factor

    layer_1 = ctx.network.add_shuffle(input_trt)
    layer_1.reshape_dims = (out_channels, scale_factor, scale_factor, in_height, in_width)

    layer_2 = ctx.network.add_shuffle(layer_1.get_output(0))
    layer_2.first_transpose = (0, 3, 1, 4, 2)
    layer_2.reshape_dims = (batch_size, out_channels, out_height, out_width)

    output._trt = layer_2.get_output(0)
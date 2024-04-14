code = """
#[version = "0.0.5"]
def @main (%pzx: Tensor[(3, 4), float16] /* ty=Tensor[(3, 4), float16] */, %pzw1: Tensor[(5, 4), float16] /* ty=Tensor[(5, 4), float16] */, %pzw2: Tensor[(5, 4), float16] /* ty=Tensor[(5, 4), float16] */, %pzb1: Tensor[(5), float16] /* ty=Tensor[(5), float16] */, %pzb2: Tensor[(5), float16] /* ty=Tensor[(5), float16] */, %pzscale1: Tensor[(1), float16] /* ty=Tensor[(1), float16] */, %pzscale2: Tensor[(1), float16] /* ty=Tensor[(1), float16] */) -> (Tensor[(1, 30), float16], Tensor[(1, 15), float16]) {
  free_var %pzx1: Tensor[(3, 4), float16] /* ty=Tensor[(3, 4), float16] */;
  free_var %pzw11: Tensor[(5, 4), float16] /* ty=Tensor[(5, 4), float16] */;
  %0 = nn.dense(%pzx1, %pzw11, units=None) /* ty=Tensor[(3, 5), float16] */;
  %1 = add(%0, %pzb1) /* ty=Tensor[(3, 5), float16] */;
  %2 = multiply(%1, %pzscale1) /* ty=Tensor[(3, 5), float16] */;
  free_var %pzw21: Tensor[(5, 4), float16] /* ty=Tensor[(5, 4), float16] */;
  %3 = nn.dense(%pzx1, %pzw21, units=None) /* ty=Tensor[(3, 5), float16] */;
  %4 = add(%3, %pzb2) /* ty=Tensor[(3, 5), float16] */;
  %5 = multiply(%4, %pzscale2) /* ty=Tensor[(3, 5), float16] */;
  %6 = reshape(%2, newshape=[1, 1, 15]) /* ty=Tensor[(1, 1, 15), float16] */;
  %7 = reshape(%5, newshape=[1, 1, 15]) /* ty=Tensor[(1, 1, 15), float16] */;
  %8 = add(%6, %7) /* ty=Tensor[(1, 1, 15), float16] */;
  %9 = max(%8, axis=[1]) /* ty=Tensor[(1, 15), float16] */;
  %10 = divide(%9, %9) /* ty=Tensor[(1, 15), float16] */;
  %11 = tan(%10) /* ty=Tensor[(1, 15), float16] */;
  %12 = expand_dims(%11, axis=1, num_newaxis=0) /* ty=Tensor[(1, 15), float16] */;
  %13 = multiply(%12, %11) /* ty=Tensor[(1, 15), float16] */;
  %14 = multiply(%13, 2f16 /* ty=float16 */) /* ty=Tensor[(1, 15), float16] */;
  %15 = (%14, %14) /* ty=(Tensor[(1, 15), float16], Tensor[(1, 15), float16]) span=from_string:9:22 */;
  %16 = concatenate(%15, axis=1) /* ty=Tensor[(1, 30), float16] */;
  %17 = nn.dense(%pzx, %pzw1, units=None) /* ty=Tensor[(3, 5), float16] */;
  %18 = add(%17, %pzb1) /* ty=Tensor[(3, 5), float16] */;
  %19 = multiply(%18, %pzscale1) /* ty=Tensor[(3, 5), float16] */;
  %20 = nn.dense(%pzx, %pzw2, units=None) /* ty=Tensor[(3, 5), float16] */;
  %21 = add(%20, %pzb2) /* ty=Tensor[(3, 5), float16] */;
  %22 = multiply(%21, %pzscale2) /* ty=Tensor[(3, 5), float16] */;
  %23 = reshape(%19, newshape=[1, 1, 15]) /* ty=Tensor[(1, 1, 15), float16] */;
  %24 = reshape(%22, newshape=[1, 1, 15]) /* ty=Tensor[(1, 1, 15), float16] */;
  %25 = add(%23, %24) /* ty=Tensor[(1, 1, 15), float16] */;
  %26 = max(%25, axis=[1]) /* ty=Tensor[(1, 15), float16] */;
  %27 = divide(%26, %26) /* ty=Tensor[(1, 15), float16] */;
  %28 = tan(%27) /* ty=Tensor[(1, 15), float16] */;
  %29 = expand_dims(%28, axis=1, num_newaxis=0) /* ty=Tensor[(1, 15), float16] */;
  %30 = multiply(%29, %28) /* ty=Tensor[(1, 15), float16] */;
  %31 = nn.leaky_relu(%30, alpha=0.0190813f) /* ty=Tensor[(1, 15), float16] */;
  %32 = ceil(%16) /* ty=Tensor[(1, 30), float16] */;
  %33 = round(%31) /* ty=Tensor[(1, 15), float16] */;
  (%32, %33) /* ty=(Tensor[(1, 30), float16], Tensor[(1, 15), float16]) span=from_string:3:5 */
}
"""
import tvm
from tvm import relay
print(relay.parse(code))

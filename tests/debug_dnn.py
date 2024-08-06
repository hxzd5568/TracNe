import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tvm
from tvm import relay
import os
from scipy import stats
from torchviz import make_dot
import sys

sys.path.append("..")
args = sys.argv
from src.viz import save_irmod_viz
from src.base_utils import Checkor
import netron
from torch import nn

caseid = args[1]
case_path = "./dnn/"
paras = "./dnn/out/resnet18/L1"
dump_path = case_path + "/out/" + caseid
print("dump_path", dump_path)

# load data
path_params = os.path.join(dump_path, "oinputs.npz")
if os.path.getsize(path_params) > 0:
    with np.load(path_params, allow_pickle=True) as f:
        loaded_params = dict(f.items())
    loaded_params = dict(reversed(list(loaded_params.items())))
else:
    print("zero size")
    exit()
baseline_model = torch.load(dump_path + "/model_scripted.pt").float().eval()


def get_runing_var():
    baseline_model.bn1.track_running_stats = False
    x = torch.randn(size=(1, 3, 224, 224))
    y = baseline_model(x.float())
    # def get_layers(model):
    #     layers = []
    #     for name, module in model.named_children():
    #         if isinstance(module, nn.Sequential):
    #             layers += get_layers(module)
    #         elif isinstance(module, nn.ModuleList):
    #             for m in module:
    #                 layers += get_layers(m)
    #         else:
    #             layers.append(module)
    #     return layers

    # layers = get_layers(baseline_model)
    # # for layer in layers:
    # #     print(layer.__class__.__name__)
    # for name, layer in baseline_model.named_modules():
    #         print(name, layer)
    print(baseline_model.bn1.running_mean)
    # baseline_model.bn1.track_running_stats = False
    # print(baseline_model.bn1.track_running_stats )


def compare_bnweight():
    # tvm
    # print('len compare', len(list(loaded_params.keys())) ,len(list(baseline_model.named_parameters())))
    # print('weight compare', loaded_params.keys() )
    # print('-'*50)
    # print(dict(baseline_model.named_parameters()).keys())
    # print(np.equal())
    tdict = dict(baseline_model.named_parameters())
    tvmiter = iter(loaded_params)
    torchiter = iter(tdict)

    length = len(tdict)
    deeps = 0
    for i in range(length):
        tvmk = next(tvmiter)
        while not ("runningmean" in tvmk or "runningvar" in tvmk):
            tvmk = next(tvmiter)
        deeps += 1
        tvmv = loaded_params[tvmk]
        # torchk = next(torchiter)
        # torchv = tdict[torchk]
        # print(torchk,tvmk)
        print(tvmk, tvmv)
        if deeps == 2:
            break
        # assert(np.equal(torchv.detach().numpy(),tvmv).all()==True)


# data = relay.load_param_dict(bytearray(open(paras+'/_tvmdbg_device_CPU_0/output_tensors.params', "rb").read()))
def compare_weight():
    # tvm
    # print('len compare', len(list(loaded_params.keys())) ,len(list(baseline_model.named_parameters())))
    # print('weight compare', loaded_params.keys() )
    # print('-'*50)
    # print(dict(baseline_model.named_parameters()).keys())
    # print(np.equal())
    tdict = dict(baseline_model.named_parameters())
    tvmiter = iter(loaded_params)
    torchiter = iter(tdict)

    length = len(tdict)
    for i in range(length):
        tvmk = next(tvmiter)
        while "runningmean" in tvmk or "runningvar" in tvmk:
            tvmk = next(tvmiter)
        tvmv = loaded_params[tvmk]
        torchk = next(torchiter)
        torchv = tdict[torchk]
        print(torchk, tvmk)
        assert np.equal(torchv.detach().numpy(), tvmv).all() == True


def draw_netron():
    x = torch.randn(size=(1, 3, 224, 224))
    y = baseline_model(x.float())
    onnx_path = f"{dump_path}/onnx_model_name.onnx"
    torch.onnx.export(baseline_model, x, onnx_path)
    netron.start(onnx_path)


def draw_torchviz():
    x = torch.randn(size=(1, 3, 224, 224))
    y = baseline_model(x.float())
    src = make_dot(y.mean(), params=dict(baseline_model.named_parameters()))
    src.render(f"{dump_path}/torchviz").replace("\\", "/")
    print("torchviz done")

    checkor = Checkor(path=case_path, case_id=caseid)
    save_irmod_viz(checkor.mod, dump_path + "/relayviz")
    print("tvmviz done")


# draw weight pictures
def draw_weight():
    if not os.path.exists(dump_path + "/" + "tvmp"):
        os.mkdir(dump_path + "/" + "tvmp")
    if not os.path.exists(dump_path + "/" + "torchp"):
        os.mkdir(dump_path + "/" + "torchp")
    for k, v in loaded_params.items():
        vn = v.flatten()
        fig = plt.figure()
        sns.set_style("darkgrid")
        # weights = np.ones_like(vn)/float(len(vn))
        # plt.hist(vn, weights=weights,color='y',bins=50)
        sns.distplot(vn, color="y", bins=50)  # fit=stats.norm
        plt.show()
        plt.savefig(dump_path + "/tvmp/" + f"{k}.png")
        plt.cla()

    for k, v in baseline_model.named_parameters():
        vn = v.detach().numpy().flatten()
        fig = plt.figure()
        sns.set_style("darkgrid")
        # weights = np.ones_like(vn)/float(len(vn))
        # plt.hist(vn, weights=weights,color='y',bins=50)
        sns.distplot(vn, color="y", bins=50)  # fit=stats.norm
        plt.show()
        plt.savefig(dump_path + "/torchp/" + f"{k}.png")
        plt.cla()


# draw_torchviz()
# draw_netron()
# compare_weight()


get_runing_var()
# compare_bnweight()
# tvm.testing.assert_allclose(baseline_output, output, rtol=rtol, atol=atol)

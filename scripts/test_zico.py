import os, sys
import torch
from torch import nn
import numpy as np

def getgrad(model:torch.nn.Module, grad_dict:dict, step_iter=0):
    if step_iter==0:
        for name,mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                # print(mod.weight.grad.data.size())
                # print(mod.weight.data.size())
                grad_dict[name]=[mod.weight.grad.data.cpu().reshape(-1).numpy()]
    else:
        for name,mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                grad_dict[name].append(mod.weight.grad.data.cpu().reshape( -1).numpy())
    return grad_dict

def caculate_zico(grad_dict):
    allgrad_array=None
    for i, modname in enumerate(grad_dict.keys()):
        grad_dict[modname]= np.array(grad_dict[modname])
    nsr_mean_sum = 0
    nsr_mean_sum_abs = 0
    nsr_mean_avg = 0
    nsr_mean_avg_abs = 0
    for j, modname in enumerate(grad_dict.keys()):
        nsr_std = np.std(grad_dict[modname], axis=0)
        nonzero_idx = np.nonzero(nsr_std)[0]
        nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
        tmpsum = np.sum(nsr_mean_abs[nonzero_idx]/nsr_std[nonzero_idx])
        if tmpsum==0:
            pass
        else:
            nsr_mean_sum_abs += np.log(tmpsum)
            nsr_mean_avg_abs += np.log(np.mean(nsr_mean_abs[nonzero_idx]/nsr_std[nonzero_idx]))
    return nsr_mean_sum_abs

def getzico(network, trainloader, lossfunc):
    grad_dict= {}
    network.train()

    network.cuda()
    for i, batch in enumerate(trainloader):
        network.zero_grad()
        data,label = batch[0],batch[1]
        data,label=data.cuda(),label.cuda()

        logits = network(data)[1]
        loss = lossfunc(logits, label)
        loss.backward()
        grad_dict= getgrad(network, grad_dict,i)
        
    res = caculate_zico(grad_dict)
    return res

if __name__ == "__main__":
    from torchvision.models import efficientnet_b0
    from zebanas.data.vision.cifar10 import DataLoaderforSearchGetter
    from zebanas.spaces.model import Network
    from zebanas.genetic.chromosome import Chromosome
    from proxylessnas.proxyless_nas.model_zoo import proxyless_base

    import xautodl
    from xautodl.models import get_cell_based_tiny_net
    from nats_bench import create
    api = create("/home/haitt/workspaces/codes/nas-bench/NATS-Bench/api/NATS-tss-v1_0-3ffb9-simple", "tss", fast_mode=True)

    config = api.get_net_config(15624, 'cifar10')
    infor = api.get_more_info(15624, "cifar10", hp="200")
    model = get_cell_based_tiny_net(config)

    data_getter = DataLoaderforSearchGetter(
        data_dir="/home/haitt/workspaces/data/vision/cifar10",
        batch_size=16,
        n_batches=2,
        image_size=32,
        crop_size=32
    )
    dataloader = data_getter.load()

    # model = proxyless_base(pretrained=False, net_config="https://raw.githubusercontent.com/han-cai/files/master/proxylessnas/proxyless_cifar.config")
    score = getzico(
        model,
        dataloader,
        nn.CrossEntropyLoss()
    )

    print(score)

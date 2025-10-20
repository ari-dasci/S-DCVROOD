from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10


class KNNPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(KNNPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.K = self.args.K
        self.activation_log = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            activation_log = []
            net.eval()
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    images, _ = batch
                    data = images.cuda()
                    data = data.float()

                    _, feature = net(data, return_feature=True)
                    activation_log.append(
                        normalizer(feature.data.cpu().numpy()))

            self.activation_log = np.concatenate(activation_log, axis=0)
            self.index = faiss.IndexFlatL2(feature.shape[1])
            self.index.add(self.activation_log)
            self.setup_flag = True
        else:
            pass

    def setup_emb(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:

            self.activation_log = np.ascontiguousarray(normalizer(id_loader_dict["train"]["embeddings"].data.cpu().numpy())).astype(np.float32)
            self.index = faiss.IndexFlatL2(id_loader_dict["train"]["embeddings"].shape[1])
            self.index.add(self.activation_log)
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output, feature = net(data, return_feature=True)
        feature_normed = normalizer(feature.data.cpu().numpy())
        D, _ = self.index.search(
            feature_normed,
            self.K,
        )
        kth_dist = -D[:, -1]
        _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        return pred, torch.from_numpy(kth_dist)
    
    @torch.no_grad()
    def postprocess_emb(self, net: nn.Module, data: Any):
        output, feature = data["logits"], data["embeddings"]
        feature_normed = np.ascontiguousarray(normalizer(feature.data.cpu().numpy())).astype(np.float32)
        D, _ = self.index.search(
            feature_normed,
            self.K,
        )
        kth_dist = -D[:, -1]
        _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        return pred, torch.from_numpy(kth_dist)

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]

    def get_hyperparam(self):
        return self.K
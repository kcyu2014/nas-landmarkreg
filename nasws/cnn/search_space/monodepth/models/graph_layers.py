import time
import cv2
import scipy.ndimage

import torch
import torch_geometric

import torch.nn as nn
import numpy as np

from skimage import data, segmentation
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt


class Segmentation:
    def __init__(self, img, sigma=0.1 * np.pi):
        self.img = img
        self.sigma = sigma

        self.A, self.segments = build_superpixel_graph(img, sigma)
        self.n_seg = self.A.shape[0]

        self.segments = torch.from_numpy(self.segments)
        self.adj = torch.from_numpy(self.A).float()


def build_superpixel_graph(img, sigma=0.1 * np.pi):
    M, N = img.shape[:2]

    superpixels = segmentation.slic(img, compactness=30, n_segments=img.shape[0] // 2)
    sp_indices = np.unique(superpixels)
    n_sp = len(sp_indices)
    sp_coord = np.zeros((n_sp, 2), np.float32)

    for seg in sp_indices:
        mask = superpixels == seg
        sp_coord[seg] = np.array(scipy.ndimage.measurements.center_of_mass(mask))

    sp_coord /= img.shape[1]
    dist = cdist(sp_coord, sp_coord)

    A = np.exp(-dist / sigma ** 2)
    A[np.diag_indices_from(A)] = 0

    return A, superpixels


def build_multi_scale_graph(img, n_scales, sigma=0.1 * np.pi):
    adjacencies = []
    for i in range(n_scales):
        adjacencies.append(Segmentation(img, sigma))
        img = cv2.resize(img, dsize=(img.shape[0] // 2, img.shape[1] // 2))

    return adjacencies


class Graph2ImageFeatures(nn.Module):
    def __init__(self):
        super(Graph2ImageFeatures, self).__init__()

    # Not working but first step towards batch-based
    def forward_b(self, graph_nodes, spx_image):
        # Flatten spatial dimensions in spx map
        flat_spx = spx_image.view(spx_image.shape[0], -1).unsqueeze(-1)

        # Extend by feature dimension
        flat_spx = flat_spx.expand((*flat_spx.shape[:2], graph_nodes.shape[2]))

        # Gather features from graph nodes to image and reshape
        out = torch.gather(graph_nodes, 1, flat_spx).view((*spx_image.shape, -1))

        return out.permute(0, 3, 1, 2)

    def forward(self, graph_nodes, spx_image):

        # Flatten spatial dimensions in spx map
        flat_spx = spx_image.view(-1, 1)

        # Extend by feature dimension
        flat_spx = flat_spx.expand((flat_spx.shape[0], graph_nodes.shape[1]))

        # Gather features from graph nodes to image and reshape
        out = torch.gather(graph_nodes, 0, flat_spx).view((*spx_image.shape, -1))

        return out.permute(2, 0, 1)


class GraphPlaneDecoder(nn.Module):
    def __init__(self, in_features):
        super(GraphPlaneDecoder, self).__init__()

        self.conv1 = torch_geometric.nn.dense.DenseGCNConv(in_features, 3)
        # self.conv1 = torch_geometric.nn.GCNConv(
        #     in_features, 3, improved=True, cached=True
        # )
        self.graph2im = Graph2ImageFeatures()

    def forward(self, feats, adjs, spxs):
        xp = torch.linspace(-1, 1, spx.shape[1], device=spxs.device)
        yp = torch.linspace(-1, 1, spx.shape[2], device=spxs.device)
        xv, yv = torch.meshgrid(xp, yp)
        xv = xv.unsqueeze(0)
        yv = yv.unsqueeze(0)

        disps = []
        planes = []

        ## TODO: Speed me up
        for i in range(len(adjs)):
            feat = self.conv1(feats[i].unsqueeze(0), adjs[i])
            plane = self.graph2im(feat[0], spxs[i])

            disp = plane[0, :] * xv + plane[1, :] * yv + plane[2, :]

            planes.append(plane.unsqueeze(0))
            disps.append(disp.unsqueeze(0))

        disps = torch.cat(disps, 0)
        planes = torch.cat(planes, 0)

        return disps, planes


# class ImagePlaneDecoder(nn.Module):
#     def __init__(self, in_features):
#         super(ImagePlaneDecoder, self).__init__()

#         self.conv1 = nn.Conv2D(in_features, 3, kernel_size=1)
#         self.graph2im = Graph2ImageFeatures()

#     def forward(self, x, adj, spx):
#         x = self.graph2im(x, spx)
#         x = self.conv1(x)

#         # TODO: Compute depth from plane per pixel

#         return depth


if __name__ == "__main__":

    img = data.coffee()
    img = cv2.resize(img, dsize=(384, 380))

    start = time.time()
    adjacencies = build_multi_scale_graph(img, 4)
    elapsed = time.time() - start
    print(elapsed)

    seg = adjacencies[0]

    in_features = 128
    feat = torch.randn((seg.n_seg, in_features)).float()

    adj = seg.adj

    # edge_index, edge_value = torch_geometric.utils.dense_to_sparse(seg.A)

    # print(edge_index.shape)
    # exit()
    #    spx = seg.segments.unsqueeze(0)
    # Test batchign
    spx = torch.cat((seg.segments.unsqueeze(0), seg.segments.unsqueeze(0)))
    adj = [seg.adj, seg.adj]
    feat = [feat, feat + 0.2]

    model = GraphPlaneDecoder(in_features)

    disp, planes = model(feat, adj, spx)

    loss = disp.mean()

    loss.backward()

    print(planes.shape)
    print(disp.shape)

    plt.subplot(221), plt.imshow(planes[0].permute(1, 2, 0).detach())
    plt.subplot(222), plt.imshow(disp[0, 0].detach())
    plt.subplot(223), plt.imshow(planes[1].permute(1, 2, 0).detach())
    plt.subplot(224), plt.imshow(disp[1, 0].detach())

    plt.show()

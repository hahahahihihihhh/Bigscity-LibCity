# from _typeshed import Self
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from libcity.model.abstract_model import AbstractModel
from libcity.model import loss

from libcity.model.utils import normalize


class gcn_operation(nn.Module):
    def __init__(self, adj, in_dim, out_dim, num_vertices, activation='GLU'):
        """
        图卷积模块
        :param adj: 邻接图
        :param in_dim: 输入维度
        :param out_dim: 输出维度
        :param num_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(gcn_operation, self).__init__()
        self.adj = adj
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_vertices = num_vertices
        self.activation = activation

        assert self.activation in {'GLU', 'relu'}

        if self.activation == 'GLU':
            self.FC = nn.Linear(self.in_dim, 2 * self.out_dim, bias=True)
        else:
            self.FC = nn.Linear(self.in_dim, self.out_dim, bias=True)

    def forward(self, x, mask=None):
        """
        :param x: (3*N, B, Cin)
        :param mask:(3*N, 3*N)
        :return: (3*N, B, Cout)
        """
        adj = self.adj
        if mask is not None:
            adj = adj.to(mask.device) * mask

        x = torch.einsum('nm, mbc->nbc', adj.to(x.device), x)  # 4*N, B, Cin

        if self.activation == 'GLU':
            lhs_rhs = self.FC(x)  # 4*N, B, 2*Cout
            lhs, rhs = torch.split(lhs_rhs, self.out_dim, dim=-1)  # 4*N, B, Cout

            out = lhs * torch.sigmoid(rhs)
            del lhs, rhs, lhs_rhs

            return out

        elif self.activation == 'relu':
            return torch.relu(self.FC(x))  # 3*N, B, Cout


class STSGCM(nn.Module):
    def __init__(self, adj, in_dim, out_dims, num_of_vertices, activation='GLU'):
        """
        :param adj: 邻接矩阵
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(STSGCM, self).__init__()
        self.adj = adj
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices
        self.activation = activation

        self.gcn_operations = nn.ModuleList()

        self.gcn_operations.append(
            gcn_operation(
                adj=self.adj,
                in_dim=self.in_dim,
                out_dim=self.out_dims[0],
                num_vertices=self.num_of_vertices,
                activation=self.activation
            )
        )

        for i in range(1, len(self.out_dims)):
            self.gcn_operations.append(
                gcn_operation(
                    adj=self.adj,
                    in_dim=self.out_dims[i-1],
                    out_dim=self.out_dims[i],
                    num_vertices=self.num_of_vertices,
                    activation=self.activation
                )
            )

    def forward(self, x, mask=None):
        """
        :param x: (3N, B, Cin)
        :param mask: (3N, 3N)
        :return: (N, B, Cout)
        """
        need_concat = []

        for i in range(len(self.out_dims)):
            x = self.gcn_operations[i](x, mask)
            need_concat.append(x)

        # shape of each element is (1, N, B, Cout)
        need_concat = [
            torch.unsqueeze(
                h[self.num_of_vertices: 2 * self.num_of_vertices], dim=0
            ) for h in need_concat
        ]

        out = torch.max(torch.cat(need_concat, dim=0), dim=0).values  # (N, B, Cout)

        del need_concat

        return out


class STSGCL(nn.Module):
    def __init__(self,
                 adj,
                 history,
                 num_of_vertices,
                 in_dim,
                 out_dims,
                 strides=4,
                 activation='GLU',
                 temporal_emb=True,
                 spatial_emb=True):
        """
        :param adj: 邻接矩阵
        :param history: 输入时间步长
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param strides: 滑动窗口步长，local时空图使用几个时间步构建的，默认为3
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        :param temporal_emb: 加入时间位置嵌入向量
        :param spatial_emb: 加入空间位置嵌入向量
        """
        super(STSGCL, self).__init__()
        self.adj = adj
        self.strides = strides
        self.history = history
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices

        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb


        self.conv1 = nn.Conv2d(self.in_dim, self.out_dims[-1], kernel_size=(1, 2), stride=(1, 1), dilation=(1, 3))
        self.conv2 = nn.Conv2d(self.in_dim, self.out_dims[-1], kernel_size=(1, 2), stride=(1, 1), dilation=(1, 3))


        self.STSGCMS = nn.ModuleList()
        for i in range(self.history - self.strides + 1):
            self.STSGCMS.append(
                STSGCM(
                    adj=self.adj,
                    in_dim=self.in_dim,
                    out_dims=self.out_dims,
                    num_of_vertices=self.num_of_vertices,
                    activation=self.activation
                )
            )

        if self.temporal_emb:
            self.temporal_embedding = nn.Parameter(torch.FloatTensor(1, self.history, 1, self.in_dim))
            # 1, T, 1, Cin

        if self.spatial_emb:
            self.spatial_embedding = nn.Parameter(torch.FloatTensor(1, 1, self.num_of_vertices, self.in_dim))
            # 1, 1, N, Cin

        self.reset()

    def reset(self):
        if self.temporal_emb:
            nn.init.xavier_normal_(self.temporal_embedding, gain=0.0003)

        if self.spatial_emb:
            nn.init.xavier_normal_(self.spatial_embedding, gain=0.0003)

    def forward(self, x, mask=None):
        """
        :param x: B, T, N, Cin
        :param mask: (N, N)
        :return: B, T-3, N, Cout
        """
        if self.temporal_emb:
            x = x + self.temporal_embedding

        if self.spatial_emb:
            x = x + self.spatial_embedding

        #############################################
        # shape is (B, C, N, T)
        data_temp = x.permute(0, 3, 2, 1)
        data_left = torch.sigmoid(self.conv1(data_temp))
        data_right = torch.tanh(self.conv2(data_temp))
        data_time_axis = data_left * data_right
        data_res = data_time_axis.permute(0, 3, 2, 1)
        # shape is (B, T-3, N, C)
        #############################################

        need_concat = []
        batch_size = x.shape[0]

        for i in range(self.history - self.strides + 1):
            t = x[:, i: i+self.strides, :, :]  # (B, 4, N, Cin)

            t = torch.reshape(t, shape=[batch_size, self.strides * self.num_of_vertices, self.in_dim])
            # (B, 4*N, Cin)

            t = self.STSGCMS[i](t.permute(1, 0, 2), mask)  # (4*N, B, Cin) -> (N, B, Cout)

            t = torch.unsqueeze(t.permute(1, 0, 2), dim=1)  # (N, B, Cout) -> (B, N, Cout) ->(B, 1, N, Cout)

            need_concat.append(t)

        mid_out = torch.cat(need_concat, dim=1)  # (B, T-3, N, Cout)
        out = mid_out + data_res

        del need_concat, batch_size

        return out


class output_layer(nn.Module):
    def __init__(self, num_of_vertices, history, in_dim, out_dim,
                 hidden_dim=128, horizon=12):
        """
        预测层，注意在作者的实验中是对每一个预测时间step做处理的，也即他会令horizon=1
        :param num_of_vertices:节点数
        :param history:输入时间步长
        :param in_dim: 输入维度
        :param hidden_dim:中间层维度
        :param horizon:预测时间步长
        """
        super(output_layer, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.history = history
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon

        #print("#####################")
        #print(self.in_dim)
        #print(self.history)
        #print(self.hidden_dim)

        self.FC1 = nn.Linear(self.in_dim * self.history, self.hidden_dim, bias=True)

        #self.FC2 = nn.Linear(self.hidden_dim, self.horizon , bias=True)

        self.FC2 = nn.Linear(self.hidden_dim, self.horizon * self.out_dim, bias=True)

    def forward(self, x):
        """
        :param x: (B, Tin, N, Cin)
        :return: (B, Tout, N)
        """
        batch_size = x.shape[0]

        x = x.permute(0, 2, 1, 3)  # B, N, Tin, Cin

        out1 = torch.relu(self.FC1(x.reshape(batch_size, self.num_of_vertices, -1)))
        # (B, N, Tin, Cin) -> (B, N, Tin * Cin) -> (B, N, hidden)

        out2 = self.FC2(out1)  # (B, N, hidden) -> (B, N, horizon * 2)

        out2 = out2.reshape(batch_size, self.num_of_vertices, self.horizon, self.out_dim)

        del out1, batch_size

        return out2.permute(0, 2, 1, 3)  # B, horizon, N
        # return out2.permute(0, 2, 1)  # B, horizon, N


class STFGNN(AbstractModel):
    def __init__(self, config, data_feature):
        """

        :param adj: local时空间矩阵
        :param history:输入时间步长
        :param num_of_vertices:节点数量
        :param in_dim:输入维度
        :param hidden_dims: lists, 中间各STSGCL层的卷积操作维度
        :param first_layer_embedding_size: 第一层输入层的维度
        :param out_layer_dim: 输出模块中间层维度
        :param activation: 激活函数 {relu, GlU}
        :param use_mask: 是否使用mask矩阵对adj进行优化
        :param temporal_emb:是否使用时间嵌入向量
        :param spatial_emb:是否使用空间嵌入向量
        :param horizon:预测时间步长
        :param strides:滑动窗口步长，local时空图使用几个时间步构建的，默认为4
        """
        super(STFGNN, self).__init__(config, data_feature)

        self.config = config
        self.data_feature = data_feature
        self.scaler = data_feature["scaler"]
        self.num_batches = data_feature["num_batches"]
        self.file_name = self.config.get("filename", f"./raw_data/{config['dataset']}/{config['dataset']}.npz")
        self.adj_mx = self.data_feature["adj_mx"]

        history = self.config.get("window", 12)
        num_of_vertices = data_feature.get('num_nodes', 1)

        in_dim = self.config.get("input_dim", 1)
        out_dim = self.config.get("output_dim", 1)
        hidden_dims = self.config.get("hidden_dims", None)
        first_layer_embedding_size = self.config.get("first_layer_embedding_size", None)
        out_layer_dim = self.config.get("out_layer_dim", None)
        activation = self.config.get("activation", "GLU")
        use_mask = self.config.get("mask")
        temporal_emb = self.config.get("temporal_emb", True)
        spatial_emb = self.config.get("spatial_emb", True)
        horizon = self.config.get("horizon", 12)
        strides = self.config.get("strides", 4)

        self.num_of_vertices = num_of_vertices
        self.hidden_dims = hidden_dims
        self.out_layer_dim = out_layer_dim
        self.activation = activation
        self.use_mask = use_mask

        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.horizon = horizon
        self.strides = strides

        self.First_FC = nn.Linear(in_dim, first_layer_embedding_size, bias=True)
        self.STSGCLS = nn.ModuleList()
        #print("____________________")
        #print(history)
        self.period = self.config.get("period", 288)
        self.sparsity = self.config.get("sparsity", 0.01)
        self._load_origin_data(self.file_name)  # self.rawdat
        self.adj = torch.FloatTensor(self._construct_adj())
        self._scaler = self.data_feature.get('scaler')
        self.output_dim = self.data_feature.get('output_dim', 1)

        self.STSGCLS.append(
            STSGCL(
                adj=self.adj,
                history=history,
                num_of_vertices=self.num_of_vertices,
                in_dim=first_layer_embedding_size,
                out_dims=self.hidden_dims[0],
                strides=self.strides,
                activation=self.activation,
                temporal_emb=self.temporal_emb,
                spatial_emb=self.spatial_emb
            )
        )

        in_dim = self.hidden_dims[0][-1]
        history -= (self.strides - 1)

        #print("!!!!!!!!!!!!!!!!!!!")
        #print(history)

        for idx, hidden_list in enumerate(self.hidden_dims):
            #print("?????? ", idx)
            if idx == 0:
                continue
            #print("---------", idx)
            self.STSGCLS.append(
                STSGCL(
                    adj=self.adj,
                    history=history,
                    num_of_vertices=self.num_of_vertices,
                    in_dim=in_dim,
                    out_dims=hidden_list,
                    strides=self.strides,
                    activation=self.activation,
                    temporal_emb=self.temporal_emb,
                    spatial_emb=self.spatial_emb
                )
            )
            history -= (self.strides - 1)
            in_dim = hidden_list[-1]

        self.predictLayer = nn.ModuleList()
        #print("***********************")
        #print(history)
        for t in range(self.horizon):
            self.predictLayer.append(
                output_layer(
                    num_of_vertices=self.num_of_vertices,
                    history=history,
                    in_dim=in_dim,
                    out_dim = out_dim,
                    hidden_dim=out_layer_dim,
                    horizon=1
                )
            )

        if self.use_mask:
            mask = torch.zeros_like(self.adj)
            mask[self.adj != 0] = self.adj[self.adj != 0]
            self.mask = nn.Parameter(mask)
        else:
            self.mask = None

    def _load_origin_data(self, file_name):
        if file_name[-3:] == "txt":
            fin = open(file_name)
            self.rawdat = np.loadtxt(fin, delimiter=',')
        elif file_name[-3:] == "csv":
            self.rawdat = pd.read_csv(file_name).values
        elif file_name[-2:] == "h5":
            self.rawdat = pd.read_hdf(file_name)
        elif file_name[-3:] == "npz":
            mid_dat = np.load(file_name)
            self.rawdat = mid_dat[mid_dat.files[0]]
        else:
            raise ValueError('file_name type error!')

    def _compute_dtw(self, a, b, order=1, Ts=12, normal=True):
        if normal:
            a = normalize(a)
            b = normalize(b)
        T0 = a.shape[1]
        d = np.reshape(a, [-1, 1, T0]) - np.reshape(b, [-1, T0, 1])
        d = np.linalg.norm(d, axis=0, ord=order)
        D = np.zeros([T0, T0])
        for i in range(T0):
            for j in range(max(0, i - Ts), min(T0, i + Ts + 1)):
                if (i == 0) and (j == 0):
                    D[i, j] = d[i, j] ** order
                    continue
                if (i == 0):
                    D[i, j] = d[i, j] ** order + D[i, j - 1]
                    continue
                if (j == 0):
                    D[i, j] = d[i, j] ** order + D[i - 1, j]
                    continue
                if (j == i - Ts):
                    D[i, j] = d[i, j] ** order + min(D[i - 1, j - 1], D[i - 1, j])
                    continue
                if (j == i + Ts):
                    D[i, j] = d[i, j] ** order + min(D[i - 1, j - 1], D[i, j - 1])
                    continue
                D[i, j] = d[i, j] ** order + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
        return D[-1, -1] ** (1.0 / order)

    def _construct_adj_fusion(self, A, A_dtw, steps):
        '''
        construct a bigger adjacency matrix using the given matrix

        Parameters
        ----------
        A: np.ndarray, adjacency matrix, shape is (N, N)

        steps: how many times of the does the new adj mx bigger than A

        Returns
        ----------
        new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)

        ----------
        This is 4N_1 mode:

        [T, 1, 1, T
         1, S, 1, 1
         1, 1, S, 1
         T, 1, 1, T]

        '''

        N = len(A)
        adj = np.zeros([N * steps] * 2)  # "steps" = 4 !!!

        for i in range(steps):
            if (i == 1) or (i == 2):
                adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A
            else:
                adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A_dtw
        # '''
        for i in range(N):
            for k in range(steps - 1):
                adj[k * N + i, (k + 1) * N + i] = 1
                adj[(k + 1) * N + i, k * N + i] = 1
        # '''
        adj[3 * N: 4 * N, 0:  N] = A_dtw  # adj[0 * N : 1 * N, 1 * N : 2 * N]
        adj[0: N, 3 * N: 4 * N] = A_dtw  # adj[0 * N : 1 * N, 1 * N : 2 * N]

        adj[2 * N: 3 * N, 0: N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
        adj[0: N, 2 * N: 3 * N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
        adj[1 * N: 2 * N, 3 * N: 4 * N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
        adj[3 * N: 4 * N, 1 * N: 2 * N] = adj[0 * N: 1 * N, 1 * N: 2 * N]

        for i in range(len(adj)):
            adj[i, i] = 1

        return adj

    def _gen_data(self, data, ntr, N):
        '''
        if flag:
            data=pd.read_csv(fname)
        else:
            data=pd.read_csv(fname,header=None)
        '''
        # data=data.as_matrix()
        data = np.reshape(data, [-1, 24, N])
        return data[0:ntr]

    def _construct_dtw(self):
        data = self.rawdat[:, :, 0]
        total_day = data.shape[0] / self.period
        tr_day = int(total_day * 0.6)
        n_route = data.shape[1]
        xtr = self._gen_data(data, tr_day, n_route)
        print(np.shape(xtr))
        N = n_route
        d = np.zeros([N, N])
        for i in range(N):
            for j in range(i + 1, N):
                d[i, j] = self._compute_dtw(xtr[:, :, i], xtr[:, :, j])

        print("The calculation of time series is done!")
        dtw = d + d.T
        n = dtw.shape[0]
        w_adj = np.zeros([n, n])
        adj_percent = self.sparsity
        top = int(n * adj_percent)
        for i in range(dtw.shape[0]):
            a = dtw[i, :].argsort()[0:top]
            for j in range(top):
                w_adj[i, a[j]] = 1

        for i in range(n):
            for j in range(n):
                if (w_adj[i][j] != w_adj[j][i] and w_adj[i][j] == 0):
                    w_adj[i][j] = 1
                if (i == j):
                    w_adj[i][j] = 1

        print("Total route number: ", n)
        print("Sparsity of adj: ", len(w_adj.nonzero()[0]) / (n * n))
        print("The weighted matrix of temporal graph is generated!")
        self.dtw = w_adj

    def _construct_adj(self):
        """
        构建local 时空图
        :param A: np.ndarray, adjacency matrix, shape is (N, N)
        :param steps: 选择几个时间步来构建图
        :return: new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)
        """
        self._construct_dtw()
        adj_mx = self._construct_adj_fusion(self.adj_mx, self.dtw, self.strides)
        print("The shape of localized adjacency matrix: {}".format(
            adj_mx.shape), flush=True)
        return adj_mx

    def forward(self, batch):   # batch
        """
        :param x: B, Tin, N, Cin
        :return: B, Tout, N, Cout
        """
        inputs = batch['X']
        x = torch.relu(self.First_FC(inputs))  # B, Tin, N, Cin

        for model in self.STSGCLS:
            x = model(x, self.mask)
        # (B, T - 8, N, Cout)
        #print(2)
        need_concat = []
        for i in range(self.horizon):
            out_step = self.predictLayer[i](x)  # (B, 1, N, 2)
            need_concat.append(out_step)
        #print(3)
        out = torch.cat(need_concat, dim=1)  # B, Tout, N, Cout

        del need_concat

        return out

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        # print('y_true', y_true.shape)
        # print('y_predicted', y_predicted.shape)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, x):
        return self.forward(x)

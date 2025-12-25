import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    lap = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(lap, 1, which='LM')
        lambda_max = lambda_max[0]
    lap = sp.csr_matrix(lap)
    m, _ = lap.shape
    identity = sp.identity(m, format='csr', dtype=lap.dtype)
    lap = (2 / lambda_max * lap) - identity
    return lap.astype(np.float32).todense()


class NConv(nn.Module):
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, adj):
        x = torch.einsum('ncvl,vw->ncwl', (x, adj))
        return x.contiguous()


class Linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(Linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN, self).__init__()
        self.nconv = NConv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class TemporalAttentionLayer(nn.Module):

    def __init__(self, device, in_channels, num_of_vertices, num_of_timesteps):
        super(TemporalAttentionLayer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(device))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(device))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(device))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(device))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(device))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.zeros_(self.be)
        nn.init.xavier_uniform_(self.Ve)
        nn.init.xavier_uniform_(self.U2)
        nn.init.uniform_(self.U1, a=-0.1, b=0.1)
        nn.init.uniform_(self.U3, a=-0.1, b=0.1)

    def forward(self, x):
        """
        Args:
            x: (batch_size, N, F_in, T)

        Returns:
            torch.tensor: (B, T, T)
        """

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)   # 左张量最后一维，右张量倒数第二魏

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        e = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        e_normalized = F.softmax(e, dim=2)

        return e_normalized


class SATSkipGate(nn.Module):
    """
    SAT gate for skip feature maps (Select-Attend-Transfer)
    Input/Output: s in (B, C, N, T)

    Select:  s_sel = s ⊙ trelu(w)          (channel-wise)
    Attend:  a = sigmoid(conv1x1(s_sel))   (spatial attention map, 1 channel)
    Transfer: s_out = s_sel ⊙ a            (gated skip feature)
    """
    def __init__(self, channels: int):
        super().__init__()
        # W in paper: one scalar per channel
        self.w = nn.Parameter(torch.ones(channels))  # init to 1 => keep original behavior at start
        # K in paper: 1x1 filter to aggregate channels to 1 attention map
        self.k = nn.Conv2d(channels, 1, kernel_size=(1, 1), bias=True)

    @staticmethod
    def trelu(x):
        # truncated ReLU: clip to [0, 1]
        return torch.clamp(x, min=0.0, max=1.0)

    def forward(self, s):
        # Select (channel-wise)
        w = self.trelu(self.w).view(1, -1, 1, 1)     # (1, C, 1, 1)
        s_sel = s * w                                # (B, C, N, T)
        # Attend (spatial)
        a = torch.sigmoid(self.k(s_sel))             # (B, 1, N, T)
        # Transfer (gated feature)
        s_out = s_sel * a                            # (B, C, N, T)
        return s_out


class STGNN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        self.adj_mx = data_feature.get('adj_mx')
        self.A_sta = self.adj_mx
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 2)
        super().__init__(config, data_feature)

        self.eps = config.get('eps', 1e-3)
        self.dropout = config.get('dropout', 0.3)
        self.blocks = config.get('blocks', 4)
        self.layers = config.get('layers', 2)
        self.gcn_bool = config.get('gcn_bool', True)
        self.addaptadj = config.get('addaptadj', True)
        self.addfusionadj = config.get('addfusionadj', True)
        self.fc = nn.Linear(2, 1)
        self.adjtype = config.get('adjtype', 'doubletransition')
        self.randomadj = config.get('randomadj', True)
        self.aptonly = config.get('aptonly', True)
        self.kernel_size = config.get('kernel_size', 2)
        self.nhid = config.get('nhid', 32)
        self.residual_channels = config.get('residual_channels', self.nhid)
        self.dilation_channels = config.get('dilation_channels', self.nhid)
        self.skip_channels = config.get('skip_channels', self.nhid * 8)
        self.end_channels = config.get('end_channels', self.nhid * 16)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.device = config.get('device', torch.device('cpu'))

        self.apt_layer = config.get('apt_layer', True)
        if self.apt_layer:
            self.layers = np.int(
                np.round(np.log((((self.input_window - 1) / (self.blocks * (self.kernel_size - 1))) + 1)) / np.log(2)))
            print('# of layers change to %s' % self.layers)

        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.skip_gates = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=self.feature_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))

        self.cal_adj(self.adjtype)
        self.supports = [torch.tensor(i).to(self.device) for i in self.adj_mx]
        if self.randomadj:
            self.aptinit = None
        else:
            self.aptinit = self.supports[0]
        if self.aptonly:
            self.supports = None

        receptive_field = self.output_dim

        self.supports_len = 0
        if self.supports is not None:
            self.supports_len += len(self.supports)

        if self.gcn_bool and self.addaptadj:
            if self.aptinit is None:
                if self.supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, 10).to(self.device),
                                             requires_grad=True).to(self.device)
                self.nodevec2 = nn.Parameter(torch.randn(10, self.num_nodes).to(self.device),
                                             requires_grad=True).to(self.device)
                self.supports_len += 1
            else:
                if self.supports is None:
                    self.supports = []
                m, p, n = torch.svd(self.aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(self.device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(self.device)
                self.supports_len += 1

        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                   out_channels=self.dilation_channels,
                                                   kernel_size=(1, self.kernel_size), dilation=new_dilation))
                # print(self.filter_convs[-1])
                self.gate_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                 out_channels=self.dilation_channels,
                                                 kernel_size=(1, self.kernel_size), dilation=new_dilation))
                # print(self.gate_convs[-1])
                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=self.dilation_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=(1, 1)))
                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=self.dilation_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=(1, 1)))
                self.skip_gates.append(SATSkipGate(self.skip_channels))  # 新增：SAT gate
                self.bn.append(nn.BatchNorm2d(self.residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(GCN(self.dilation_channels, self.residual_channels,
                                          self.dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=self.end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.output_window,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.receptive_field = receptive_field
        self.num_skip_layers = self.blocks * self.layers
        self.skip_alpha = nn.Parameter(torch.zeros(self.num_skip_layers))  # learnable layer weights
        num_of_timestamps = receptive_field
        self.tAtt = nn.ModuleList()
        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            for i in range(self.layers):
                num_of_timestamps -= additional_scope
                additional_scope *= 2
                self.tAtt.append(TemporalAttentionLayer(self.device, self.residual_channels, self.num_nodes, num_of_timestamps))
        self._logger.info('receptive_field: ' + str(self.receptive_field))

    def forward(self, batch):
        inputs = batch['X']  # (batch_size, input_window, num_nodes, feature_dim)
        inputs = inputs.transpose(1, 3)  # (batch_size, feature_dim, num_nodes, input_window)
        inputs = nn.functional.pad(inputs, (1, 0, 0, 0))  # (batch_size, feature_dim, num_nodes, input_window+1)

        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = inputs
        x = self.start_conv(x)  # (batch_size, residual_channels, num_nodes, self.receptive_field)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            A_sta = torch.tensor(self.A_sta).to(self.device)
            A_adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            if self.addfusionadj:
                # fusion
                A_stack = torch.stack([A_sta, A_adp], dim=-1)
                gate = self.fc(A_stack)
                F_gate = torch.sigmoid(gate).squeeze(-1)
                A_fused = F_gate * A_adp + (1.0 - F_gate) * A_sta
                A_thr = F.relu(A_fused - self.eps)
                # row normalize
                A_adpsta = A_thr / A_thr.sum(dim=1, keepdim=True)
                A_adpsta = torch.nan_to_num(A_adpsta)
                A_adpsta = torch.nan_to_num(A_adpsta, nan=0.0, posinf=0.0, neginf=0.0)
                new_supports = self.supports + [A_adpsta]
            else:
                new_supports = self.supports + [A_adp]
        skip_list = []
        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            # (dilation, init_dilation) = self.dilations[i]
            # residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # (batch_size, residual_channels, num_nodes, self.receptive_field)
            # dilated convolution
            filter = self.filter_convs[i](residual)
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            # parametrized skip connection
            s = x
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            s = self.skip_convs[i](s)
            s = self.skip_gates[i](s)  # SAT-gated skip (B, C, N, T_i)
            skip_list.append(s)
            # (batch_size, skip_channels, num_nodes, receptive_field-kernel_size+1)
            # try:
            #     skip = skip[:, :, :, -s.size(3):]
            # except(Exception):
            #     skip = 0
            # skip = s + skip
            # (batch_size, skip_channels, num_nodes, receptive_field-kernel_size+1)
            if self.gcn_bool and self.supports is not None:
                # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
                # (batch_size, residual_channels, num_nodes, receptive_field-kernel_size+1)
            else:
                # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
                x = self.residual_convs[i](x)
                # (batch_size, residual_channels, num_nodes, receptive_field-kernel_size+1)
            x_temp = x.permute(0, 2, 1, 3)
            temporal_at = self.tAtt[i](x_temp)
            x_tat = torch.matmul(x.reshape(x_temp.shape[0], -1, x_temp.shape[3]), temporal_at) \
                .reshape(x_temp.shape[0], x_temp.shape[1], x_temp.shape[2], x_temp.shape[3])
            x = x_tat.permute(0, 2, 1, 3)
            # residual: (batch_size, residual_channels, num_nodes, self.receptive_field)
            x = x + residual[:, :, :, -x.size(3):]
            # (batch_size, residual_channels, num_nodes, receptive_field-kernel_size+1)
            x = self.bn[i](x)
        # ---- Weighted SAT Skip Aggregation ----
        # Align all skip tensors to the same temporal length (use the smallest T)
        min_t = min(s.size(3) for s in skip_list)
        skip_list = [s[:, :, :, -min_t:] for s in skip_list]
        # Layer-wise weights (softmax) -> (L,)
        w = torch.softmax(self.skip_alpha, dim=0)
        # Weighted sum
        skip = 0.0
        for i, s in enumerate(skip_list):
            skip = skip + w[i] * s
        x = F.relu(skip)
        # (batch_size, skip_channels, num_nodes, self.output_dim)
        x = F.relu(self.end_conv_1(x))
        # (batch_size, end_channels, num_nodes, self.output_dim)
        x = self.end_conv_2(x)
        # (batch_size, output_window, num_nodes, self.output_dim)
        return x

    def cal_adj(self, adjtype):
        if adjtype == "scalap":
            self.adj_mx = [calculate_scaled_laplacian(self.adj_mx)]
        elif adjtype == "normlap":
            self.adj_mx = [calculate_normalized_laplacian(self.adj_mx).astype(np.float32).todense()]
        elif adjtype == "symnadj":
            self.adj_mx = [sym_adj(self.adj_mx)]
        elif adjtype == "transition":
            self.adj_mx = [asym_adj(self.adj_mx)]
        elif adjtype == "doubletransition":
            self.adj_mx = [asym_adj(self.adj_mx), asym_adj(np.transpose(self.adj_mx))]
        elif adjtype == "identity":
            self.adj_mx = [np.diag(np.ones(self.adj_mx.shape[0])).astype(np.float32)]
        else:
            assert 0, "adj type not defined"

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        # print('y_true', y_true.shape)
        # print('y_predicted', y_predicted.shape)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)

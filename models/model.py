from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import copy
from models.transformer import Transformer
import matplotlib.pyplot as plt


class GraphConvolutionLayer_t(nn.Module):
    def __init__(self, in_channels, out_channels, frames, dropout_rate=0.4):
        super(GraphConvolutionLayer_t, self).__init__()
        self.weight = nn.Parameter(torch.ones(in_channels, out_channels)*(1/in_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.dropout = nn.Dropout(dropout_rate)
        self.adjweight = nn.Parameter(torch.ones(frames))


    def forward(self, x, adj_matrix):

        batch_size, frames, in_channels = x.size()
        adj_weight = adj_matrix * self.adjweight
        adj_weight = F.softmax(adj_weight,dim=1)
        featuremap_adj_t = adj_weight
        x = x.permute(0, 2, 1)
        x = torch.bmm(x, adj_weight)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        weight = self.weight.unsqueeze(0).repeat(batch_size,1,1)
        attention_weights = F.softmax(weight, dim=1)
        attention_weights_t = attention_weights

        output = torch.bmm(x, attention_weights)

        return output, featuremap_adj_t, attention_weights_t


class GraphConvolutionalNetwork_t(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphConvolutionalNetwork_t, self).__init__()
        self.gc1_t = GraphConvolutionLayer_t(in_channels, hidden_channels,frames=68)
        self.gc2_t = GraphConvolutionLayer_t(hidden_channels, out_channels,frames=68)
        '''Fis V'''
        # self.gc1_t = GraphConvolutionLayer_t(in_channels, hidden_channels,frames=124)
        # self.gc2_t = GraphConvolutionLayer_t(hidden_channels, out_channels,frames=124)

    def forward(self, x, adj_matrix):


        result, featuremap_adj_t1,attention_weights_t1 = self.gc1_t(x, adj_matrix)
        x = F.elu(result,alpha=1.0)
        result2, featuremap_adj_t2, attention_weights_t2 = self.gc2_t(x, adj_matrix)
        x = F.elu(result2, alpha=1.0)
        return x,featuremap_adj_t2, attention_weights_t2
channels_t = 1024
hidden_channels_t = 512
output_channels_t = 1024




def construct_adjacency_matrix_t(image_size):

    adj_matrix = torch.zeros((image_size, image_size), dtype=torch.float)

    # 遍历所有数据点
    for i in range(image_size):
        # 每个点与自己相连
        adj_matrix[i, i] = 1

        # 如果不是第一个点，则与前一个点相连
        if i > 0:
            adj_matrix[i, i - 1] = 1
            adj_matrix[i - 1, i] = 1

        # 如果不是最后一个点，则与后一个点相连
        if i < image_size-1:
            adj_matrix[i, i + 1] = 1
            adj_matrix[i + 1, i] = 1


    return adj_matrix

class GraphConvolutionLayer_s(nn.Module):
    def __init__(self, in_channels, out_channels, frames, dropout_rate=0.4):
        super(GraphConvolutionLayer_s, self).__init__()
        self.weight = nn.Parameter(torch.ones(in_channels, out_channels)*(1/in_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.dropout = nn.Dropout(dropout_rate)
        self.adjweight = nn.Parameter(torch.ones(frames))



    def forward(self, x, adj_matrix):

        batch_size, frames, in_channels = x.size()
        adj_weight = adj_matrix * self.adjweight
        adj_weight = F.softmax(adj_weight,dim=1)
        featuremap_adj_s = adj_weight
        x = torch.bmm(x, adj_weight)
        x = self.dropout(x)


        x = x.permute(0, 2, 1)
        weight = self.weight.unsqueeze(0).repeat(batch_size,1,1)
        attention_weights = F.softmax(weight, dim=1)
        attention_weights_s = attention_weights
        output = torch.bmm(x, attention_weights)
        output = output.permute(0, 2, 1)
        return output, featuremap_adj_s, attention_weights_s
class GraphConvolutionalNetwork_s(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphConvolutionalNetwork_s, self).__init__()
        self.gc1_s = GraphConvolutionLayer_s(in_channels, hidden_channels,frames=1024)
        self.gc2_s = GraphConvolutionLayer_s(hidden_channels, out_channels,frames=1024)

    def forward(self, x, adj_matrix):
        result, featuremap_adj_s1, attention_weights_s1 = self.gc1_s(x, adj_matrix)
        x = F.elu(result,alpha=1.0)
        result2, featuremap_adj_s2, attention_weights_s2 = self.gc2_s(x, adj_matrix)
        x = F.elu(result2,alpha=1.0)
        return x, featuremap_adj_s2, attention_weights_s2
channels_s = 68
hidden_channels_s = 136
output_channels_s = 68

# channels_s = 124
# hidden_channels_s = 248
# output_channels_s = 124


def construct_adjacency_matrix_s(image_size):
    num_pixels = image_size * image_size
    # 28*28= 784
    adjacency_matrix_s = torch.zeros((num_pixels, num_pixels), dtype=torch.float)

    # Iterate over all pixels
    for i in range(num_pixels):
        # Get row and column indices of current pixel
        row = i // image_size
        col = i % image_size

        # Check if neighboring pixels are within the image boundaries
        if row > 0:
            adjacency_matrix_s[i, i - image_size] = 1.0  # Connect to pixel above
        if row < image_size - 1:
            adjacency_matrix_s[i, i + image_size] = 1.0  # Connect to pixel below
        if col > 0:
            adjacency_matrix_s[i, i - 1] = 1.0  # Connect to pixel on the left
        if col < image_size - 1:
            adjacency_matrix_s[i, i + 1] = 1.0  # Connect to pixel on the right

        # Connect to diagonal pixels
        if row > 0 and col > 0:
            adjacency_matrix_s[i, i - image_size - 1] = 1.0  # Connect to pixel on the top-left diagonal
        if row > 0 and col < image_size - 1:
            adjacency_matrix_s[i, i - image_size + 1] = 1.0  # Connect to pixel on the top-right diagonal
        if row < image_size - 1 and col > 0:
            adjacency_matrix_s[i, i + image_size - 1] = 1.0  # Connect to pixel on the bottom-left diagonal
        if row < image_size - 1 and col < image_size - 1:
            adjacency_matrix_s[i, i + image_size + 1] = 1.0  # Connect to pixel on the bottom-right diagonal

        # Connect to self
        adjacency_matrix_s[i, i] = 1.0

    return adjacency_matrix_s



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        try:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
        except AttributeError:
            pass

class ASGTN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, n_decoder, n_query, dropout):
        super(ASGTN, self).__init__()
        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(0.1)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2.data.fill_(0.1)
        self.gcn_t = GraphConvolutionalNetwork_t(channels_t, hidden_channels_t, output_channels_t).cuda()
        self.gcn_s = GraphConvolutionalNetwork_s(channels_s, hidden_channels_s, output_channels_s).cuda()

        self.in_proj = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),
            nn.ReLU(),
            nn.Conv1d(kernel_size=1, in_channels=in_dim // 2, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )



        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout
        )

        self.prototype = nn.Embedding(n_query, hidden_dim)

        self.weight = torch.linspace(0, 1, n_query, requires_grad=False).cuda()

        self.regressor = nn.Linear(hidden_dim, n_query)


    def forward(self, x):
        # x (b, t, c)

        b, t, c = x.shape

        adj_matrix_t = construct_adjacency_matrix_t(t)
        adj_matrix_t.unsqueeze(0)
        adj_matrix_t = adj_matrix_t.repeat(b,1,1).cuda()
        x1, featuremap_adj_t2, attention_weights_t2  = self.gcn_t(x, adj_matrix_t)

        adj_matrix_s = construct_adjacency_matrix_s(32)
        adj_matrix_s.unsqueeze(0)
        adj_matrix_s = adj_matrix_s.repeat(b,1,1).cuda()
        x2, featuremap_adj_s2, attention_weights_s2 = self.gcn_s(x, adj_matrix_s)

        x = x * 0.8 + x1 * self.w1 + x2 * self.w2

        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2)
        q = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        encode_x = self.transformer.encoder(x)
        q1, attention_decoder = self.transformer.decoder(q, encode_x)



        s = self.regressor(q1)  # (b, n, n) (32,4,4)
        s = torch.diagonal(s, dim1=-2, dim2=-1)  # (b, n)
        norm_s = torch.sigmoid(s)
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out = torch.sum(self.weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)


        return {'output': out, 'embed': q1}

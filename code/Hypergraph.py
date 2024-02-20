import torch.nn.functional as F
import torch
from torch_geometric.nn import HypergraphConv, Linear
import torch.nn as nn

class Hypergraph(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hyperedge_weight):
        super().__init__()
        self.conv_hyper = HypergraphConv(in_channels, out_channels, hyperedge_weight=None, use_attention=False)
        self.W1 = nn.Linear(in_channels, 64)
        self.W2 = nn.Linear(64, out_channels)
        self.dropout = nn.Dropout(0.5)
        
    def weighted_average(self, tensors, weights):
        weighted_tensors = [tensor * weight for tensor, weight in zip(tensors, weights)]
        weighted_sum = sum(weighted_tensors)
        return weighted_sum
    
    def PAfusion(self, product, av, hyper):
        product_tensor = torch.stack([torch.arange(av.size(0), av.size(0) + product.size(0)), torch.tensor(range(product.size(0)))])
        final_tensor = torch.cat([hyper, product_tensor.cuda()], dim=1)
        node = torch.cat([av, product], dim=0)
        result = self.conv_hyper(node, final_tensor)
        return result
    
    def CPAfusion(self, product, av, category, hyper_CPA, hyper_PA):
       # hyper_PA = hyper_PA.cpu()
        category_tensor = torch.stack([torch.arange(product.size(0), product.size(0) + category.size(0)), torch.tensor(range(category.size(0)))])
        p_av_dictionary = {key: [] for key in range(torch.max(hyper_PA[1]).item()+1)}
        for i in range(len(hyper_PA[1])):
            p_av_dictionary[hyper_PA[1][i].item()].append(hyper_PA[0][i].item())
        
        tensor1 = []
        tensor2 = []
        for i in range(len(hyper_CPA[0])):
            tensor1.extend(p_av_dictionary[hyper_CPA[0][i].item()])
            for j in range(len(p_av_dictionary[hyper_CPA[0][i].item()])):
                tensor2.append(hyper_CPA[1][i].item())
        tensor1_n = [x + (product.size(0) + category.size(0)) for x in tensor1]       
        av_tensor = torch.stack([torch.tensor(tensor1_n), torch.tensor(tensor2)])

        final_tensor = torch.cat([hyper_CPA, category_tensor, av_tensor], dim=1).long()
        node = torch.cat([product, category, av], dim=0)
        result = self.conv_hyper(node, final_tensor.cuda())
        return result
        
        
    
    def hyper_calculation(self, node, matrix):
        H = dglsp.spmatrix(matrix)
        H = H + dglsp.identity(H.shape)
        # Compute node degree.
        d_V = H.sum(1)
        # Compute edge degree.
        d_E = H.sum(0)
        # Compute the inverse of the square root of the diagonal D_v.
        D_v_invsqrt = dglsp.diag(d_V**-0.5)
        # Compute the inverse of the diagonal D_e.
        D_e_inv = dglsp.diag(d_E**-1)
        n_edges = d_E.shape[0]
        
        B = dglsp.identity((n_edges, n_edges)) # B is identity matrix
        L = D_v_invsqrt @ H @ B @ D_e_inv @ H.T @ D_v_invsqrt
        node_i = self.W1(self.dropout(node)).cpu()
        out = L @ node_i
        X = F.relu(out)
        node_i = self.W2(self.dropout(X.cuda())).cpu()
        out = L @ node_i
        
        return out
        
    def forward(self, info, hyperedges, hyperedge_type): #info -> dict; hyperedges -> dict
        #also_view_out = self.hyper_calculation(info['product'], hyperedges['also_view'])
        #also_buy_out = self.hyper_calculation(info['product'], hyperedges['also_buy'])
        
        if hyperedge_type == 'user':
        
            also_view_out = self.conv_hyper(info['product'], hyperedges['also_view'].cuda())
            also_buy_out = self.conv_hyper(info['product'], hyperedges['also_buy'].cuda())
            out_p = self.weighted_average([also_view_out, also_buy_out], [0.5, 0.5])
            x_dict = {
              "product": out_p.cuda(),
              "category": info['category'].cuda(),
              "av": info['av'].cuda(),
            }
            
        elif hyperedge_type == 'inventory':
            
            hyper_PA = self.PAfusion(info['product'], info['av'], hyperedges['product_aspect'])
            hyper_PA_av_node = hyper_PA[:info['av'].size(0), :]
            hyper_PA_p_node = hyper_PA[info['av'].size(0):, :]
     
            hyper_CPA = self.CPAfusion(info['product'], info['av'], info['category'], hyperedges['category_product_aspect'], hyperedges['product_aspect'])
            hyper_CPA_p_node = hyper_CPA[:info['product'].size(0), :]
            hyper_CPA_c_node = hyper_CPA[info['product'].size(0): info['product'].size(0)+info['category'].size(0)]
            hyper_CPA_av_node = hyper_CPA[-info['av'].size(0):]
            
            out_p = self.weighted_average([hyper_PA_p_node, hyper_CPA_p_node], [0.5, 0.5])
            out_av = self.weighted_average([hyper_PA_av_node, hyper_CPA_av_node], [0.5, 0.5])
            x_dict = {
              "product": out_p.cuda(),
              "category": hyper_CPA_c_node.cuda(),
              "av": out_av.cuda(),
            }
            
        elif hyperedge_type == 'all':
            also_view_out = self.conv_hyper(info['product'], hyperedges['also_view'].cuda())
            also_buy_out = self.conv_hyper(info['product'], hyperedges['also_buy'].cuda())
            hyper_PA = self.PAfusion(info['product'], info['av'], hyperedges['product_aspect'])
            hyper_PA_av_node = hyper_PA[:info['av'].size(0), :]
            hyper_PA_p_node = hyper_PA[info['av'].size(0):, :]
     
            hyper_CPA = self.CPAfusion(info['product'], info['av'], info['category'], hyperedges['category_product_aspect'], hyperedges['product_aspect'])
            hyper_CPA_p_node = hyper_CPA[:info['product'].size(0), :]
            hyper_CPA_c_node = hyper_CPA[info['product'].size(0): info['product'].size(0)+info['category'].size(0)]
            hyper_CPA_av_node = hyper_CPA[-info['av'].size(0):]
            
            out_p = self.weighted_average([also_view_out, also_buy_out, hyper_PA_p_node, hyper_CPA_p_node], [0.4, 0.4, 0.1, 0.1])
            out_av = self.weighted_average([hyper_PA_av_node, hyper_CPA_av_node], [0.5, 0.5])
            x_dict = {
              "product": out_p.cuda(),
              "category": hyper_CPA_c_node.cuda(),
              "av": out_av.cuda(),
            }
            
            
        else:
            
            print('Specify the hyperedge_type.')
            x_dict = {
              "product": info['product'].cuda(),
              "category": info['category'].cuda(),
              "av": info['av'].cuda(),
            }
        
        
        return x_dict
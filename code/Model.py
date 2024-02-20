from torch_geometric.nn import SAGEConv, to_hetero, HypergraphConv, GATConv, Linear
from BERTEncoder import BERTEncoder
from torch_geometric.data import HeteroData
from GNN import GNN
from Classifier import Classifier
from Hypergraph import Hypergraph
import torch.nn.functional as F
import torch

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, data):
        super().__init__()
        self.bert = BERTEncoder(max_length=512)

        self.product_emb = torch.nn.Embedding(data["product"].num_nodes, hidden_channels)
        self.av_emb = torch.nn.Embedding(data["av"].num_nodes, hidden_channels)
        self.category_emb = torch.nn.Embedding(data["category"].num_nodes, hidden_channels)
        
        self.hypergraph = Hypergraph(hidden_channels,hidden_channels,None)

        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, metadata=data.metadata(), aggr='sum')

        
        self.classifier = Classifier()
        
    def forward(self, data: HeteroData, hyperedge_info, embedding_type, hyperedge_type):
        if embedding_type == 'nodeID':
        
            x_dict = {
              "product": self.product_emb(data["product"].node_id),
              "category": self.category_emb(data['category'].node_id),
              "av": self.av_emb(data["av"].node_id),
            }
        elif embedding_type == 'bert':
        
            x_dict = {
              "product": data['product'].x,
              "category": data['category'].x,
              "av": data["av"].x,
            }
        else:
        
            try:
                product = self.bert(data['product'].x.squeeze())
                category = self.bert(data['category'].x.squeeze())
                av = self.bert(data['av'].x.squeeze())
                x_dict = {
                  "product": product,
                  "category": category,
                  "av": av,
                }
            except:
                x_dict = {
                  "product": self.product_emb(data["product"].node_id),
                  "category": self.category_emb(data['category'].node_id),
                  "av": self.av_emb(data["av"].node_id),
                }
        
        x_dict = self.hypergraph(x_dict, hyperedge_info, hyperedge_type)
        
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
      
        x_dicts = self.gnn(x_dict, data.edge_index_dict)
        
        
        pred = self.classifier(
            x_dicts["product"],
            x_dicts["av"],
            data["product", "has", "av"].edge_label_index,
        )
        
        
        return pred
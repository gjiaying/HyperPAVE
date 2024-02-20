import torch

class Classifier(torch.nn.Module):
    def forward(self, x_product, x_av, edge_label_index):
        # Convert node embeddings to edge-level representations:
        edge_feat_product = x_product[edge_label_index[0]]
        edge_feat_av = x_av[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        feature = (edge_feat_product * edge_feat_av).sum(dim=-1)
        #features = nn.Sigmoid(feature)
        return feature
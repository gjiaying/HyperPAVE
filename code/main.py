import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero, HypergraphConv, GATConv, Linear
import torch.nn.functional as F
import torch
import pandas as pd
import ast
from collections import Counter
import numpy as np
import torch.nn as nn
from BERTEncoder import BERTEncoder
from Generator import Generator
from Model import Model
import copy
import random
from collections import defaultdict
import statistics
import sys
import warnings
import dgl.sparse as dglsp
import argparse
import tqdm
import math
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import ndcg_score
from itertools import chain


warnings.filterwarnings("ignore")
torch.set_printoptions(threshold=10000)


def GraphConstruction(data_path, save_path, max_length, max_length_g):
    # edge construction
    data = HeteroData()
    data_df = pd.read_csv(data_path)
    unique_product_id = data_df['asin'].unique()
    unique_product_id = pd.DataFrame(data={
        'asin': unique_product_id,
        'mappedID': pd.RangeIndex(len(unique_product_id)),
    })
    '''
    unique_category_id = data_df['category_m_id'].unique()
    unique_category_id = pd.DataFrame(data={
        'category_m_id': unique_category_id,
        'mappedID': pd.RangeIndex(len(unique_category_id)),
    })
    '''
    unique_av_id = data_df['label_text'].values.tolist()
    av_id = []
    y = []
    for item in unique_av_id:
        y.append(len(ast.literal_eval(item)))
        for i in range(len(ast.literal_eval(item))):  
            av_id.append(ast.literal_eval(item)[i])
    av_ids = [*set(av_id)]
    asin_id = np.repeat(data_df['asin'].to_numpy(), y)
    data_df2 = pd.DataFrame({'asin':asin_id, 'av_id': av_id})

    unique_av_id = pd.DataFrame(data={
        'av_id': av_ids,
        'mappedID': pd.RangeIndex(len(av_ids)),
    })

    unique_c_id = data_df['category_a'].values.tolist()
    c_id = []
    y = []
    for item in unique_c_id:
        y.append(len(ast.literal_eval(item)))
        for i in range(len(ast.literal_eval(item))):  
            c_id.append(ast.literal_eval(item)[i])
    c_ids = [*set(c_id)]
    asin_id_c = np.repeat(data_df['asin'].to_numpy(), y)
    data_df3 = pd.DataFrame({'asin':asin_id_c, 'c_id': c_id})

    unique_c_id = pd.DataFrame(data={
        'c_id': c_ids,
        'mappedID': pd.RangeIndex(len(c_ids)),
    })


    product_id = pd.merge(data_df['asin'], unique_product_id, left_on='asin', right_on='asin', how='left')
    product_id = torch.from_numpy(product_id['mappedID'].values)
    #category_id = pd.merge(data_df['category_m_id'], unique_category_id, left_on='category_m_id', right_on='category_m_id', how='left')
    #category_id = torch.from_numpy(category_id['mappedID'].values)

    product2_id = pd.merge(data_df2['asin'], unique_product_id, left_on='asin', right_on='asin', how='left')
    product2_id = torch.from_numpy(product2_id['mappedID'].values)
    av_id = pd.merge(data_df2['av_id'], unique_av_id, left_on='av_id', right_on='av_id', how='left')
    av_id = torch.from_numpy(av_id['mappedID'].values)

    product3_id = pd.merge(data_df3['asin'], unique_product_id, left_on='asin', right_on='asin', how='left')
    product3_id = torch.from_numpy(product3_id['mappedID'].values)
    c_id = pd.merge(data_df3['c_id'], unique_c_id, left_on='c_id', right_on='c_id', how='left')
    category_id = torch.from_numpy(c_id['mappedID'].values)

    # With this, we are ready to construct our `edge_index` in COO format
    # following PyG semantics:
    edge_index_product_to_category = torch.stack([product3_id, category_id], dim=0)
    data["product", "in", "category"].edge_index = edge_index_product_to_category
    edge_index_product_to_av = torch.stack([product2_id, av_id], dim=0)
    data["product", "has", "av"].edge_index = edge_index_product_to_av

    data["product"].node_id = torch.arange(len(unique_product_id))
    data["category"].node_id = torch.arange(len(unique_c_id))
    data["av"].node_id = torch.arange(len(unique_av_id))

    #Node Embedding:
    product = data_df['title&description'].unique()
    #category = data_df['category_m'].unique()
    encoder = BERTEncoder(max_length)
    generator = Generator(max_length_g)

    product_node = []
    for item in product:
        text, mask = encoder.other_tokenize(item)
        text = encoder(text)
        text = text.squeeze()
        product_node.append(text.detach().numpy())

    product_node = torch.FloatTensor(product_node)

    category_node = []
    for item in c_ids:
        item = generator(item)
        text, mask = encoder.other_tokenize(item)
        text = encoder(text)
        text = text.squeeze()
        category_node.append(text.detach().numpy())
    category_node = torch.FloatTensor(category_node)

    av_node = []
    for item in av_ids:
        item = item.replace(":", " is")
        #text = item + '. ' + generator(item)
        text, mask = encoder.other_tokenize(item)
        text = encoder(text)
        text = text.squeeze()
        av_node.append(text.detach().numpy())
    av_node = torch.FloatTensor(av_node)

    data['product'].x = product_node
    data['category'].x = category_node
    data['av'].x = av_node

    data = T.ToUndirected()(data)
    #print(data)
    torch.save(data, save_path)
    print('Graph is constructed.')
    
def HypergraphConstruction(input_file, graph_file):
    data = torch.load(graph_file)
    data_df = pd.read_csv(input_file)
    # P-P-P
    also_view = data_df['also_view'].values.tolist()
    num_hyperedge = 0
    h1, h2 = [], []
    for i in range(len(also_view)):
        if also_view[i] != '[]':
            also_view[i] = ast.literal_eval(also_view[i])
            N = [j for j in range(len(data_df['asin'].unique())) if data_df['asin'].unique()[j] in also_view[i]]
            if len(N) > 1:
                h1.extend(N)
                h2.extend([num_hyperedge for i in range(len(N))])
                num_hyperedge += 1
    hyperedge_index_alsoview = torch.tensor([h1, h2])

    also_buy = data_df['also_buy'].values.tolist()
    num_hyperedge = 0
    h1, h2 = [], []
    for i in range(len(also_buy)):
        if also_buy[i] != '[]':
            also_buy[i] = ast.literal_eval(also_buy[i])
            N = [j for j in range(len(data_df['asin'].unique())) if data_df['asin'].unique()[j] in also_buy[i]]
            if len(N) > 1:
                h1.extend(N)
                h2.extend([num_hyperedge for i in range(len(N))])
                num_hyperedge += 1
    hyperedge_index_alsobuy = torch.tensor([h1, h2])


    # P-A
    unique_av_id = data_df['label_text'].values.tolist()
    av_id = []
    y, h2 = [], []
    num_hyperedge = 0
    for item in unique_av_id:
        h2.extend([num_hyperedge for i in range(len(ast.literal_eval(item)))])
        num_hyperedge += 1
        y.append(len(ast.literal_eval(item)))
        for i in range(len(ast.literal_eval(item))):  
            av_id.append(ast.literal_eval(item)[i])
    av_ids = [*set(av_id)]
    asin_id = np.repeat(data_df['asin'].to_numpy(), y)
    data_df2 = pd.DataFrame({'asin':asin_id, 'av_id': av_id})

    unique_av_id = pd.DataFrame(data={
        'av_id': av_ids,
        'mappedID': pd.RangeIndex(len(av_ids)),
    })
    av_id = pd.merge(data_df2['av_id'], unique_av_id, left_on='av_id', right_on='av_id', how='left')
    av_id = av_id['mappedID'].values.tolist()
    hyperedge_PA = torch.tensor([av_id, h2])

    #C-P-A
    unique_c_id = data_df['category_a'].values.tolist()
    product_id = data_df['asin'].unique()
    c_id = []
    y = []
    for item in unique_c_id:
        for i in range(len(ast.literal_eval(item))):  
            c_id.append(ast.literal_eval(item)[i])
    keys = [*set(c_id)]
    c_dictionary = {key: [] for key in keys}

    for index, element in enumerate(unique_c_id):
        for i in range(len(ast.literal_eval(element))):
            value_list = c_dictionary.get(ast.literal_eval(element)[i], [])
            value_list.append(index)
            c_dictionary[ast.literal_eval(element)[i]] = value_list

    h1, h2, num_hyperedge = [], [], 0
    for values in c_dictionary.values():
        h1.extend(values)
        h2.extend([num_hyperedge for i in range(len(values))])
        num_hyperedge += 1
    hyperedge_CPA = torch.tensor([h1, h2])

    hyperedge_info = {'also_view': hyperedge_index_alsoview, 'also_buy': hyperedge_index_alsobuy, 'product_aspect': hyperedge_PA, 'category_product_aspect': hyperedge_CPA}
    return hyperedge_info

def data_sampling(data, NUMBER, percentage):
    t = data["product", "has", "av"].edge_index
    seed = random.sample(range(0, data['av'].x.size(0)), NUMBER*2)
    av_seed = []
    for i in range(len(seed[0:NUMBER])): 
        av_seed.extend((t[1] == seed[i]).nonzero()[:,0].tolist())
    product_seed = torch.index_select(t,1, torch.IntTensor(av_seed))[0]
    val_mask = [index for index,value in enumerate(t[0]) if value in product_seed.tolist()] 
    
    av_seed = []
    for i in range(len(seed[NUMBER:])): 
        av_seed.extend((t[1] == seed[NUMBER+i]).nonzero()[:,0].tolist())
    product_seed = torch.index_select(t,1, torch.IntTensor(av_seed))[0]
    test_mask = [index for index,value in enumerate(t[0]) if value in product_seed.tolist()] 
    
    edge_mask = list(range(0,t.size(1)))
    val_seen = random.sample(range(0, t.size(1)), int(len(val_mask)/percentage))
    test_seen = random.sample(range(0, t.size(1)), int(len(test_mask)/percentage))
    val_result = [x for x in val_seen if x not in val_mask]
    test_result = [x for x in test_seen if x not in test_mask]
   
    return val_result, test_result, seed

def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() if len(locs)>1)

def zero_sampling(data):
    positive_edge_index = data['product', 'has', 'av'].edge_label_index
    # calculating negative samples
    source = positive_edge_index[0].tolist()
    av_mask = list(range(0,data['av'].node_id.size(0)))

    d1 = []
    d2 = []
    labels = []
    for dup in sorted(list_duplicates(source)):

        d2_in = []
        d1_in = []
        label = []
        positive = sorted(positive_edge_index[1][dup[1]].tolist())
        negative = [x for x in av_mask if x not in positive]
        negative_key = [dup[0]]*len(negative)
        positive_key = [dup[0]]*len(positive)

        d1_in.extend(positive_key)
        d1_in.extend(negative_key)
        d2_in.extend(positive)
        d2_in.extend(negative)
        d2.append(d2_in)
        d1.append(d1_in)
        
        label.extend([1]*len(positive))
        label.extend([0]*len(negative))
        labels.append(label)


    d1 = torch.LongTensor(d1)
    d2 = torch.LongTensor(d2)
    final_label = torch.LongTensor(labels)
    samples = torch.stack((d1,d2), dim = 0)  
    return samples, final_label

def ZeroshotHypergraph(data, NUMBER):
    #data = torch.load(f'./graphs/graph_applications.pt')
    val_mask, test_mask, seed = data_sampling(data, NUMBER, 1.0) #percentage is the unseen data percentage

    transform = T.RandomLinkSplit(
        num_val=0.0,
        num_test=0.0,
        is_undirected = True,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=0.0,
        add_negative_train_samples=False,
        edge_types=("product", "has", "av"),
        rev_edge_types=("av", "rev_has", "product"), 
    )

    test_mask = [x for x in test_mask if x not in val_mask]
    all_mask = list(range(0,data['product', 'has', 'av'].num_edges))
    val_train_mask = [x for x in all_mask if x not in test_mask]
    val_test_mask = val_mask + test_mask
    train_mask = [x for x in all_mask if x not in val_test_mask]

    train_data = copy.copy(data)
    train_data['product', 'has', 'av'].edge_index = train_data['product', 'has', 'av'].edge_index[:, train_mask]

    val_data = copy.copy(data)
    val_data['product', 'has', 'av'].edge_index = train_data['product', 'has', 'av'].edge_index
    val_data['product', 'has', 'av'].edge_label_index = data['product', 'has', 'av'].edge_index[:, val_mask]
    val_data['product', 'has', 'av'].edge_label=torch.ones(val_data['product', 'has', 'av'].edge_label_index.size(1))

    train_data, _, _ = transform(train_data)

    test_data = copy.copy(data)
    test_data['product', 'has', 'av'].edge_index = data['product', 'has', 'av'].edge_index[:, val_train_mask]
    test_data['product', 'has', 'av'].edge_label_index = data['product', 'has', 'av'].edge_index[:, test_mask]
    test_data['product', 'has', 'av'].edge_label=torch.ones(test_data['product', 'has', 'av'].edge_label_index.size(1))


    val_data['product', 'has', 'av'].edge_label_index, val_data['product', 'has', 'av'].edge_label = zero_sampling(val_data)
    test_data['product', 'has', 'av'].edge_label_index, test_data['product', 'has', 'av'].edge_label = zero_sampling(test_data)

    return train_data, val_data, test_data

def add_values_to_tensor(tensor, values_list):
    tensor_set = set(tensor.tolist())
    new_values = [val for val in values_list if val not in tensor_set]
    updated_tensor = torch.cat((tensor, torch.tensor(new_values)))
    return updated_tensor

def hyperedge_calculation(data, hyperedge_info):
    hyperedge_view = hyperedge_info
    product = data # Tensor product number
    tensor_dict = {val.item(): idx for idx, val in enumerate(data)}

    unique_list = list(tensor_dict.values())

    hyperedge_view = hyperedge_view.to('cuda')
    mask = torch.isin(hyperedge_view[0], product)
    tensor1 = hyperedge_view[0][mask]
    tensor2 = hyperedge_view[1][mask]
    mapped_tensor = torch.tensor([tensor_dict[val.item()] for val in tensor1])
    updated_tensor1 = add_values_to_tensor(mapped_tensor, unique_list)

    value_counts = torch.bincount(tensor2)
    value_counts = value_counts[value_counts != 0]
    
    generated_tensor = torch.IntTensor([])
    for value, frequency in enumerate(value_counts):
        repeated_values = torch.tensor([value] * frequency)
        generated_tensor = torch.cat((generated_tensor, repeated_values))
    

    new_length = len(updated_tensor1)-len(generated_tensor)
    begin,_ = torch.max(generated_tensor, dim=0)
    begin = begin.item()+1
    tensor_part2 = torch.arange(begin, begin + new_length)
    updated_tensor2 = torch.cat((generated_tensor, tensor_part2)) 
    final_tensor = torch.stack((updated_tensor1, updated_tensor2), dim=0)
    
    return final_tensor

def hyperedge_calculation_PA(data, hyperedge_info):
    hyperedge_view = hyperedge_info
    product = data # Tensor product number
    tensor_dict = {val.item(): idx for idx, val in enumerate(data)}
    unique_list = list(tensor_dict.values())
    hyperedge_view = hyperedge_view.to('cuda')
    mask = torch.isin(hyperedge_view[1], product)
    tensor1 = hyperedge_view[0][mask]
    tensor2 = hyperedge_view[1][mask] 
    value_counts = torch.bincount(tensor2)
    value_counts = value_counts[value_counts != 0] 
    generated_tensor = torch.IntTensor([])
    for value, frequency in enumerate(value_counts):
        repeated_values = torch.tensor([value] * frequency)
        generated_tensor = torch.cat((generated_tensor, repeated_values))
    final_tensor = torch.stack((tensor1, generated_tensor.to('cuda')), dim=0)
    return final_tensor

def hyperedge_update(data, hyperedge_info):
    final_also_view = hyperedge_calculation(data, hyperedge_info['also_view'])
    final_also_buy = hyperedge_calculation(data, hyperedge_info['also_buy'])
    final_hyperedge_PA = hyperedge_calculation_PA(data, hyperedge_info['product_aspect'])
    final_hyperedge_CPA = hyperedge_calculation(data, hyperedge_info['category_product_aspect'])
   # hyper_info = {'also_view': final_also_view, 'also_buy': final_also_buy}
    hyper_info = {'also_view': final_also_view, 'also_buy': final_also_buy, 'product_aspect': final_hyperedge_PA, 'category_product_aspect': final_hyperedge_CPA}
    return hyper_info

def hit(ground_truth, pred, k):
    result = []
    for i in range(len(ground_truth)):
        y_pred = sorted(range(len(pred[i])), key=lambda x: pred[i][x])[-k:]
        y_true = [x for x, e in enumerate(ground_truth[i]) if e == 1]
        overlap = [value for value in y_pred if value in y_true]
        result.append(len(overlap)/len(y_true))        
    return sum(result)/len(result)

def map(ground_truth, pred):
    result = []
    for i in range(len(ground_truth)):
        result.append(average_precision_score(ground_truth[i], pred[i]))
    return sum(result)/len(result)


def validation(val_data, hyperedge_info, model, device, BATCH_SIZE, embedding_type, hyperedge_type):
    edge_label_index = val_data["product", "has", "av"].edge_label_index
    edge_label = val_data["product", "has", "av"].edge_label
    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[20, 10],
        edge_label_index=(("product", "has", "av"), edge_label_index),
        edge_label=edge_label,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    sampled_data = next(iter(val_loader))
    
    preds = []
    ground_truths = []
    
    for sampled_data in (val_loader):
        validationStep_loss = []
        with torch.no_grad():
            sampled_data.to(device)
            try:
                hyperedge_info_new = hyperedge_update(sampled_data['product'].node_id, hyperedge_info)
            except:
                continue
            preds.append(model(sampled_data, hyperedge_info_new, embedding_type, hyperedge_type))
            ground_truths.append(sampled_data["product", "has", "av"].edge_label)
            target = sampled_data["product", "has", "av"].edge_label
            validation_loss = F.binary_cross_entropy_with_logits(model(sampled_data, hyperedge_info_new, embedding_type, hyperedge_type), target.float())
            validationStep_loss.append(validation_loss.item())
               
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

    new_pred = np.where(pred > 0, 1, 0)
    #list(chain.from_iterable())

    auc = roc_auc_score(ground_truth.flatten(), pred.flatten())
    f1_macro = f1_score(ground_truth, new_pred, average='macro')
    f1_micro = f1_score(ground_truth, new_pred, average='micro')

    mrr = label_ranking_average_precision_score(ground_truth, new_pred)
    ndcg = ndcg_score(ground_truth, pred)
    hit_10 = hit(ground_truth, pred, 10)
    hit_50 = hit(ground_truth, pred, 50)
    hit_100 = hit(ground_truth, pred, 100)
    Map = map(ground_truth.tolist(), pred.tolist())

    return validationStep_loss, auc, f1_macro, f1_micro, mrr, ndcg, hit_10, hit_50, hit_100, Map

def Test_Evaluation(test_data, hyperedge_info, model, device, BATCH_SIZE, RUNS, embedding_type, hyperedge_type):
    auc, f1_macro, f1_micro, mrr, ndcg, hit_5, hit_10, hit_50, hit_100 = [],[],[],[],[],[],[],[],[]
    # Define the testing seed edges:
    for i in range(RUNS):
        edge_label_index = test_data["product", "has", "av"].edge_label_index
        edge_label = test_data["product", "has", "av"].edge_label
        test_loader = LinkNeighborLoader(
            data=test_data,
            num_neighbors=[20, 10],
            edge_label_index=(("product", "has", "av"), edge_label_index),
            edge_label=edge_label,
            batch_size=3 * BATCH_SIZE,
            shuffle=True,
        )
        sampled_data = next(iter(test_loader))

        preds = []
        ground_truths = []
        for sampled_data in tqdm.tqdm(test_loader):
            with torch.no_grad():
                sampled_data.to(device)
                try:
                    hyperedge_info_new = hyperedge_update(sampled_data['product'].node_id, hyperedge_info)
                except:
                    continue
                preds.append(model(sampled_data, hyperedge_info_new, embedding_type, hyperedge_type))
                ground_truths.append(sampled_data["product", "has", "av"].edge_label)

        pred = torch.cat(preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

        new_pred = np.where(pred > 0, 1, 0)
        #list(chain.from_iterable())



        auc.append(roc_auc_score(ground_truth.flatten(), pred.flatten()))
        f1_macro.append(f1_score(ground_truth, new_pred, average='macro'))
        f1_micro.append(f1_score(ground_truth, new_pred, average='micro'))

        mrr.append(label_ranking_average_precision_score(ground_truth, new_pred))
        ndcg.append(ndcg_score(ground_truth, pred, k=50))
        hit_5.append(hit(ground_truth, pred, 5))
        hit_10.append(hit(ground_truth, pred, 10))
        hit_50.append(hit(ground_truth, pred, 50))
        hit_100.append(hit(ground_truth, pred, 100))


    print(f"Testing AUC: {statistics.mean(auc):.4f}+{statistics.stdev(auc): .2f}")
    print(f"Macro F1: {statistics.mean(f1_macro):.4f}+{statistics.stdev(f1_macro): .2f}")
    print(f"Micro F1: {statistics.mean(f1_micro):.4f}+{statistics.stdev(f1_micro): .2f}")
    print(f"MRR: {statistics.mean(mrr):.4f}+{statistics.stdev(mrr): .2f}")
    print(f"NDCG: {statistics.mean(ndcg):.4f}+{statistics.stdev(ndcg): .2f}")
    print(f"Hit@5: {statistics.mean(hit_5):.4f}+{statistics.stdev(hit_5): .2f}")
    print(f"Hit@10: {statistics.mean(hit_10):.4f}+{statistics.stdev(hit_10): .2f}")
    print(f"Hit@50: {statistics.mean(hit_50):.4f}+{statistics.stdev(hit_50): .2f}")
    print(f"Hit@100: {statistics.mean(hit_100):.4f}+{statistics.stdev(hit_100): .2f}")

    


def main():
    #GPU checking:
    if torch.cuda.is_available():    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
        
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', default='./data_csv/example.csv',
            help='Please input data csv file.')
    parser.add_argument('--graph_path', default='./graphs/graph_example_bert.pt',
            help='Please input graph saving path.')
    parser.add_argument('--graph_file', default='./graphs/graph_example_bert.pt',
            help='Please input graph .pt path.')
    parser.add_argument('--NUMBER', default=5,
            help='Please input the number of zero-shot attribute value pairs.')
    parser.add_argument('--max_length', default=512,
            help='Please input the max length for BERTEncoder.')
    parser.add_argument('--max_length_g', default=32,
            help='Please input the max length for the generator.')
    parser.add_argument('--hidden_size', default=768,
            help='Please input the hidden_size for graph.')
    parser.add_argument('--epoch', default=50,
            help='Please input number of training epochs.')
    parser.add_argument('--batch_size', default=4,
            help='Please input batch size.')
    parser.add_argument('--RUNS', default=10,
            help='Please input number of experiment runs for testing evaluation.')
    parser.add_argument('--learning_rate', default=5e-5,
            help='Please input learning rate.')
    parser.add_argument('--weight_decay', default=1e-6,
            help='Please input weight_decay.')
    parser.add_argument('--embedding_type', default='bertf',
            help='Please input type of embedding.')
    parser.add_argument('--hyperedge_type', default='user',
            help='Please input type of hyperedges.')
    opt = parser.parse_args()
    data_input_file = opt.data_file
    graph_file = opt.graph_path
    graph_pt = opt.graph_file
    NUMBER = opt.NUMBER
    max_length = opt.max_length
    max_length_g = opt.max_length_g
    HID_SIZE = opt.hidden_size
    EPOCH = opt.epoch
    BATCH_SIZE = opt.batch_size
    RUNS = opt.RUNS
    learning_rate = opt.learning_rate
    weight_decay = opt.weight_decay
    embedding_type = opt.embedding_type
    hyperedge_type = opt.hyperedge_type
    
    GraphConstruction(data_input_file, graph_file, max_length, max_length_g)
    data = torch.load(graph_pt)
    hyperedge_info = HypergraphConstruction(data_input_file, graph_pt)
    #zero-shot checker
    for i in range(10):

        train_data, val_data, test_data = ZeroshotHypergraph(data, NUMBER)
        if val_data['product', 'has', 'av'].edge_label_index[0].size()[0] != 0 and test_data['product', 'has', 'av'].edge_label_index[0].size()[0] != 0:
            break
        else:
            continue
    
    print(train_data)
    print(val_data)
    print(test_data)
    
    edge_label_index = train_data["product", "has", "av"].edge_label_index
    edge_label = train_data["product", "has", "av"].edge_label
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[20],#20
        neg_sampling_ratio=2.0,
        directed = False,
        edge_label_index=(("product", "has", "av"), edge_label_index),
        edge_label=edge_label,
        batch_size=BATCH_SIZE,#128
        shuffle=True,
    )
    batch = next(iter(train_loader))

    
    model = Model(HID_SIZE, data)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)#lr=0.00001
    
    trainingEpoch_loss = []
    validationEpoch_loss = []
    for epoch in range(EPOCH):
        step_loss = []
        model.train()
        total_loss = total_examples = 0
        for sampled_data in train_loader:
            optimizer.zero_grad() 
            sampled_data.to(device)
            try:
                hyperedge_info_new = hyperedge_update(sampled_data['product'].node_id, hyperedge_info)
            except:
                continue
            pred = model(sampled_data, hyperedge_info_new, embedding_type, hyperedge_type)
            ground_truth = sampled_data["product", "has", "av"].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
            step_loss.append(loss.item())

            new_pred = np.where(pred.cpu().detach().numpy() > 0, 1, 0)

            f1_macro = f1_score(ground_truth.cpu().detach().numpy(), new_pred,average='macro')
        #print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}") 
        sys.stdout.write('step: {0:4} | loss: {1:2.6f}, F1_Macro: {2:3.2f}'.format(epoch, total_loss / total_examples, f1_macro) +'\r')
        sys.stdout.flush()

        trainingEpoch_loss.append(np.array(step_loss).mean())   
        model.eval()
        validationStep_loss, auc, f1_macro, f1_micro, mrr, ndcg, hit_10, hit_50, hit_100, Map = validation(val_data, hyperedge_info, model, device, BATCH_SIZE, embedding_type, hyperedge_type)
        validationEpoch_loss.append(np.array(validationStep_loss).mean())
    
    
    #Testing Results
    Test_Evaluation(test_data, hyperedge_info, model, device, BATCH_SIZE, RUNS, embedding_type, hyperedge_type)
    
    
        
    
if __name__ == "__main__":
    main()
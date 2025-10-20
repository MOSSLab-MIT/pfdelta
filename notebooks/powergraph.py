# PowerGrid dataset is licensed under a CC BY-SA 4.0 license.

"""
General file to load the Inmemory datasets (UK, IEEE24, IEEE39, IEEE118)
"""


import os.path as osp
import torch
import mat73
from sklearn.model_selection import train_test_split
import os
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
import torch
from torch_geometric.data import Data
import numpy as np
import scipy
from gen_utils import from_edge_index_to_adj, padded_datalist


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def index_edgeorder(edge_order):
    return torch.tensor(edge_order["bList"]-1)

class PowerGrid(InMemoryDataset):
    # Base folder to download the files
    names = {
        "uk": ["uk", "Uk", "UK", None],
        "ieee24": ["ieee24", "Ieee24", "IEEE24", None],
        "ieee39": ["ieee39", "Ieee39", "IEEE39", None],
        "ieee118": ["ieee118", "Ieee118", "IEEE118", None],
        "texas": ["texas", "texas", "texas", None],
            }

    def __init__(self, root, name, datatype='Binary', transform=None, pre_transform=None, pre_filter=None):
        
        self.datatype = datatype.lower()
        self.name = name.lower()
        self.raw_path = os.path.join(root, 'raw')
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        
        assert self.name in self.names.keys()
        super(PowerGrid, self).__init__(root, transform, pre_transform, pre_filter)
        print(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False) 

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        if self.datatype.lower() == 'binary':
            return osp.join(self.root, self.name, "processed_b")
        if self.datatype.lower() == 'regression':
            return osp.join(self.root, self.name, "processed_r")
        if self.datatype.lower() == 'multiclass':
            return osp.join(self.root, self.name, "processed_m")
        if self.datatype.lower() == 'node':
            return osp.join(self.root, self.name, "processed_node")
        if self.datatype.lower() == 'nodeopf':
            return osp.join(self.root, self.name, "processed_nodeopf")

    @property
    def raw_file_names(self):
        # List of the raw files
        return ['Bf.mat',
                'blist.mat',
                'Ef.mat',
                "exp.mat",
                'of_bi.mat',
                'of_mc.mat',
                'of_reg.mat',
                'X.mat',
                'Y_polar.mat',
                'edge_index.mat',
                'edge_attr.mat',
                'edge_index_opf.mat',
                'edge_attr_opf.mat',
                'Xopf.mat',
                'Y_polar_opf.mat',
                ]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download the file specified in self.url and store
        # it in self.raw_dir
        path = self.raw_path

    def process(self):
        # function that deletes row
        def th_delete(tensor, indices):
            mask = torch.ones(tensor.size(), dtype=torch.bool)
            mask[indices] = False
            return tensor[mask]

        if self.datatype.lower() == 'binary' or self.datatype.lower() == 'regression' or self.datatype.lower() == 'multiclass':

            # load branch list also called edge order or edge index
            path = os.path.join(self.raw_dir, 'blist.mat')
            edge_order = mat73.loadmat(path)
            edge_order = torch.tensor(edge_order["bList"] - 1)
            # load output binary classification labels
            path = os.path.join(self.raw_dir, 'of_bi.mat')
            of_bi = mat73.loadmat(path)
            # load output binary regression labels
            path = os.path.join(self.raw_dir, 'of_reg.mat')
            of_reg = mat73.loadmat(path)
            # load output mc labels
            path = os.path.join(self.raw_dir, 'of_mc.mat')
            of_mc = mat73.loadmat(path)
            # load output node feature matrix
            path = os.path.join(self.raw_dir, 'Bf.mat')
            node_f = mat73.loadmat(path)
            # load output edge feature matrix
            path = os.path.join(self.raw_dir, 'Ef.mat')
            edge_f = mat73.loadmat(path)
            # load explanations
            path = os.path.join(self.raw_dir, "exp.mat")
            exp = mat73.loadmat(path)

            node_f = node_f['B_f_tot']
            edge_f = edge_f['E_f_post']
            of_bi = of_bi['output_features']
            of_mc = of_mc['category']
            of_reg = of_reg['dns_MW']
            exp_mask = exp["explainations"]

            data_list = []
            adj_list = []
            max_num_nodes = 0
            index = 0
            # MAIN data processing loop
            for i in range(int(len(node_f))):
                # node feat
                x = torch.tensor(node_f[i][0], dtype=torch.float32).reshape([-1, 3]).to(device)
                # edge feat
                f = torch.tensor(edge_f[i][0], dtype=torch.float32)
                e_mask = torch.zeros(len(edge_f[i][0]), 1)
                if exp_mask[i][0] is None:  # .all() == 0:
                    e_mask = e_mask
                else:
                    e_mask[exp_mask[i][0].astype('int')-1] = 1
                # contigency lists, finds where do we have contigencies from the .mat edge feature matrices
                # ( if a line is part of the contigency list all egde features are set 0)
                cont = [j for j in range(len(f)) if np.all(np.array(f[j])) == 0]
                e_mask_post = th_delete(e_mask, cont)
                e_mask_post = torch.cat((e_mask_post, e_mask_post), 0).to(device)
                # remove edge features of the associated line
                f_tot = th_delete(f, cont).reshape([-1, 4]).type(torch.float32)
                # concat the post-contigency edge feature matrix to take into account the reversed edges
                f_totw = torch.cat((f_tot, f_tot), 0).to(device)
                # remove failed lines from branch list
                edge_iw = th_delete(edge_order, cont).reshape(-1, 2).type(torch.long)
                # flip branch list
                edge_iwr = torch.fliplr(edge_iw)
                #  and concat the non flipped and flipped branch list
                edge_iw = torch.cat((edge_iw, edge_iwr), 0)
                edge_iw = edge_iw.t().contiguous().to(device)

                data_type = self.datatype.lower()
                if self.datatype.lower() == 'binary':
                    ydata = torch.tensor(of_bi[i][0], dtype=torch.float, device=device).view(1, -1)
                if self.datatype.lower() == 'regression':
                    ydata = torch.tensor(of_reg[i], dtype=torch.float, device=device).view(1, -1)
                if self.datatype.lower() == 'multiclass':
                    #do argmax
                    ydata = torch.tensor(np.argmax(of_mc[i][0]), dtype=torch.float, device=device).view(1, -1)
                    # ydata = torch.tensor(of_mc[i][0], dtype=torch.int, device=device).view(1, -1)
                # Fill Data object, 1 Data object -> 1 graph

                data = Data(x=x, edge_index=edge_iw, edge_attr=f_totw, y=ydata.to(torch.float), edge_mask=e_mask_post, idx=index)
                #index+=1
                #if ydata == 0:
                #    ydata_cf = torch.tensor(1, dtype=torch.int, device=device).view(-1)
                #else:
                #    ydata_cf = torch.tensor(-1, dtype=torch.int, device=device).view(-1)
                #data.y_cf = ydata_cf
                #adj = from_edge_index_to_adj(data.edge_index, None, data.num_nodes)
                #adj_list.append(adj)
                #max_num_nodes = max(max_num_nodes, data.num_nodes)
                ## append Data object to datalist
                data_list.append(data)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                #if self.pre_transform is not None:
                #    data = self.pre_transform(data)

            #data_list = padded_datalist(data_list, adj_list, max_num_nodes)

        elif self.datatype.lower() == 'nodeopf':
            path = os.path.join(self.raw_dir, 'edge_index_opf.mat')
            edge_index = scipy.io.loadmat(path)
            path = os.path.join(self.raw_dir, 'edge_attr_opf.mat')
            edge_attr = scipy.io.loadmat(path)
            path = os.path.join(self.raw_dir, 'Xopf.mat')
            X = mat73.loadmat(path)
            path = os.path.join(self.raw_dir, f'Y_polar_opf.mat')
            Y = mat73.loadmat(path)
            edge_order = torch.tensor(edge_index['edge_index'].astype(np.int32) - 1, dtype=torch.long).t().contiguous().to(device)
            edge_attr = torch.tensor(edge_attr['edge_attr'], dtype=torch.float)
            edge_attr = torch.nn.functional.normalize(edge_attr, dim=0)
            data_list = []
            fullX = []
            fullY= []

            for i in range(len(X['X'])):
                fullX.append(torch.tensor(X['X'][i][:, [0, 1, 3]], dtype=torch.float, device=device))
                fullY.append(torch.tensor(Y['Y_polar'][i], dtype=torch.float, device=device))
            fullXcat = torch.cat(fullX, dim=0)
            fullYcat = torch.cat(fullY, dim=0)
            maxsX, _ = torch.max(torch.abs(fullXcat), dim=0)
            maxsY, _ = torch.max(torch.abs(fullYcat), dim=0)
            
            for i in range(len(X['X'])):
                N = fullX[i]
                mask = fullY[i] != 0
                N_norm = N/maxsX

                Y_o = fullY[i]
                Y_norm = Y_o /maxsY
                data = Data(x=N_norm, edge_index=edge_order.to(device), y=Y_norm, edge_attr=edge_attr, maxs=maxsY, mask=mask).to(device)
                data_list.append(data)

        else:
            path = os.path.join(self.raw_dir, 'edge_index.mat')
            edge_index = scipy.io.loadmat(path)
            
            path = os.path.join(self.raw_dir, 'edge_attr.mat')
            edge_attr = scipy.io.loadmat(path)
            
            path = os.path.join(self.raw_dir, 'X.mat')
            X = mat73.loadmat(path)
            
            path = os.path.join(self.raw_dir, f'Y_polar.mat')
            Y = mat73.loadmat(path)

            edge_order = torch.tensor(edge_index['edge_index'].astype(np.int32) - 1, dtype=torch.long).t().contiguous().to(device)
            edge_attr = torch.tensor(edge_attr['edge_attr'], dtype=torch.float)
            edge_attr = torch.nn.functional.normalize(edge_attr, dim=0)
            
            data_list = []

            fullX = []
            fullY = []
            for i in range(len(X['Xpf'])):
                fullX.append(torch.tensor(X['Xpf'][i], dtype=torch.float, device=device))  # input features
                fullY.append(torch.tensor(Y['Y_polarpf'][i], dtype=torch.float, device=device))  # target outputs

            # --- Normalization step (commented out) ---
            # fullXcat = torch.cat(fullX, dim=0)
            # fullYcat = torch.cat(fullY, dim=0)
            # maxsX, _ = torch.max(torch.abs(fullXcat), dim=0)
            # maxsY, _ = torch.max(torch.abs(fullYcat), dim=0)

            for i in range(len(X['Xpf'])):
                N = fullX[i]
                Y_o = fullY[i]
                mask = Y_o != 0

                # --- Normalized input/output (commented out) ---
                # N_norm = N / maxsX
                # Y_norm = Y_o / maxsY
                # data = Data(x=N_norm, edge_index=edge_order.to(device), y=Y_norm,
                #             edge_attr=edge_attr, maxs=maxsY, mask=mask).to(device)

                # --- Unnormalized version ---
                data = Data(x=N, edge_index=edge_order.to(device), y=Y_o,
                            edge_attr=edge_attr, maxs=None, mask=mask).to(device)

                data_list.append(data)


        torch.save(self.collate(data_list), self.processed_paths[0])
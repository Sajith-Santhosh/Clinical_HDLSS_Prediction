"""
TOOLBOX MODULE (PURE REGRESSION)
================================
Contains:
- GRACES_Selector: Graph Convolutional Network for Regression (MSELoss)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as gnn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression


from scipy.optimize import linear_sum_assignment
from scipy.stats.qmc import Sobol



# BASE CLASS for feature selection models

class BaseSelector:
    def __init__(self):
        self.selected_indices_ = None
        self.is_fitted_ = False
    
    def fit(self, X, y):
        raise NotImplementedError
    
    def transform(self, X):
        if not self.is_fitted_:
            raise ValueError("Selector not fitted. Call fit() first.")
        if hasattr(X, 'iloc'):
            return X.iloc[:, self.selected_indices_]
        return X[:, self.selected_indices_]


# GRACES embedding along with regression

class GraphConvNet_Regression(nn.Module):
    """Output layer size=1 for continuous regression target."""
    def __init__(self, input_size, hidden_size, alpha):
        super().__init__()
        self.relu = nn.ReLU()
        self.input = nn.Linear(input_size, hidden_size[0], bias=False)
        self.alpha = alpha
        self.hiddens = nn.ModuleList([
            gnn.SAGEConv(hidden_size[h], hidden_size[h+1]) 
            for h in range(len(hidden_size)-1)
        ])
        self.output = nn.Linear(hidden_size[-1], 1)

    def forward(self, x):
        edge_index = self.create_edge_index(x)
        x = self.relu(self.input(x))
        for hidden in self.hiddens:
            x = self.relu(hidden(x, edge_index))
        return self.output(x)
    
    def create_edge_index(self, x):
        sim = torch.abs(F.cosine_similarity(x[..., None, :, :], x[..., :, None, :], dim=-1))
        sim = sim - torch.diag_embed(torch.diag(sim))
        eps = torch.quantile(sim.view(-1), self.alpha, interpolation='nearest')
        row, col = torch.where(sim >= eps)
        return torch.cat((row.reshape(1, -1), col.reshape(1, -1)), dim=0)

class GRACES_Selector(BaseSelector):
    def __init__(self, n_features=50, hidden_size=[64, 32], 
                 epochs=100, batch_size=8, alpha=0.95, lr=0.001):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.lr = lr
        self.scaler = StandardScaler()
        self.S = []

  # In toolbox1.py -> GRACES_Selector -> fit
# REPLACE your existing loop logic with this:

    def fit(self, X, y):
        X_val = X.values if isinstance(X, pd.DataFrame) else X
        y_val = y.values if isinstance(y, pd.Series) else y
        
        # Scale
        X_scaled = self.scaler.fit_transform(X_val)
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=True) # Enable Grad on Input
        y_tensor = torch.tensor(y_val, dtype=torch.float32)
        
        # 1. Train ONE GNN on ALL features (or a large subset)
        print(f"    [GRACES] Training GNN on full feature set to learn importance...")
        model = GraphConvNet_Regression(x_tensor.shape[1], self.hidden_size, self.alpha)        
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            out = model(x_tensor).squeeze()
            loss = loss_fn(out, y_tensor)
            loss.backward()
            optimizer.step()
            
        # 2. Calculate Feature Saliency (Importance)
        # We measure how much changing a feature's input changes the loss
        print(f"    [GRACES] Calculating Saliency Map...")
        saliency = torch.mean(torch.abs(x_tensor.grad), dim=0) # Average grad across all samples
        
        # 3. Select Top K
        top_k_indices = torch.topk(saliency, self.n_features).indices.numpy()
        
        self.selected_indices_ = top_k_indices
        self.is_fitted_ = True
        return self


"""
TOOLBOX MODULE (PURE REGRESSION)
================================
Contains:
- GRACES_Selector: Graph Convolutional Network for Regression (MSELoss)
- DeepFS_Selector: Deep Feature Selection using Supervised Autoencoder
"""


# DEEPFS feature selection along with regression

class SupervisedAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SupervisedAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.ReLU(),
            nn.Linear(100, input_dim)
        )
        self.regression_head = nn.Linear(latent_dim, 1)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        y_hat = self.regression_head(z)
        return z, x_hat, y_hat

class DeepFS_Selector(BaseSelector):
    def __init__(self, n_features=50, latent_dim=15, lr=1e-3, epochs=500, lambda_reg=0.01):
        super().__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.lambda_reg = lambda_reg
        self.feature_scores_ = None
        self.scaler = StandardScaler()

    def _train_autoencoder(self, X, Y):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1).to(device)

        model = SupervisedAutoencoder(X.shape[1], self.latent_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion_recon = nn.MSELoss()
        criterion_sup = nn.MSELoss()

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            z, x_hat, y_hat = model(X_tensor)
            loss_recon = criterion_recon(x_hat, X_tensor)
            loss_sup = criterion_sup(y_hat, Y_tensor)
            loss = loss_sup + self.lambda_reg * loss_recon
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            embeddings = model.encoder(X_tensor).cpu().numpy()
            z_min = embeddings.min(axis=0)
            z_max = embeddings.max(axis=0)
            x_encode = (embeddings - z_min) / (z_max - z_min + 1e-12)
        return x_encode

    def _multivariate_rank(self, X):
        n, p = X.shape
        sobol = Sobol(d=p, scramble=False)
        c_points = sobol.random_base2(m=int(np.ceil(np.log2(n))))
        c_points = c_points[:n]
        cost_matrix = np.linalg.norm(X[:, None, :] - c_points[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        ranks = c_points[col_ind]
        return ranks

    def _rd_cov(self, X_rank, Y_rank):
        n = X_rank.shape[0]
        A = np.linalg.norm(X_rank[:, None, :] - X_rank[None, :, :], axis=2)
        B = np.linalg.norm(Y_rank[:, None, :] - Y_rank[None, :, :], axis=2)
        S1 = np.sum(A * B) / (n * n)
        S2 = np.sum(A) * np.sum(B) / (n ** 4)
        S3 = np.sum(np.sum(A, axis=0) * np.sum(B, axis=0)) / (n ** 3)
        return S1 + S2 - 2 * S3

    def _rd_corr(self, X, Y):
        X_rank = self._multivariate_rank(X)
        Y_rank = self._multivariate_rank(Y)
        cov_xy = self._rd_cov(X_rank, Y_rank)
        cov_xx = self._rd_cov(X_rank, X_rank)
        cov_yy = self._rd_cov(Y_rank, Y_rank)
        return cov_xy / np.sqrt(cov_xx * cov_yy + 1e-12)

    def fit(self, X, y):
        X_val = X.values if isinstance(X, pd.DataFrame) else X
        y_val = y.values if isinstance(y, pd.Series) else y

        print(f"    [DeepFS] Logic: Supervised Autoencoder + RdCorr -> Selecting {self.n_features} features")

        # Scale features
        X_scaled = self.scaler.fit_transform(X_val)

        # Train autoencoder and get latent encoding
        x_encode = self._train_autoencoder(X_scaled, y_val)

        n_features = X_scaled.shape[1]
        omega = np.zeros(n_features)

        # Compute RdCorr between each feature and latent embedding
        print(f"    [DeepFS] Computing RdCorr scores for {n_features} features...")
        for i in range(n_features):
            Xi = X_scaled[:, i].reshape(-1, 1)
            omega[i] = self._rd_corr(Xi, x_encode)

        self.feature_scores_ = omega

        # Select top-k features
        ranked_idx = np.argsort(-omega)
        self.selected_indices_ = ranked_idx[:self.n_features]
        
        self.is_fitted_ = True
        print(f"    [DeepFS] Selected {len(self.selected_indices_)} features")
        print(f"    [DeepFS] Score range: [{omega[self.selected_indices_].min():.4f}, {omega[self.selected_indices_].max():.4f}]")
        
        return self
 
 # Get toolbox for feature selection models
def get_toolbox(n_features=50):
    return {
        "GRACES": GRACES_Selector(n_features=n_features),
        "DEEPFS": DeepFS_Selector(n_features=n_features)
    }

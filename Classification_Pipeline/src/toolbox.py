"""Feature selection models: GRACES and DeepFS."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as gnn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.optimize import linear_sum_assignment
from scipy.stats.qmc import Sobol
import copy


class BaseSelector:
    """Base class for feature selection models."""
    def __init__(self):
        self.selected_indices_ = None
        self.is_fitted_ = False
        self.feature_scores_ = None

    def fit(self, X, y):
        """Fit the selector to data."""
        raise NotImplementedError("Subclasses must implement fit()")

    def transform(self, X):
        """Transform X using selected features."""
        if not self.is_fitted_:
            raise ValueError("Selector not fitted. Call fit() first.")
        if hasattr(X, 'iloc'):
            return X.iloc[:, self.selected_indices_]
        return X[:, self.selected_indices_]

    def fit_transform(self, X, y):
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)


class GraphConvNet_Classification(nn.Module):
    """GraphSAGE-based neural network for GRACES classification."""

    def __init__(self, input_size, output_size, hidden_size, alpha):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.input = nn.Linear(self.input_size, self.hidden_size[0], bias=False)
        self.alpha = alpha
        self.hiddens = nn.ModuleList([
            gnn.SAGEConv(self.hidden_size[h], self.hidden_size[h + 1])
            for h in range(len(self.hidden_size) - 1)
        ])
        self.output = nn.Linear(hidden_size[-1], output_size)

    def forward(self, x):
        edge_index = self.create_edge_index(x)
        x = self.input(x)
        x = self.relu(x)
        for hidden in self.hiddens:
            x = hidden(x, edge_index)
            x = self.relu(x)
        x = self.output(x)
        return self.softmax(x)

    def create_edge_index(self, x):
        """Create edge index based on cosine similarity."""
        similarity_matrix = torch.abs(
            F.cosine_similarity(x[..., None, :, :], x[..., :, None, :], dim=-1)
        )
        similarity = torch.sort(similarity_matrix.view(-1))[0]
        eps = torch.quantile(similarity, self.alpha, interpolation='nearest')
        adj_matrix = similarity_matrix >= eps
        row, col = torch.where(adj_matrix)
        edge_index = torch.cat((row.reshape(1, -1), col.reshape(1, -1)), dim=0)
        return edge_index


class GRACES_Selector(BaseSelector):
    """
    GRACES: GRAph Convolutional nEtwork feature Selector for Classification.

    Based on: Chen et al. "Graph Convolutional Network-based Feature Selection
    for High-dimensional and Low-sample Size Data" (arXiv:2211.14144)
    """

    def __init__(self, n_features=100, hidden_size=None, q=2, n_dropouts=10,
                 dropout_prob=0.5, batch_size=16, learning_rate=0.001,
                 epochs=50, alpha=0.95, sigma=0, f_correct=0):
        super().__init__()
        self.n_features = n_features
        self.q = q
        self.hidden_size = hidden_size if hidden_size is not None else [64, 32]
        self.n_dropouts = n_dropouts
        self.dropout_prob = dropout_prob
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.sigma = sigma
        self.f_correct = f_correct
        self.S = None
        self.new = None
        self.model = None
        self.last_model = None
        self.loss_fn = None
        self.f_scores = None
        self.scaler = StandardScaler()

    @staticmethod
    def bias(x):
        """Add bias term."""
        if not all(x[:, 0] == 1):
            x = torch.cat((torch.ones(x.shape[0], 1), x.float()), dim=1)
        return x

    def f_test(self, x, y):
        """F-test for feature scoring."""
        slc = SelectKBest(f_classif, k=x.shape[1])
        slc.fit(x, y)
        return getattr(slc, 'scores_', np.zeros(x.shape[1]))

    def xavier_initialization(self):
        """Initialize weights using Xavier initialization."""
        if self.last_model is not None:
            weight = torch.zeros(self.hidden_size[0], len(self.S))
            nn.init.xavier_normal_(weight, gain=nn.init.calculate_gain('relu'))
            old_s = self.S.copy()
            if self.new in old_s:
                old_s.remove(self.new)
            for i in self.S:
                if i != self.new:
                    weight[:, self.S.index(i)] = self.last_model.input.weight.data[:, old_s.index(i)]
            self.model.input.weight.data = weight
            for h in range(len(self.hidden_size) - 1):
                self.model.hiddens[h].lin_l.weight.data = self.last_model.hiddens[h].lin_l.weight.data
                self.model.hiddens[h].lin_r.weight.data = self.last_model.hiddens[h].lin_r.weight.data
            self.model.output.weight.data = self.last_model.output.weight.data

    def train(self, x, y):
        """Train the GNN model."""
        input_size = len(self.S)
        output_size = len(torch.unique(y))
        self.model = GraphConvNet_Classification(input_size, output_size, self.hidden_size, self.alpha)
        self.xavier_initialization()
        x = x[:, self.S]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_set = [[x[i, :], y[i]] for i in range(x.shape[0])]
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True
        )

        for e in range(self.epochs):
            for data, label in train_loader:
                input_0 = data.view(data.shape[0], -1)
                optimizer.zero_grad()
                output = self.model(input_0.float())
                loss = self.loss_fn(output, label)
                loss.backward()
                optimizer.step()

        self.last_model = copy.deepcopy(self.model)

    def dropout(self):
        """Apply dropout to model."""
        model_dp = copy.deepcopy(self.model)
        for h in range(len(self.hidden_size) - 1):
            h_size = self.hidden_size[h]
            dropout_index = np.random.choice(
                range(h_size), int(h_size * self.dropout_prob), replace=False
            )
            model_dp.hiddens[h].lin_l.weight.data[:, dropout_index] = torch.zeros(
                model_dp.hiddens[h].lin_l.weight[:, dropout_index].shape
            )
            model_dp.hiddens[h].lin_r.weight.data[:, dropout_index] = torch.zeros(
                model_dp.hiddens[h].lin_r.weight[:, dropout_index].shape
            )
        dropout_index = np.random.choice(
            range(self.hidden_size[-1]),
            int(self.hidden_size[-1] * self.dropout_prob),
            replace=False
        )
        model_dp.output.weight.data[:, dropout_index] = torch.zeros(
            model_dp.output.weight.data[:, dropout_index].shape
        )
        return model_dp

    def gradient(self, x, y, model):
        """Compute gradients for feature importance."""
        model_gr = GraphConvNet_Classification(
            x.shape[1], len(torch.unique(y)), self.hidden_size, self.alpha
        )
        temp = torch.zeros(model_gr.input.weight.shape)
        temp[:, self.S] = model.input.weight
        model_gr.input.weight.data = temp

        for h in range(len(self.hidden_size) - 1):
            model_gr.hiddens[h].lin_l.weight.data = (
                model.hiddens[h].lin_l.weight +
                self.sigma * torch.randn(model.hiddens[h].lin_l.weight.shape)
            )
            model_gr.hiddens[h].lin_r.weight.data = (
                model.hiddens[h].lin_r.weight +
                self.sigma * torch.randn(model.hiddens[h].lin_r.weight.shape)
            )
        model_gr.output.weight.data = model.output.weight

        output_gr = model_gr(x.float())
        loss_gr = self.loss_fn(output_gr, y)
        loss_gr.backward()
        input_gradient = model_gr.input.weight.grad
        return input_gradient

    def average(self, x, y, n_average):
        """Average gradients over multiple dropouts."""
        grad_cache = None
        for num in range(n_average):
            model = self.dropout()
            input_grad = self.gradient(x, y, model)
            if grad_cache is None:
                grad_cache = input_grad
            else:
                grad_cache += input_grad
        return grad_cache / n_average

    def find(self, input_gradient):
        """Find next best feature."""
        gradient_norm = input_gradient.norm(p=self.q, dim=0)
        gradient_norm = gradient_norm / gradient_norm.norm(p=2)
        gradient_norm[1:] = (
            (1 - self.f_correct) * gradient_norm[1:] +
            self.f_correct * self.f_scores
        )
        gradient_norm[self.S] = 0
        max_index = torch.argmax(gradient_norm)
        return max_index.item()

    def fit(self, X, y):
        """
        Fit the GRACES selector.

        Parameters:
        -----------
        X : array-like or DataFrame
            Feature matrix
        y : array-like or Series
            Target labels
        """
        X_val = X.values if isinstance(X, pd.DataFrame) else X
        y_val = y.values if isinstance(y, pd.Series) else y

        # Scale features
        X_scaled = self.scaler.fit_transform(X_val)

        x = torch.tensor(X_scaled)
        y_tensor = torch.tensor(y_val)

        self.f_scores = torch.tensor(self.f_test(x, y_tensor))
        self.f_scores[torch.isnan(self.f_scores)] = 0
        if torch.norm(self.f_scores) > 0:
            self.f_scores = self.f_scores / self.f_scores.norm(p=2)

        x = self.bias(x)
        self.S = [0]
        self.loss_fn = nn.CrossEntropyLoss()

        print(f"    [GRACES] Selecting {self.n_features} features...")

        while len(self.S) < self.n_features + 1:
            self.train(x, y_tensor)
            input_gradient = self.average(x, y_tensor, self.n_dropouts)
            self.new = self.find(input_gradient)
            self.S.append(self.new)

            if len(self.S) % 10 == 0:
                print(f"      Selected {len(self.S) - 1}/{self.n_features} features")

        selection = self.S.copy()
        selection.remove(0)
        selection = [s - 1 for s in selection]

        self.selected_indices_ = np.array(selection)
        self.is_fitted_ = True

        print(f"    [GRACES] Feature selection complete.")
        return self


class SupervisedAutoencoder(nn.Module):
    """Supervised Autoencoder for DeepFS."""

    def __init__(self, input_dim, latent_dim, n_classes):
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
        self.classification_head = nn.Linear(latent_dim, n_classes)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        y_hat = self.classification_head(z)
        return z, x_hat, y_hat


class DeepFS_Selector(BaseSelector):
    """
    DeepFS: Deep Feature Screening for Classification.

    Based on: Li et al. "Deep Feature Screening: Feature Selection for
    Ultra High-Dimensional Data via Deep Neural Networks" (arXiv:2204.01682)
    """

    def __init__(self, n_features=100, latent_dim=15, lr=1e-3, epochs=500,
                 lambda_reg=0.01):
        super().__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.lambda_reg = lambda_reg
        self.scaler = StandardScaler()

    def _train_autoencoder(self, X, Y):
        """Train the supervised autoencoder."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        Y_tensor = torch.tensor(Y, dtype=torch.long).to(device)

        n_classes = len(np.unique(Y))
        model = SupervisedAutoencoder(X.shape[1], self.latent_dim, n_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion_recon = nn.MSELoss()
        criterion_sup = nn.CrossEntropyLoss()

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
        """Compute empirical multivariate rank using Sobol sequence."""
        n, p = X.shape
        sobol = Sobol(d=p, scramble=False)
        c_points = sobol.random_base2(m=int(np.ceil(np.log2(n))))
        c_points = c_points[:n]
        cost_matrix = np.linalg.norm(X[:, None, :] - c_points[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        ranks = c_points[col_ind]
        return ranks

    def _rd_cov(self, X_rank, Y_rank):
        """Compute rank distance covariance."""
        n = X_rank.shape[0]
        A = np.linalg.norm(X_rank[:, None, :] - X_rank[None, :, :], axis=2)
        B = np.linalg.norm(Y_rank[:, None, :] - Y_rank[None, :, :], axis=2)
        S1 = np.sum(A * B) / (n * n)
        S2 = np.sum(A) * np.sum(B) / (n ** 4)
        S3 = np.sum(np.sum(A, axis=0) * np.sum(B, axis=0)) / (n ** 3)
        return S1 + S2 - 2 * S3

    def _rd_corr(self, X, Y):
        """Compute rank distance correlation."""
        X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)
        Y_normalized = (Y - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)

        X_rank = self._multivariate_rank(X_normalized)
        Y_rank = self._multivariate_rank(Y_normalized)

        cov_xy = self._rd_cov(X_rank, Y_rank)
        cov_xx = self._rd_cov(X_rank, X_rank)
        cov_yy = self._rd_cov(Y_rank, Y_rank)

        if cov_xx * cov_yy == 0:
            return 0
        return cov_xy / np.sqrt(cov_xx * cov_yy)

    def fit(self, X, y):
        """
        Fit the DeepFS selector.

        Parameters:
        -----------
        X : array-like or DataFrame
            Feature matrix
        y : array-like or Series
            Target labels
        """
        X_val = X.values if isinstance(X, pd.DataFrame) else X
        y_val = y.values if isinstance(y, pd.Series) else y

        print(f"    [DeepFS] Supervised Autoencoder + RdCorr -> {self.n_features} features")

        # Scale features
        X_scaled = self.scaler.fit_transform(X_val)

        # Train autoencoder and get latent encoding
        print(f"    [DeepFS] Training autoencoder...")
        x_encode = self._train_autoencoder(X_scaled, y_val)

        n_total_features = X_scaled.shape[1]
        omega = np.zeros(n_total_features)

        # Compute RdCorr between each feature and latent embedding
        print(f"    [DeepFS] Computing RdCorr scores...")
        for i in range(n_total_features):
            Xi = X_scaled[:, i].reshape(-1, 1)
            omega[i] = self._rd_corr(Xi, x_encode)
            if (i + 1) % 50 == 0:
                print(f"      Processed {i+1}/{n_total_features} features")

        self.feature_scores_ = omega

        # Select top-k features
        sorted_idx = np.argsort(-omega)
        self.selected_indices_ = sorted_idx[:self.n_features]

        self.is_fitted_ = True
        print(f"    [DeepFS] Selected {len(self.selected_indices_)} features")
        print(f"    [DeepFS] Score range: [{omega[self.selected_indices_].min():.4f}, {omega[self.selected_indices_].max():.4f}]")

        return self


class TabPFNEmbeddingSelector(BaseSelector):
    """
    Feature selection using TabPFN embeddings.
    Selects features based on importance derived from TabPFN's internal representations.
    """

    def __init__(self, n_features=100, n_fold=5):
        super().__init__()
        self.n_features = n_features
        self.n_fold = n_fold
        self.embedder = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """
        Fit using TabPFN embeddings (if tabpfn_extensions available).

        Note: This requires tabpfn_extensions package.
        """
        try:
            from tabpfn import TabPFNClassifier
            from tabpfn_extensions.embedding import TabPFNEmbedding
        except ImportError:
            print("    [TabPFNEmbedding] tabpfn_extensions not available. Using F-test fallback.")
            # Fallback to F-test
            slc = SelectKBest(f_classif, k=self.n_features)
            X_val = X.values if isinstance(X, pd.DataFrame) else X
            y_val = y.values if isinstance(y, pd.Series) else y
            slc.fit(X_val, y_val)
            self.selected_indices_ = slc.get_support(indices=True)
            self.is_fitted_ = True
            return self

        print(f"    [TabPFNEmbedding] Computing embeddings for selection...")

        X_val = X.values if isinstance(X, pd.DataFrame) else X
        y_val = y.values if isinstance(y, pd.Series) else y

        # Train TabPFN and get embeddings
        tabpfn_clf = TabPFNClassifier(device='cpu', n_estimators=32)
        tabpfn_clf.fit(X_val, y_val)

        self.embedder = TabPFNEmbedding(tabpfn_clf=tabpfn_clf, n_fold=self.n_fold)

        # Get embeddings for training data
        embeddings = self.embedder.get_embeddings(X_val, y_val, X_val, data_source='train')[0]

        # Select features based on correlation with embeddings
        n_features = X_val.shape[1]
        scores = np.zeros(n_features)

        for i in range(n_features):
            corr = np.abs(np.corrcoef(X_val[:, i], embeddings[:, 0])[0, 1])
            scores[i] = corr if not np.isnan(corr) else 0

        self.feature_scores_ = scores
        self.selected_indices_ = np.argsort(-scores)[:self.n_features]
        self.is_fitted_ = True

        print(f"    [TabPFNEmbedding] Selected {len(self.selected_indices_)} features")
        return self


def get_toolbox(n_features=100, selection_method='all'):
    """
    Get a dictionary of feature selectors.

    Parameters:
    -----------
    n_features : int
        Number of features to select
    selection_method : str or list
        Which selectors to return ('all', 'graces', 'deepfs', 'tabpfn_emb')

    Returns:
    --------
    dict : Dictionary of selector instances
    """
    toolbox = {}

    if selection_method == 'all' or 'graces' in selection_method:
        toolbox["GRACES"] = GRACES_Selector(n_features=n_features)

    if selection_method == 'all' or 'deepfs' in selection_method:
        toolbox["DEEPFS"] = DeepFS_Selector(n_features=n_features)

    if selection_method == 'all' or 'tabpfn_emb' in selection_method:
        toolbox["TABPFN_EMB"] = TabPFNEmbeddingSelector(n_features=n_features)

    return toolbox

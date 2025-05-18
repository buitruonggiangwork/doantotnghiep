import os
import torch
import torch.nn as nn
import torchaudio
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import joblib
import numpy as np
from model_lstm import LSTMFakeDetectionModel
from model_rnn import RNNFakeDetectionModel
from data_utils import SceneFakeDataset
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from data_utils import load_or_extract_features


class ModelDriver:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def train(self):
        raise NotImplementedError
    def evaluate(self):
        raise NotImplementedError


class LSTMDriver(ModelDriver):
    def __init__(self, model_path="models/model_lstm.pth"):
        super().__init__(model_path)
        self.model = LSTMFakeDetectionModel().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = DataLoader(SceneFakeDataset("ScenceFake/train"), batch_size=1024, shuffle=True)
        self.eval_loader = DataLoader(SceneFakeDataset("ScenceFake/eval"), batch_size=1024)
    def _load_checkpoint(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            return checkpoint["epoch"] + 1
        return 0
    def train(self, num_epochs=10):
        start_epoch = self._load_checkpoint()
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.model.train()
            total_loss = 0
            progress = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
            for waveforms, labels in progress:
                waveforms, labels = waveforms.to(self.device), labels.to(self.device)
                outputs = self.model(waveforms)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                avg_loss = total_loss / (progress.n + 1)
                progress.set_postfix(loss=f"{avg_loss:.4f}")
            torch.save({
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict()
            }, self.model_path)


    def evaluate(self):
        self._load_checkpoint()
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for waveforms, labels in tqdm(self.eval_loader, desc="Evaluating"):
                waveforms = waveforms.to(self.device)
                outputs = self.model(waveforms)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        print(classification_report(all_labels, all_preds, digits=4))


class RNNDriver(LSTMDriver):
    def __init__(self, model_path="models/model_rnn.pth"):
        ModelDriver.__init__(self, model_path)
        self.model = RNNFakeDetectionModel().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = DataLoader(SceneFakeDataset("ScenceFake/train"), batch_size=1024, shuffle=True)
        self.eval_loader = DataLoader(SceneFakeDataset("ScenceFake/eval"), batch_size=1024)


    def _load_checkpoint(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            return checkpoint["epoch"] + 1
        return 0


class SVMDriver(ModelDriver):
    def __init__(self, model_path="models/model_svm.pkl"):
        super().__init__(model_path)
        self.X_train, self.y_train = load_or_extract_features("ScenceFake/train", "features/features_train.npz")
        self.X_eval, self.y_eval = load_or_extract_features("ScenceFake/eval", "features/features_eval.npz")
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_eval = scaler.transform(self.X_eval)
        self.model = None


    def _load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)


    def train(self):
        self._load_model()
        print("Tuning SVM vi GridSearchCV...")
        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"]
        }
        grid = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, verbose=1)
        grid.fit(self.X_train, self.y_train)
        print("Best params:", grid.best_params_)
        self.model = grid.best_estimator_
        joblib.dump(self.model, self.model_path)
    def evaluate(self):
        self._load_model()
        y_pred = self.model.predict(self.X_eval)
        print(classification_report(self.y_eval, y_pred, digits=4))


class RandomForestDriver(ModelDriver):
    def __init__(self, model_path="models/model_rf.pkl"):
        super().__init__(model_path)
        self.X_train, self.y_train = load_or_extract_features("ScenceFake/train", "features/features_train.npz")
        self.X_eval, self.y_eval = load_or_extract_features("ScenceFake/eval", "features/features_eval.npz")
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_eval = scaler.transform(self.X_eval)
        self.model = None


    def _load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)


    def train(self):
        self._load_model()
        print("Tuning Random Forest vi GridSearchCV...")
        param_grid = {
            "n_estimators": [100, 200, 500],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
        grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1, verbose=1)
        grid.fit(self.X_train, self.y_train)
        print("Best params:", grid.best_params_)
        self.model = grid.best_estimator_
        joblib.dump(self.model, self.model_path)


    def evaluate(self):
        self._load_model()
        y_pred = self.model.predict(self.X_eval)
        print(classification_report(self.y_eval, y_pred, digits=4))
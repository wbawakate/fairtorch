from pathlib import Path
import os
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from fairtorch import ConstraintLoss, DemographicParityLoss, EqualiedOddsLoss
from sklearn.metrics import accuracy_score, roc_auc_score


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(2020)


class DatasetGenerator:
    def __clean_up_data(self, df):
        # use_columns = ["gender", "lsat", "pass_bar", "race1", "ugpa", "fulltime"]
        use_columns = [
            "decile1b",
            "decile3",
            "decile1",
            "cluster",
            "lsat",
            "ugpa",
            "zfygpa",
            "DOB_yr",
            "grad",
            "zgpa",
            "fulltime",
            "fam_inc",
            "age",
            "gender",
            "parttime",
            "male",
            "race1",
            "Dropout",
            "pass_bar",
            "tier",
            "index6040",
        ]
        df.loc[:, "race1"] = df.loc[:, "race1"].astype(str)
        df.loc[:, "race1"] = df.loc[:, "race1"].where(df.loc[:, "race1"] == "white", 0)
        df.loc[:, "race1"] = df.loc[:, "race1"].where(df.loc[:, "race1"] != "white", 1)
        df.loc[:, "race1"] = df.loc[:, "race1"].astype(int)

        categorical_cols = ["grad", "gender", "Dropout"]
        df = df.dropna()
        for col in use_columns:
            if col not in categorical_cols:
                df.loc[:, col] = df.loc[:, col].astype(float)

        df.loc[:, "gender"] = df.loc[:, "gender"].astype(str)

        df = df[use_columns]
        df.loc[:, categorical_cols] = df.loc[:, categorical_cols].apply(
            LabelEncoder().fit_transform
        )

        return df.reset_index(drop=True)

    def generate_dataset(self, dataset_csv_path=Path("./examples/inputs/bar_pass_prediction.csv")):
        df = pd.read_csv(dataset_csv_path)
        df = self.__clean_up_data(df)
        return df


class BarPassDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, sensitive_feature, transform=None):
        self.transform = transform
        self.x = x
        self.y = y
        self.sensitive_feature = sensitive_feature

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x_i = self.x[idx]
        y_i = self.y[idx]
        sensitive_feature_i = self.sensitive_feature[idx]

        if self.transform:
            x_i = self.transform(x_i)

        return x_i, y_i, sensitive_feature_i


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {value" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        train_dataloader,
        valid_dataloader,
        fairness_constraint=None,
        max_epoch=100,
        use_fairness_penalty=True,
        metrics="valid_auc",
        metrics_direction="max",
        early_stopping=True,
        early_stopping_patience=10,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.fairness_constraint = fairness_constraint.to(self.device)
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.max_epoch = max_epoch
        self.use_fairness_penalty = use_fairness_penalty
        self.metrics = metrics
        self.metrics_direction = metrics_direction
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience

    def training_step(self, epoch):
        train_loss_epoch = AverageMeter("train_loss", ":.4e")
        penalty_epoch = AverageMeter("train_fairness_penalty", ":.4e")
        y_list = []
        prediction_list = []
        self.model.train()
        for batch_idx, (x, y, sensitive_feature) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            logit = self.model(x.to(self.device))
            loss = self.criterion(logit.view(-1), y.to(self.device))
            if self.fairness_constraint:
                penalty = self.fairness_constraint(
                    x, logit.view(-1), sensitive_feature.to(self.device), y.to(self.device)
                )
            else:
                penalty = 0

            if self.use_fairness_penalty:
                loss = loss + penalty

            prediction = torch.sigmoid(logit)

            loss.backward()
            self.optimizer.step()
            train_loss_epoch.update(loss.item(), x.size(0))
            penalty_epoch.update(penalty.item(), x.size(0))
            y_list.append(y.detach().cpu())
            prediction_list.append(prediction[:, 0])

        y = torch.cat(y_list)
        prediction = torch.cat(prediction_list)
        train_acc_epoch = accuracy_score(
            y.detach().cpu().numpy(), (prediction >= 0.5).detach().cpu()
        )
        train_auc_epoch = roc_auc_score(y.detach().cpu().numpy(), prediction.detach().cpu())
        result = {
            "epoch": epoch,
            "train_loss_epoch": train_loss_epoch.value,
            "train_acc_epoch": train_acc_epoch,
            "train_auc_epoch": train_auc_epoch,
            "train_fairness_penalty": penalty_epoch.value,
        }
        print(result)
        return result

    def validation_step(self, epoch):
        valid_loss = AverageMeter("validation_loss", ":.4e")
        penalty_epoch = AverageMeter("valid_fairness_penalty", ":.4e")

        y_list = []
        prediction_list = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (x, y, sensitive_feature) in enumerate(self.valid_dataloader):
                logit = self.model(x.to(self.device))
                loss = self.criterion(logit.view(-1), y.to(self.device))

                if self.fairness_constraint:
                    penalty = self.fairness_constraint(
                        x, logit.view(-1), sensitive_feature.to(self.device), y.to(self.device),
                    )
                else:
                    penalty = 0

                if self.use_fairness_penalty:
                    loss = loss + penalty

                valid_loss.update(loss.item(), x.size(0))
                penalty_epoch.update(penalty.item(), x.size(0))

                prediction = torch.sigmoid(logit)
                y_list.append(y.detach().cpu())
                prediction_list.append(prediction[:, 0])

        prediction = torch.cat(prediction_list)
        y = torch.cat(y_list)
        valid_acc = accuracy_score(y.detach().cpu().numpy(), (prediction >= 0.5).detach().cpu())
        valid_auc = roc_auc_score(y.detach().cpu().numpy(), prediction.detach().cpu())

        result = {
            "epoch": epoch,
            "valid_loss": valid_loss.value,
            "valid_acc": valid_acc,
            "valid_auc": valid_auc,
            "valid_fairness_penalty": penalty_epoch.value,
        }
        print(result)
        return result

    def fit(self):
        if self.metrics_direction == "max":
            metrics_best = -np.inf
        else:
            metrics_best = np.inf
        train_result_best = {}
        valid_result_best = {}

        no_improvement = 0
        for epoch in range(self.max_epoch):
            train_result = self.training_step(epoch)
            valid_result = self.validation_step(epoch)
            if self.early_stopping:
                if self.metrics_direction == "max":
                    if metrics_best < valid_result[self.metrics]:
                        metrics_best = valid_result[self.metrics]
                        train_result_best = train_result
                        valid_result_best = valid_result
                    else:
                        no_improvement += 1
                else:
                    if metrics_best > valid_result[self.metrics]:
                        metrics_best = valid_result[self.metrics]
                        train_result_best = train_result
                        valid_result_best = valid_result
                    else:
                        no_improvement += 1
                if no_improvement > self.early_stopping_patience:
                    break

        return metrics_best, train_result_best, valid_result_best


def get_dataloader(
    df, train_index, val_index, label="pass_bar", sensitive_feature_elements="gender"
):
    drop_elements = [label]

    x_train = df.drop(drop_elements, axis=1).loc[train_index]
    y_train = df.loc[train_index, label]
    sensitive_feature_train = df.loc[train_index, sensitive_feature_elements]

    x_valid = df.drop(drop_elements, axis=1).loc[val_index]
    y_valid = df.loc[val_index, label]
    sensitive_feature_valid = df.loc[val_index, sensitive_feature_elements]

    x_train = torch.from_numpy(x_train.values).float()
    y_train = torch.from_numpy(y_train.values).float()
    sensitive_feature_train = torch.from_numpy(sensitive_feature_train.values).float()
    train_dataset = BarPassDataset(x=x_train, y=y_train, sensitive_feature=sensitive_feature_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    x_valid = torch.from_numpy(x_valid.values).float()
    y_valid = torch.from_numpy(y_valid.values).float()
    sensitive_feature_valid = torch.from_numpy(sensitive_feature_valid.values).float()
    valid_dataset = BarPassDataset(x=x_valid, y=y_valid, sensitive_feature=sensitive_feature_valid)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False)
    return train_dataloader, valid_dataloader


if __name__ == "__main__":
    label = "pass_bar"
    sensitive_feature_elements = "race1"
    # sensitive_feature_elements = "gender"

    data_generator = DatasetGenerator()
    df = data_generator.generate_dataset()
    metric_list = []
    penalty_list = []
    skf = StratifiedKFold(n_splits=5)
    for fold, (train_index, val_index) in enumerate(skf.split(df, df["pass_bar"])):
        train_dataloader, valid_dataloader = get_dataloader(
            df=df,
            train_index=train_index,
            val_index=val_index,
            label=label,
            sensitive_feature_elements=sensitive_feature_elements,
        )

        feature_num = df.drop(label, axis=1).shape[1]
        model = nn.Sequential(
            nn.Linear(feature_num, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1),
        )
        # model = nn.Sequential(
        #     nn.Linear(feature_num, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Linear(128, 1),
        # )

        criterion = nn.BCEWithLogitsLoss(
            # pos_weight=torch.tensor((df.pass_bar == 0).sum() / (df.pass_bar == 1).sum())
            pos_weight=None
        )
        fairness_constraint = DemographicParityLoss(
            alpha=100,
            sensitive_classes=df[sensitive_feature_elements].unique().astype(int).tolist(),
        )
        optimizer = optim.Adam(model.parameters())

        trainer = Trainer(
            model=model,
            criterion=criterion,
            fairness_constraint=fairness_constraint,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            use_fairness_penalty=True,
        )
        metrics_best, train_result_best, valid_result_best = trainer.fit()
        metric_list.append(metrics_best)
        penalty_list.append(valid_result_best["valid_fairness_penalty"])
        print(f"fold {fold}: metrics_best: {metrics_best}")
        print(
            f"fold {fold}: valid_fairness_penalty: {valid_result_best['valid_fairness_penalty']}"
        )
    print(f"metrics fold {metric_list}")
    print(f"metrics CV mean {np.mean(metric_list)}")
    print(f"penalty_list fold {penalty_list}")
    print(f"penalty_list CV mean {np.mean(penalty_list)}")

import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data.dataset import Subset
import torch.nn.functional as F
import sklearn.metrics
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
import datetime
import pathlib
import pickle
import json
import argparse

parser = argparse.ArgumentParser()

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

# データセットを読み込む
def get_datasets():
  train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
  )
  test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
  )

  train_data.targets[train_data.targets < 7] = 0
  train_data.targets[7 <= train_data.targets] = 1
  test_data.targets[test_data.targets < 7] = 0
  test_data.targets[7 <= test_data.targets] = 1

  return train_data, test_data

# ユーザーごとにデータセットを分割する
def split_by_user(user_count, train_data):
  res = {}
  unselected_indexes = [i for i in range(len(train_data))]
  data_count = int(len(train_data) / user_count)

  for i in range(0, user_count):
    res[i] = set(np.random.choice(unselected_indexes, data_count, replace=False))
    unselected_indexes = list(set(unselected_indexes) - res[i])

  return res

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        #return torch.tensor(image), torch.tensor(label)
        return image, label

def get_indices_with_label(data, label):
  """与えられたラベルを持つ/持たないデータのindexの配列を求める
  """
  # マスクを取得する ex) [True, False, ...]
  label_mask = data.targets == label
  non_label_mask = data.targets != label

  # マスクがTrueであるindexを配列として取得する
  label_indices = [i for (i, value) in enumerate(label_mask) if value]
  non_label_indices = [i for (i, value) in enumerate(non_label_mask) if value]

  return label_indices, non_label_indices

def get_user_indices(data_allocation, normal_idxs, abnormal_idxs):
  """与えられたデータのindexの割り当てを計算する
  
  与えられたデータのindexを、data_allocationの割合で各ユーザーに割り当てる。（データのindexを全て利用するとは限らない）

  Args:
    data_allocation (list[(int, float)]): (そのユーザが持つデータ総数, 含まれる異常値の割合)の配列
    normal_idxs: 通常値のindexの配列
    abnormal_idxs: 異常値のindexの配列
    data_count_per_user: ユーザー1人当たりのデータ数
  
  Returns:
    list[list[int]]: 各ユーザーごとのindexの配列（[[1,2,4], [3,5,6], ...]）
  """
  normal_cnt = len(normal_idxs)
  abnormal_cnt = len(abnormal_idxs)

  # データが足りるかチェックする
  total_sum = 0
  abnormal_sum = 0
  for (total, rate) in data_allocation:
    total_sum += total
    abnormal_sum += int(total * rate)

  if normal_cnt + abnormal_cnt < total_sum or abnormal_cnt < abnormal_sum:
      print(f"┌────────────────────────────────────────────────────────────────────────────")
      print(f"│   ⚠️ ERROR: invalid data allocation...!!!")
      print(f"├────────────────────────────────────────────────────────────────────────────")
      print(f"│   total_cnt={normal_cnt + abnormal_cnt}, total_sum={total_sum}")
      print(f"│   abnormal_cnt={abnormal_cnt}, abnormal_sum={abnormal_sum}")
      print(f"└────────────────────────────────────────────────────────────────────────────")
      raise

  # 配分する
  normal_offset = 0
  abnormal_offset = 0
  user_idxs_list = []
  for (total, rate) in data_allocation:
    # 異常/通常値のデータ数
    abnormal_cnt = int(total * rate)
    normal_cnt = total - abnormal_cnt

    # データの追加
    user_idxs = normal_idxs[normal_offset: normal_offset + normal_cnt]
    user_idxs.extend(abnormal_idxs[abnormal_offset: abnormal_offset + abnormal_cnt])

    # シャッフルする
    np.random.shuffle(user_idxs)
    user_idxs_list.append(user_idxs)

    # offsetの更新
    normal_offset += normal_cnt
    abnormal_offset += abnormal_cnt

  return user_idxs_list

# これ、　ユーザ1が他のユーザのテストデータを持っていることにならない？
def get_user_indices_2(data_allocation, normal_idxs, abnormal_idxs):
  """与えられたデータのindexの割り当てを計算する

  2人目以降のユーザーは1人目のユーザーとデータを共有する

  Args:
    data_allocation (list[(int, float, float)]): (そのユーザが持つデータ総数, 含まれる異常値の割合、1人目のユーザーと共有する異常値の割合)の配列
    normal_idxs: 通常値のindexの配列
    abnormal_idxs: 異常値のindexの配列
    data_count_per_user: ユーザー1人当たりのデータ数
  
  Returns:
    list[list[int]]: 各ユーザーごとのindexの配列（[[1,2,4], [3,5,6], ...]）
  """
  print("share mode")

  normal_cnt = len(normal_idxs)
  abnormal_cnt = len(abnormal_idxs)

  # データが足りるかチェックする
  total_sum = 0
  abnormal_sum = 0
  abnormal_share_sum = 0
  for (total, rate, share_rate) in data_allocation:
    total_sum += total
    abnormal_sum += int(total * rate * (1 - share_rate))
    abnormal_share_sum += int(total * rate * share_rate)

  if normal_cnt + abnormal_cnt < total_sum or abnormal_cnt < abnormal_sum or data_allocation[0][0] * data_allocation[0][1] < abnormal_share_sum:
      print(f"┌────────────────────────────────────────────────────────────────────────────")
      print(f"│   ⚠️ ERROR: invalid data allocation...!!!")
      print(f"├────────────────────────────────────────────────────────────────────────────")
      print(f"│   total_cnt={normal_cnt + abnormal_cnt}, total_sum={total_sum}")
      print(f"│   abnormal_cnt={abnormal_cnt}, abnormal_sum={abnormal_sum}")
      print(f"│   abnormal_share_cnt={data_allocation[0][0] * data_allocation[0][1]}, abnormal_share_sum={abnormal_share_sum}")
      print(f"└────────────────────────────────────────────────────────────────────────────")
      print(data_allocation)
  


  # 配分する
  normal_offset = 0
  abnormal_offset = 0
  abnormal_share_offset = 0
  user_idxs_list = []
  for j, (total, rate, share_rate) in enumerate(data_allocation):
    # 異常/通常値のデータ数
    abnormal_share_cnt = int(total * rate * share_rate)
    abnormal_original_cnt = int(total * rate * (1.0 - share_rate))
    print(f"user: {j}, 異常値シェア数: {abnormal_share_cnt}, 残り: {abnormal_original_cnt}")
    normal_cnt = total - abnormal_share_cnt - abnormal_original_cnt

    # データの追加
    user_idxs = normal_idxs[normal_offset: normal_offset + normal_cnt]
    user_idxs.extend(abnormal_idxs[abnormal_offset: abnormal_offset + abnormal_original_cnt])
    user_idxs.extend(abnormal_idxs[abnormal_share_offset: abnormal_share_offset + abnormal_share_cnt])

    # シャッフルする
    np.random.shuffle(user_idxs)
    user_idxs_list.append(user_idxs)

    # offsetの更新
    normal_offset += normal_cnt
    abnormal_offset += abnormal_original_cnt
    abnormal_share_offset += abnormal_share_cnt

  return user_idxs_list

def calc_evaluation_score(test_x, test_y, model):
    result = {}
    
    prob = model(test_x)
    prob = np.exp(prob[:, (0,1)].cpu().detach().numpy())
    
    threshold = 0.5
    pred = [int(v) for v in (prob[:, 1] >= threshold)]

    result["precision"] = sklearn.metrics.precision_score(test_y, pred, zero_division=0)
    result["recall"] = sklearn.metrics.recall_score(test_y, pred, zero_division=0)
    result["accuracy"] = sklearn.metrics.accuracy_score(test_y, pred)
    result["f1"] = sklearn.metrics.f1_score(test_y, pred, zero_division=0)
    result["auroc"] = sklearn.metrics.roc_auc_score(test_y, prob[:, 1])
    result["neg_log_loss"] = sklearn.metrics.log_loss(test_y, prob[:, 1])
    
    return result

from opacus import PrivacyEngine
privacy_engine = PrivacyEngine()

class UserHyperParams:
  def __init__(self, batch_size: int, epoch: int, log_per: int, noise_multiplier: float):
    self.batch_size = batch_size
    self.epoch = epoch
    self.log_per = log_per
    #self.test_batch_size = test_batch_size
    print("warning: test_batch_size is decrepted")
    self.noise_multiplier = noise_multiplier

class User:
  def __init__(self, id, data, idxs, params):
    self.id = id
    self.params = params
    self.idxs = idxs
    print(f"user id: {id}")
    self.train_loader, self.test_loader = self.get_data_loaders(data, idxs)
    # self.train_loader = self.get_data_loaders(data, idxs)
    

  def get_data_loaders(self, data, idxs):
    # train_idxs = idxs
    train_idxs = idxs[:int(len(idxs) * 0.8)]
    test_idxs = idxs[int(len(idxs) * 0.8):]
    print(f"num train data:\t{len(train_idxs)}, num test data:\t({len(test_idxs)})")

    train_loader = DataLoader(
      DatasetSplit(data, train_idxs),
      batch_size=self.params.batch_size,
      shuffle=True
    )

    test_loader = DataLoader(
      DatasetSplit(data, test_idxs),
      #batch_size=self.params.test_batch_size,
      batch_size = int(len(idxs) * 0.8),
      shuffle=True
    )

    # return train_loader
    return train_loader, test_loader
    

  def train(self, model):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print("noise\t", self.params.noise_multiplier)

    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=self.train_loader,
        noise_multiplier=self.params.noise_multiplier,
        max_grad_norm=1,
    )

    # else:
    #   train_loader = self.train_loader
    
    epoch_loss = []

    for epoch in range(self.params.epoch):
      # print(f"┌────────────────────────────────────────────────────────────────────────────")
      # print(f"│   [u{self.id}] local epoch{epoch}")
      # print(f"├────────────────────────────────────────────────────────────────────────────")
      batch_loss = []
      print("local epoch:", epoch)
      # train loop
      for batch, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = F.nll_loss(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % self.params.log_per == 0:
          loss = loss.item()
          current = batch * len(X)
          # print(f"│   loss: {loss:>7f} current: {current:>5d}")

        batch_loss.append(loss)
      epoch_loss.append(sum(batch_loss) / len(batch_loss))
      # print(f"└────────────────────────────────────────────────────────────────────────────")

    return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

  def test(self, model):
    for X, y in self.test_loader:
      X = X.to(device)

    print(f"test_data_size:\t{len(X)}")
    result = calc_evaluation_score(X, y, model)
    
    return result

import re
mat = re.compile("_module\.(.*)")

def train_loop(model, users, ptn):
  global_model = model
  global_model.to(device)
  global_model.train()
  print(global_model)

  dataset_cnt = float(sum([len(user.idxs) for user in users]))
  
  local_loss_list = []
  local_acc_list = []
  local_recall_list = []
  local_precision_list = []
  local_auroc_list = []
  local_f1_list = []


  for epoch in range(n_global_iterations):
    print(f"\nglobal epoch{epoch}")
    print(f"────────────────────────────────────────")
    local_weights = []
    local_losses = []

    global_model.train()

    # train loop
    ## 各ユーザーで学習を行う
    if ptn == 0:
      #print("ptn == 0")
      for i in range(len(users)):
        print("user\t", i, end=", ")
        w, loss = users[i].train(copy.deepcopy(global_model))
        local_weights.append(w)
        local_losses.append(loss)
    else:
      print("ptn == 1")
      for i in range(1):
        w, loss = users[i].train(copy.deepcopy(global_model))
        local_weights.append(w)
        local_losses.append(loss)
    
    ## パラメータを統合する
    avg_weights = {}
    for key in local_weights[0].keys():
      global_key = mat.match(key).group(1)
      avg_weights[global_key] = torch.mul(local_weights[0][key], (float(len(users[0].idxs)) / dataset_cnt))
      for i in range(1, len(local_weights)):
        avg_weights[global_key] += torch.mul(local_weights[i][key], (float(len(users[i].idxs)) / dataset_cnt))
      # avg_weights[key] = torch.div(avg_weights[key], len(local_weights))

    ## パラメータをモデルに反映させる

    print("update the global model")
    global_model.load_state_dict(avg_weights)

    # test loop
    global_model.eval()
    list_loss = []
    list_acc = []
    list_recall = []
    list_precision = []
    list_auroc = []
    list_f1 = []

    print(f"test the model of global iteration {epoch}")
    for i in range(len(users)):
        print("user\t", i, end=", ")
        result = users[i].test(global_model)
        list_loss.append(result['neg_log_loss'])
        list_acc.append(result['accuracy'])
        list_recall.append(result['recall'])
        list_precision.append(result['precision'])
        list_auroc.append(result['auroc'])
        list_f1.append(result['f1'])

    local_loss_list.append(list_loss)
    local_acc_list.append(list_acc)
    local_recall_list.append(list_recall)
    local_precision_list.append(list_precision)
    local_auroc_list.append(list_auroc)
    local_f1_list.append(list_f1)
    
    print("average scores of all users:")
    print(f"AUROC: {sum(list_auroc) / len(list_auroc)}\nF1: {sum(list_f1) / len(list_f1)}\nAccuracy: {sum(list_acc) / len(list_acc)}\nRecall: {sum(list_recall) / len(list_recall)}\nPrecision: {sum(list_precision) / len(list_precision)}\nAvg Loss: {sum(list_loss) / len(list_loss)}\n")
  
  return local_loss_list, local_acc_list, local_recall_list, local_precision_list, local_auroc_list, local_f1_list


train_data, test_data = get_datasets()

dt_now = datetime.datetime.now()
now_time = dt_now.strftime('%Y%m%d-%H%M%S')

result_dir = pathlib.Path(f"results/{now_time}")
result_dir.mkdir(exist_ok=True)

parser.add_argument('--n_global_iterations', type=int, default=30)
parser.add_argument('--n_iterations', type=int, default=3)
parser.add_argument('--n_small_users', type=int, default=4)
parser.add_argument('--noise_for_small', type=float, default=0)
parser.add_argument('--noise_for_large', type=float, default=0)
parser.add_argument('--share_abnormal_ratio', type=float, default=0)

parser.add_argument('--no_dp', action='store_true')
args = parser.parse_args()

n_global_iterations = args.n_global_iterations
learning_rate=0.01
noise_for_small = args.noise_for_small
noise_for_large = args.noise_for_large
n_epoch_in_large = 1
n_epoch_in_small = 1
batch_for_large = 64
batch_for_small = 64
data_size_for_large = 30000
data_size_for_small = 3000
n_small_users = args.n_small_users
n_large_users = 1
abnormal_ratio_for_small = 0.1
abnormal_ratio_for_large = 0.5
ITR = args.n_iterations
share_abnormal_ratio = args.share_abnormal_ratio

if noise_for_small == 0:
    eps = float("inf")
else:
    eps, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(data_size_for_small, batch_for_small, noise_for_small, n_global_iterations, 1e-5)

args = {"n_global_iterations": n_global_iterations, "learning_rate":learning_rate, "noise_for_small":noise_for_small,
"noise_for_large": noise_for_large, "n_epoch_in_large": n_epoch_in_large, "n_epoch_in_small":n_epoch_in_small,
"batch_for_large": batch_for_large, "data_size_for_small": data_size_for_small, "n_small_users": n_small_users,
"n_large_users": n_large_users, "abnormal_ratio_for_small": abnormal_ratio_for_small, "abnormal_ratio_for_large": abnormal_ratio_for_large,
"n_global_iterations": n_global_iterations, "ITR": ITR, "epsilon": eps, "share_abnormal_ratio": share_abnormal_ratio}

with open(result_dir / 'param.json', 'w') as f:
    json.dump(args, f, indent=4)

if share_abnormal_ratio > 0:
    print("n_large_users must be 1")
    assert(n_large_users == 1)
    data_allocation = [(data_size_for_large, abnormal_ratio_for_large, 0) for _ in range(n_large_users)] + [(data_size_for_small, abnormal_ratio_for_small, share_abnormal_ratio) for _ in range(n_small_users)]
    get_user_indices = get_user_indices_2
else:
    data_allocation = [(data_size_for_large, abnormal_ratio_for_large) for _ in range(n_large_users)] + [(data_size_for_small, abnormal_ratio_for_small) for _ in range(n_small_users)]

import warnings
warnings.simplefilter('ignore')
print("WARNING is IGNORED")
print(f"A small user guarantees ({eps},{1e-5})-DP")

abnormal_label = 1


abnormal_idxs, normal_idxs = get_indices_with_label(train_data, abnormal_label)
user_idxs_list = get_user_indices(data_allocation, normal_idxs, abnormal_idxs)

user_hyper_params_large = UserHyperParams(batch_size=batch_for_large, epoch=n_epoch_in_large, log_per=20, noise_multiplier=noise_for_large)
user_hyper_params_small = UserHyperParams(batch_size=batch_for_small, epoch=n_epoch_in_small, log_per=20, noise_multiplier=noise_for_small)

print(f"User count: {len(user_idxs_list)}")
users = []

for (j, idxs) in enumerate(user_idxs_list):
    if j in range(n_large_users):
        users.append(User(j, train_data, idxs, user_hyper_params_large))
    else:
        users.append(User(j, train_data, idxs, user_hyper_params_small))

avg_local_losses = np.zeros((n_global_iterations, len(users)))
avg_local_accs = np.zeros((n_global_iterations, len(users)))
avg_local_recalls = np.zeros((n_global_iterations, len(users)))
avg_local_precisions = np.zeros((n_global_iterations, len(users)))
avg_local_aurocs = np.zeros((n_global_iterations, len(users)))
avg_local_f1s = np.zeros((n_global_iterations, len(users)))

for i in range(ITR):
    # モデルのリセット
    model = NeuralNetwork().to(device)

    (_local_loss_list,
    _local_acc_list,
    _local_recall_list,
    _local_precision_list,
    _local_auroc_list,
    _local_f1_list) = train_loop(model, users, 0)

    assert(avg_local_losses.shape == np.array(_local_loss_list).shape)

    avg_local_losses += np.array(_local_loss_list, dtype=float) / ITR
    avg_local_accs += np.array(_local_acc_list, dtype=float) / ITR
    avg_local_recalls += np.array(_local_recall_list, dtype=float) / ITR
    avg_local_precisions += np.array(_local_precision_list, dtype=float) /ITR
    avg_local_aurocs += np.array(_local_auroc_list, dtype=float) /ITR
    avg_local_f1s += np.array(_local_f1_list, dtype=float) /ITR

results = (avg_local_losses, avg_local_accs, avg_local_recalls, avg_local_precisions, avg_local_aurocs, avg_local_f1s)

with open(result_dir / "result.pkl", "wb") as f:
    pickle.dump(results, f)

print(f"finish")
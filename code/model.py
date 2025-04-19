import torch
import torch.nn as nn
import pandas as pd
import torch.nn.init as init
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, importance_values=None):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

        # if importance_values is not None:
        #     self._initialize_weights(importance_values)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    # Set Weight
    def _initialize_weights(self, importance_values):
        with torch.no_grad():
            importance_values_tensor = torch.tensor(importance_values, dtype=torch.float32)
            fc1_weights_shape = self.fc1.weight.shape
            for i in range(fc1_weights_shape[1]):
                init.normal_(self.fc1.weight[:, i], mean=0, std=1) 
                self.fc1.weight.data[:, i] *= importance_values_tensor[i] 

data = pd.read_csv('../processed_data/final_processed_data.csv')
data_pre = pd.read_csv('../processed_data/standardized_data_pre.csv')
X = data.drop(columns=['label'])
feature_names = X.columns.tolist()
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',  # 二分类任务
    'eval_metric': 'logloss',        # 评估指标
    'max_depth': 6,                  # 树的最大深度
    'learning_rate': 0.01,            # 学习率
}
num_round = 100

bst = xgb.train(params, dtrain, num_round)
score_dict = bst.get_score(importance_type='weight')

y_pred = bst.predict(dtest)
y_pred_binary = (y_pred > 0.5).astype(int)

feature_map = {f'f{i}': name for i, name in enumerate(feature_names)}
importance_df_xgb = pd.DataFrame([
    {'feature': feature_map.get(k, k), 'importance': v}
    for k, v in score_dict.items()
])

importance_df_xgb['importance_norm'] = importance_df_xgb['importance'] / importance_df_xgb['importance'].sum()

precision_xgb = precision_score(y_test, y_pred_binary)
recall_xgb = recall_score(y_test, y_pred_binary)
f1_xgb = f1_score(y_test, y_pred_binary)
accuracy_xgb = np.sum(y_pred_binary == y_test) / len(y_test)

print(f"XGBoost 模型 - 准确率: {accuracy_xgb}")
print(f"XGBoost 模型 - 查准率: {precision_xgb}")
print(f"XGBoost 模型 - 召回率: {recall_xgb}")
print(f"XGBoost 模型 - F1 分数: {f1_xgb}")

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
# X_predict_tensor = torch.tensor(data_pre.values, dtype=torch.float32)

# 输入层 隐藏层 输出层 初始化参数
model = MLP(input_size=X_train.shape[1], hidden_size=64, 
            output_size=len(data['label'].unique()), importance_values = importance_df_xgb['importance_norm'].values)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, test_label = torch.max(test_outputs, 1)
    accuracy_mlp = (test_label == y_test_tensor).sum().item() / y_test_tensor.size(0)

    test_label_np = test_label.cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy()

    precision_mlp = precision_score(y_test_np, test_label_np)
    recall_mlp = recall_score(y_test_np, test_label_np)
    f1_mlp = f1_score(y_test_np, test_label_np)

    print(f"MLP 模型 - 准确率: {accuracy_mlp}")
    print(f"MLP 模型 - 查准率: {precision_mlp}")
    print(f"MLP 模型 - 召回率: {recall_mlp}")
    print(f"MLP 模型 - F1 分数: {f1_mlp}")

    # predict_outputs = model(X_predict_tensor)
    # _, predicted_labels = torch.max(predict_outputs, 1)
    # predicted_labels_np = predicted_labels.cpu().numpy()

    # print("Predicted optputs", predicted_labels_np)

input_weights = model.fc1.weight.detach().cpu().numpy()
# 计算每一列L2范数
feature_importance_mlp = np.linalg.norm(input_weights, axis=0) 
feature_importance_norm_mlp = feature_importance_mlp / np.sum(feature_importance_mlp)

importance_df_mlp = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance_norm_mlp
}).sort_values(by='importance', ascending=False)

print(importance_df_mlp)

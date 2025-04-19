import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

rf = RandomForestClassifier(n_estimators=100, random_state=42)

data = pd.read_csv('../processed_data/standardized_data.csv')

X = data.drop(columns=['label'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
rf.fit(X_train, y_train)

test_preds = rf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_preds))

# Get possibility
train_rf = rf.predict_proba(X_train)
test_rf = rf.predict_proba(X_test)

# Standardize
scaler = StandardScaler()
train_rf_scaled = scaler.fit_transform(train_rf)
test_rf_scaled = scaler.transform(test_rf) # do not use fit_transform

# Name new column
train_rf_scaled = pd.DataFrame(train_rf_scaled, columns=[f'rf_{i}' for i in range(train_rf_scaled.shape[1])])
test_rf_scaled = pd.DataFrame(test_rf_scaled, columns=[f'rf_{i}' for i in range(test_rf_scaled.shape[1])])

# cat
# X_train = pd.concat([X_train.reset_index(drop=True), pd.DataFrame(train_rf_scaled)], axis=1)
# X_test = pd.concat([X_test.reset_index(drop=True), pd.DataFrame(test_rf_scaled)], axis=1)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# 输入层 隐藏层 输出层
model = MLP(input_size=X_train.shape[1], hidden_size=64, output_size=len(data['label'].unique()))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)

    # softmax = nn.Softmax(dim=1)
    # probabilities = softmax(test_outputs)
    # print(probabilities)

    # 求出概率最大值的索引， 0 or 1
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'Test Accuracy: {accuracy}')

    # 2numpy
    predicted_np = predicted.cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy()

    # compute
    precision = precision_score(y_test_np, predicted_np)
    recall = recall_score(y_test_np, predicted_np)
    f1 = f1_score(y_test_np, predicted_np)

    print(f'Test Precision: {precision}')
    print(f'Test Recall: {recall}')
    print(f'Test F1-score: {f1}')
# Import numpy library for numerical computation
import numpy as np
# Import pandas library for data processing
import pandas as pd
# Import matplotlib library for plotting
import matplotlib.pyplot as plt
# Import PyTorch library and its neural network modules
import torch
import torch.nn as nn
import torch.optim as optim
# Import random library for generating random numbers
import random
# Import evaluation metrics from sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Import deque and namedtuple from collections for experience replay
from collections import deque, namedtuple
import pickle
import joblib

# Set random seed to ensure consistent results for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# 1. Data generation and preprocessing
SEQ_LENGTH = 3  # Sequence length, using data from the previous 3 months to predict the 4th month
# input_features = ['乌东德水位', '乌东德入库','乌出白入']
input_features = ['乌东德入库','乌出白入'] # List of input feature names
# Target feature, the quantity we want to predict
target_feature = '乌出白入'

# Path to the table data, modify according to your actual path
filepath = r'D:\code\学习\决策出的连续时间序列.xlsx'
# Read Excel file and load data
# df is a DataFrame, similar to a table
# Keep only the required input features and target feature
df = pd.read_excel(filepath)
df = df[input_features + [target_feature]]
data = df.values


# Data normalization
from sklearn.preprocessing import MinMaxScaler

# Normalize input features and target feature separately
scaler_input = MinMaxScaler()
scaler_target = MinMaxScaler()

input_data = data[:, :-1]
target_data = data[:, -1].reshape(-1, 1)

input_data = scaler_input.fit_transform(input_data)
target_data = scaler_target.fit_transform(target_data)

# Merge the data
normalized_data = np.hstack([input_data, target_data])

# Create sequence data
def create_sequences(data, seq_length):
    X, y, next_inflows = [], [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length, :-1]  # Input features from the previous 3 months
        y_val = data[i+seq_length, -1]     # Outflow for the 4th month
        # Add inflow for the 4th month ('乌东德入库' is the 2nd column of data)
        next_inflow = data[i+seq_length, 0]  # Index 0 corresponds to '乌东德入库'
        X.append(x)
        y.append(y_val)
        next_inflows.append(next_inflow)
    return np.array(X), np.array(y), np.array(next_inflows)

# Create sequence data
X, y, next_inflows = create_sequences(normalized_data, SEQ_LENGTH)

# Split training and test sets (use first 80% for training, last 20% for testing)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
next_inflows_train, next_inflows_test = next_inflows[:train_size], next_inflows[train_size:]

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)
next_inflows_train = torch.FloatTensor(next_inflows_train)
next_inflows_test = torch.FloatTensor(next_inflows_test)

# 2. LSTM prediction model
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward propagation
        out, _ = self.lstm(x, (h0, c0))
        
        # Take only the output of the last time step in the sequence
        out = self.fc(out[:, -1, :])
        return out

# 3. Physical constraint model
def physical_constraint_loss(prediction, sequence, next_inflow):
    """
    Physical constraint loss function for reservoir outflow prediction
    Constraints:
    1. Predictions cannot be negative
    2. Water balance constraint: total inflow = total outflow
    3. Reservoir release boundary constraint: discharge must be within minimum and maximum boundaries
    4. Reservoir operation cycle constraint: consider the cyclical characteristics of reservoir operation
    5. Reservoir capacity boundary constraint: active storage must remain within prescribed limits
    """
    # Constraint 1: Predictions cannot be negative
    non_negative_penalty = torch.mean(torch.clamp(-prediction, min=0)**2)
    
    # Constraint 2: Water balance constraint
    # Get inflow (column 1) and outflow (column 2) from the sequence
    past_inflows = sequence[:, :, 0]  # Inflow for the previous 3 months
    past_outflows = sequence[:, :, 1]  # Outflow for the previous 3 months
    
    # Calculate total inflow = previous 3 months inflow + 4th month inflow
    total_inflow = torch.sum(past_inflows, dim=1) + next_inflow
    
    # Calculate total outflow = previous 3 months outflow + predicted 4th month outflow
    total_outflow = torch.sum(past_outflows, dim=1) + prediction.squeeze()
    
    # Water balance loss (total inflow = total outflow)
    balance_loss = torch.mean((total_outflow - total_inflow)**2)
    
    # Constraint 3: Reservoir Release Boundary
    # Define upper and lower boundaries for discharge (set according to actual reservoir parameters)
    Q_low = Q_wdd_low   # Minimum discharge (m³/s)
    Q_up = Q_wdd_up  # Maximum discharge (m³/s)
    
    # Calculate boundary constraint loss
    # Penalty when predicted value is below lower boundary
    lower_boundary_penalty = torch.mean(torch.clamp(Q_low - prediction.squeeze(), min=0)**2)
    # Penalty when predicted value is above upper boundary
    upper_boundary_penalty = torch.mean(torch.clamp(prediction.squeeze() - Q_up, min=0)**2)
    
    boundary_loss = lower_boundary_penalty + upper_boundary_penalty
    
    # Constraint 4: Reservoir Operation Cycle
    # Consider the cyclical characteristics of reservoir operation, adjacent time steps should not have drastic outflow changes
    # Calculate change rate constraint between predicted value and historical outflow
    last_outflow = sequence[:, -1, 1]  # Outflow of the last month
    
    # Calculate change rate (relative change)
    flow_change_rate = torch.abs((prediction.squeeze() - last_outflow) / (last_outflow + 1e-6))
    
    # Set reasonable change rate threshold (e.g., monthly change rate should not exceed 50%)
    max_change_rate = 0.5
    cycle_loss = torch.mean(torch.clamp(flow_change_rate - max_change_rate, min=0)**2)
    
    # Constraint 5: Reservoir Capacity Boundary
    # Calculate reservoir volume based on water balance equation: V(t) = V(t-1) + (Q_in - Q_out) * Δt
    # Assume time step Δt = 1 (monthly), initial volume V0 (can be set based on historical data)
    dt = 1.0  # Time step (monthly)
    V0 = V_wdd_0  # Initial reservoir volume (million m³), set according to actual reservoir parameters
    
    # Calculate predicted reservoir volume
    # V(t) = V(t-1) + (Q_in(t-1) - Q_out(t-1)) * Δt
    V_predicted = V0 + (next_inflow - prediction.squeeze()) * dt
    
    # Define reservoir capacity boundaries (set according to actual reservoir parameters)
    V_low = V_wdd_low   # Minimum active storage (million m³)
    V_up = V_wdd_up    # Maximum active storage (million m³)
    
    # Calculate capacity constraint loss function g(V)
    # g(V) = (V - V_up)² if V > V_up
    #      = 0 if V_low ≤ V ≤ V_up  
    #      = (V_low - V)² if V < V_low
    capacity_penalty_upper = torch.mean(torch.clamp(V_predicted - V_up, min=0)**2)
    capacity_penalty_lower = torch.mean(torch.clamp(V_low - V_predicted, min=0)**2)
    
    capacity_loss = capacity_penalty_upper + capacity_penalty_lower
    
    # Total physical loss (weighted combination of all constraints)
    total_physical_loss = balance_loss +  boundary_loss +  cycle_loss +  capacity_loss
    
    return total_physical_loss

# 4. DQN agent for dynamic weight adjustment
# Define experience replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        """Save transition"""
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# DQN network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

# DQN agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, 
                 memory_capacity=10000, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Policy network and target network
        self.policy_net = DQN(state_dim, action_dim, hidden_dim)
        self.target_net = DQN(state_dim, action_dim, hidden_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_capacity)
        
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        self.steps_done = 0
    
    def select_action(self, state):
        """Select action using ε-greedy strategy"""
        sample = random.random()
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        if sample > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_dim)]], dtype=torch.long)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Compute mask for non-final states and concatenate batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        # Compute Q(s_t, a) - model computes Q(s_t), then select columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for next states
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

# 5. Training function
def train_lstm_with_dqn(lstm_model, dqn_agent, X_train, y_train, next_inflows_train,
                        X_test, y_test, num_episodes=100, lstm_lr=1e-3, 
                        target_update=10, device='cpu', beta=0.23):
    """
    Use DQN to dynamically adjust the weights of LSTM loss and physical loss
    
    Parameters:
    - lstm_model: LSTM prediction model
    - dqn_agent: DQN agent for weight selection
    - X_train, y_train: Training data
    - X_test, y_test: Test data
    - num_episodes: Number of training episodes
    - lstm_lr: LSTM learning rate
    - target_update: How often to update DQN target network
    - device: Training device
    - beta: Reward coefficient for weight fluctuation
    """
    lstm_model = lstm_model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # LSTM optimizer
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=lstm_lr)
    
    # Define available weight set (action space)
    # Here we define 11 possible weight values, from 0.0 to 1.0, with step size 0.1
    # Weight represents the weight of LSTM loss, physical loss weight is 1-weight
    possible_weights = torch.linspace(0.0, 1.0, 11)
    num_actions = len(possible_weights)
    
    # Record training process
    lstm_losses = []
    physical_losses = []
    total_losses = []
    selected_weights = []
    test_maes = []
    episode_rewards = []
    
    # Create fixed batch indices
    num_samples = X_train.size()[0]
    fixed_indices = torch.randperm(num_samples)  
    batch_starts = torch.arange(0, num_samples, 32)
    
    for episode in range(num_episodes):
        last_weight = None  # Record the weight of the previous batch
        # Initialize state
        # State includes: current episode, previous LSTM loss, previous physical loss, previous total loss, previous selected weight (5 states in total)
        state = torch.zeros(5).to(device)
        
        # One episode is a complete traversal of the training set
        lstm_epoch_loss = 0
        physical_epoch_loss = 0
        total_epoch_loss = 0
        episode_reward = 0
        batch_count = 0
        weights_in_episode = []  # Record the weight of each batch in this episode
        
        # Use fixed batch order
        for start_idx in batch_starts:
            end_idx = min(start_idx + 32, num_samples)
            indices = fixed_indices[start_idx:end_idx]
            
            batch_X = X_train[indices]
            batch_y = y_train[indices]
            batch_next_inflows = next_inflows_train[indices]
            
            # Fix weight to 1, completely remove physical mechanism (LSTM loss weight=1, physical loss weight=0)
            state_tensor = state.unsqueeze(0)
            action = dqn_agent.select_action(state_tensor)
            weight = possible_weights[action.item()]
            
            weights_in_episode.append(weight.item())  # Record weight
            
            # Forward propagation
            lstm_model.train()
            lstm_optimizer.zero_grad()
            
            # LSTM prediction
            outputs = lstm_model(batch_X)
            
            # Calculate LSTM loss (MSE between predicted and true values)
            lstm_loss = nn.MSELoss()(outputs, batch_y.unsqueeze(1))

            
            # Calculate physical constraint loss
            physical_loss = physical_constraint_loss(outputs, batch_X, batch_next_inflows)
            
            # Combined loss
            total_loss = weight * lstm_loss + (1 - weight) * physical_loss
            
            # Backpropagation and optimization
            total_loss.backward()
            lstm_optimizer.step()
            
            # Record losses
            lstm_epoch_loss += lstm_loss.item() * batch_X.size(0)
            physical_epoch_loss += physical_loss.item() * batch_X.size(0)
            total_epoch_loss += total_loss.item() * batch_X.size(0)
            
            # Calculate reward (negative loss, as we want to minimize loss)
            #reward = -total_loss.item()
            reward = -np.sqrt(lstm_loss.item() * physical_loss.item())
            # Add weight stability reward term
            if last_weight is not None:
                # Calculate absolute value of weight change
                weight_change = abs(weight.item() - last_weight)
                # Add penalty term: larger weight change, larger penalty
                stability_reward = -beta * weight_change
                reward += stability_reward

            episode_reward += reward
            batch_count += 1
            # Update last weight
            last_weight = weight.item()
            
            # Prepare next state
            next_state = torch.tensor([
                episode / num_episodes,  # Normalized episode
                lstm_loss.item(),
                physical_loss.item(),
                total_loss.item(),
                weight.item()
            ]).to(device)

            # Save experience
            dqn_agent.memory.push(state_tensor, action, next_state.unsqueeze(0), 
                                 torch.tensor([reward], device=device))

            # Optimize DQN model
            dqn_agent.optimize_model()
            
            # Update state
            state = next_state
        
        # Calculate average loss
        lstm_epoch_loss /= num_samples
        physical_epoch_loss /= num_samples
        total_epoch_loss /= num_samples
        
        # Calculate average reward
        avg_episode_reward = episode_reward / batch_count
        episode_rewards.append(avg_episode_reward)
        
        # Record average weight for this episode
        avg_weight = sum(weights_in_episode) / len(weights_in_episode)
        selected_weights.append(avg_weight)
        
        # Record losses
        lstm_losses.append(lstm_epoch_loss)
        physical_losses.append(physical_epoch_loss)
        total_losses.append(total_epoch_loss)
        
        # Evaluate on test set
        lstm_model.eval()
        with torch.no_grad():
            test_outputs = lstm_model(X_test)
            test_mae = mean_absolute_error(y_test.cpu().numpy(), test_outputs.cpu().numpy())
            test_maes.append(test_mae)
        
        # Print training progress
        if (episode + 1) % 10 == 0:
            print(f'Episode {episode+1}/{num_episodes}, '
                  f'LSTM Loss: {lstm_epoch_loss:.6f}, '
                  f'Physical Loss: {physical_epoch_loss:.6f}, '
                  f'Total Loss: {total_epoch_loss:.6f}, '
                  f'Avg Weight: {avg_weight:.2f}, '
                  f'Test MAE: {test_mae:.6f}, '
                  f'Avg Reward: {avg_episode_reward:.6f}')
        
        # Update DQN target network
        if episode % target_update == 0:
            dqn_agent.target_net.load_state_dict(dqn_agent.policy_net.state_dict())
    
    # Return training history
    return {
        'lstm_losses': lstm_losses,
        'physical_losses': physical_losses,
        'total_losses': total_losses,
        'selected_weights': selected_weights,
        'test_maes': test_maes,
        'episode_rewards': episode_rewards
    }
    
# 6. Run training
def main():
    # Check if GPU is available
    device = torch.device('cpu')
    print(f'Using device: {device}')

    # Create LSTM model
    input_size = len(input_features)
    hidden_size = 128
    num_layers = 2
    output_size = 1
    
    lstm_model = LSTMPredictor(input_size, hidden_size, num_layers, output_size)
    
    # Create DQN agent
    # State dimension: current episode, previous LSTM loss, previous physical loss, previous total loss, previous selected weight
    state_dim = 5
    # Action dimension: 11 possible weight values
    action_dim = 11
    
    dqn_agent = DQNAgent(state_dim, action_dim)
    
    # Train model
    history = train_lstm_with_dqn(
        lstm_model, 
        dqn_agent, 
        X_train, 
        y_train, 
        next_inflows_train,
        X_test, 
        y_test,
        num_episodes=120,
        device=device
    )
    torch.save(lstm_model.state_dict(), "model_best.pt")
    joblib.dump(scaler_input, "scaler_input.pkl")
    joblib.dump(scaler_target, "scaler_target.pkl")
    print("✅ Saved: model_best.pt, scaler_input.pkl, scaler_target.pkl")
    
    # Evaluate model
    lstm_model.eval()
    with torch.no_grad():
        test_outputs = lstm_model(X_test.to(device))
        test_outputs = test_outputs.cpu().numpy()
        y_test_np = y_test.cpu().numpy()
    # Extract historical inflow, outflow, and true 4th month inflow from test data
    past_inflows = X_test[:, :, 0].cpu().numpy()  # Inflow for the first 3 months
    past_outflows = X_test[:, :, 1].cpu().numpy()  # Outflow for the first 3 months
    true_next_inflows = next_inflows_test.cpu().numpy()  # True inflow for the 4th month

    # Total inflow = first 3 months true inflow + 4th month true inflow
    total_true_inflows = np.sum(past_inflows, axis=1) + true_next_inflows.squeeze()

    # Total outflow = first 3 months true outflow + 4th month predicted outflow (model prediction)
    total_predicted_outflows = np.sum(past_outflows, axis=1) + test_outputs.squeeze()

    # Prevent division by 0 (add small constant)
    total_true_inflows = np.where(total_true_inflows == 0, 1e-6, total_true_inflows)

    # WBI = outflow / inflow
    wbi_array = total_predicted_outflows / total_true_inflows
    wbi_mean = np.mean(wbi_array)


    # Inverse normalization
    test_outputs = scaler_target.inverse_transform(test_outputs)
    y_test_np = scaler_target.inverse_transform(y_test_np.reshape(-1, 1))
    
    # Calculate evaluation metrics
    # ===== Calculate NSE =====
    def nse(y_true, y_pred):
        return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    # ===== Calculate KGE =====
    def kge(y_true, y_pred):
        r = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
        alpha = np.std(y_pred) / np.std(y_true)
        beta = np.mean(y_pred) / np.mean(y_true)
        return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    mae = mean_absolute_error(y_test_np, test_outputs)
    mse = mean_squared_error(y_test_np, test_outputs)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_np, test_outputs)
    nse_score = nse(y_test_np, test_outputs)
    kge_score = kge(y_test_np, test_outputs)
    print(f'\nFinal evaluation results:')
    print(f'MAE: {mae:.4f}')
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'R2 Score: {r2:.4f}')
    print(f'NSE: {nse_score:.4f}')
    print(f'KGE: {kge_score:.4f}')
    print(f'WBI (Water Balance Index): {wbi_mean:.4f}')

    # Calculate exponential moving average (EMA) for weight curve
    def calculate_ema(values, alpha=0.1):
        ema = [values[0]]
        for i in range(1, len(values)):
            ema.append(alpha * values[i] + (1 - alpha) * ema[i-1])
        return ema

    # Calculate exponential moving average for reward curve
    alpha = 1  # Smoothing factor, smaller values result in smoother curves
    smoothed_weights = calculate_ema(history['selected_weights'], alpha)
    smoothed_rewards = calculate_ema(history['episode_rewards'], alpha)
    
    # Plot training history
    plt.figure(figsize=(18, 12))
    
    # Loss curves
    plt.subplot(2, 3, 1)
    plt.plot(history['lstm_losses'], label='LSTM Loss')
    plt.plot(history['physical_losses'], label='Physical Loss')
    plt.plot(history['total_losses'], label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    
    # Weight selection curve - with smoothing
    plt.subplot(2, 3, 2)
    plt.plot(history['selected_weights'], alpha=0.1, label='Original Weights')
    plt.plot(smoothed_weights, 'r-', linewidth=2, label=f'Smoothed (α={alpha})')
    plt.xlabel('Epoch')
    plt.ylabel('Weight')
    plt.title('Selected Weights for LSTM Loss')
    plt.legend()
    plt.grid(True)
    
    # Test MAE curve
    plt.subplot(2, 3, 3)
    plt.plot(history['test_maes'])
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Test MAE')
    plt.grid(True)
    
    # Reward curve - newly added
    plt.subplot(2, 3, 4)
    plt.plot(history['episode_rewards'], color='purple')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('DQN Reward per Episode')
    plt.grid(True)
    
    # Actual vs predicted comparison
    plt.subplot(2, 3, 5)
    plt.plot(y_test_np, label='Actual')
    plt.plot(test_outputs, label='Predicted')
    plt.xlabel('Time Step')
    plt.ylabel('Radiation')
    plt.title('Actual vs Predicted Radiation')
    plt.legend()
    plt.grid(True)
    
    
    #plt.savefig('training_results.png', dpi=300)  # Save result plots
    plt.show()
    # ========== Save true values + training + test predictions ==========
    
    # Get training set predictions
    lstm_model.eval()
    with torch.no_grad():
        train_preds = lstm_model(X_train.to(device)).cpu().numpy()
        test_preds = lstm_model(X_test.to(device)).cpu().numpy()
    
    # Inverse normalize all data
    y_train_inv = scaler_target.inverse_transform(y_train.cpu().numpy().reshape(-1, 1)).flatten()
    y_test_inv = scaler_target.inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).flatten()
    train_preds_inv = scaler_target.inverse_transform(train_preds).flatten()
    test_preds_inv = scaler_target.inverse_transform(test_preds).flatten()
    
    # Create training set results DataFrame
    train_result_df = pd.DataFrame({
        'Dataset': ['Training'] * len(y_train_inv),
        'Sample Index': range(1, len(y_train_inv) + 1),
        'True Value': y_train_inv,
        'Predicted Value': train_preds_inv,
        'Absolute Error': np.abs(y_train_inv - train_preds_inv),
        'Relative Error (%)': np.abs(y_train_inv - train_preds_inv) / np.abs(y_train_inv) * 100
    })
    
    # Create test set results DataFrame
    test_result_df = pd.DataFrame({
        'Dataset': ['Test'] * len(y_test_inv),
        'Sample Index': range(1, len(y_test_inv) + 1),
        'True Value': y_test_inv,
        'Predicted Value': test_preds_inv,
        'Absolute Error': np.abs(y_test_inv - test_preds_inv),
        'Relative Error (%)': np.abs(y_test_inv - test_preds_inv) / np.abs(y_test_inv) * 100
    })
    
    # Merge training and test set results
    all_results_df = pd.concat([train_result_df, test_result_df], ignore_index=True)
    
    # Add statistical information
    stats_df = pd.DataFrame({
        'Dataset': ['Training Statistics', 'Test Statistics', 'Overall Statistics'],
        'Sample Index': ['', '', ''],
        'True Value': ['', '', ''],
        'Predicted Value': ['', '', ''],
        'Absolute Error': [
            f'MAE: {np.mean(np.abs(y_train_inv - train_preds_inv)):.4f}',
            f'MAE: {np.mean(np.abs(y_test_inv - test_preds_inv)):.4f}',
            f'MAE: {np.mean(np.abs(np.concatenate([y_train_inv, y_test_inv]) - np.concatenate([train_preds_inv, test_preds_inv]))):.4f}'
        ],
        'Relative Error (%)': [
            f'MAPE: {np.mean(np.abs(y_train_inv - train_preds_inv) / np.abs(y_train_inv) * 100):.2f}%',
            f'MAPE: {np.mean(np.abs(y_test_inv - test_preds_inv) / np.abs(y_test_inv) * 100):.2f}%',
            f'MAPE: {np.mean(np.abs(np.concatenate([y_train_inv, y_test_inv]) - np.concatenate([train_preds_inv, test_preds_inv])) / np.abs(np.concatenate([y_train_inv, y_test_inv])) * 100):.2f}%'
        ]
    })
    
    # Add statistical information to results
    final_results_df = pd.concat([all_results_df, stats_df], ignore_index=True)
    
    # Save as Excel file
    result_path = 'Complete_Prediction_Results_Training_and_Test.xlsx'
    with pd.ExcelWriter(result_path, engine='openpyxl') as writer:
        # Save complete results
        final_results_df.to_excel(writer, sheet_name='Complete Results', index=False)
        # Save training and test sets separately
        train_result_df.to_excel(writer, sheet_name='Training Results', index=False)
        test_result_df.to_excel(writer, sheet_name='Test Results', index=False)
    
    print(f"\n✅ Complete prediction results saved to: {result_path}")
    print(f"📊 Training samples: {len(y_train_inv)}, Test samples: {len(y_test_inv)}")
    print(f"📈 Training MAE: {np.mean(np.abs(y_train_inv - train_preds_inv)):.4f}")
    print(f"📈 Test MAE: {np.mean(np.abs(y_test_inv - test_preds_inv)):.4f}")

if __name__ == "__main__":
    main()


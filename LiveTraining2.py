import os
#import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PyQt6.QtCore import pyqtSignal, QObject
from stable_baselines3 import PPO, DQN, A2C, SAC, TD3
from UsersField.MarketData.marketdata import MarketData  # Your MarketData class here
from Main.Functions.core import appy
from Main.Functions.utils import resolve_stock_name, get_model_folder
from UsersField.ReplayBuffer.ReplayBuffer import ReplayBuffer
#from UsersField.Logs.logger_setup import log

class BaseTradingEnv(gym.Env, QObject):
    def __init__(self, symbol):
        super().__init__()
        self.repo = appy.get_repo()
        self.symbol =resolve_stock_name(self.repo) 

        self.wallet_total = self.repo.get_value("wallet/total", 10001, float)
        self.net_worth = self.wallet_total
        self.max_net_worth = self.wallet_total
        self.shares_held = 0
        self.current_step = 0

        self.action_space = spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10 * 10,), dtype=np.float32)

        # Initialize MarketData instance for symbol
        self.market_data = MarketData(symbol=self.symbol)

    def reset(self):
        self.current_step = 0
        self.shares_held = 0
        self.net_worth = self.wallet_total
        obs = self._get_observation()
        return obs, {}

    def _get_observation(self):
        # Get the next MarketData observation at current step
        obs = self.market_data.get_window_obs(self.current_step)
        return obs

    def step(self, action):
        done = False
        reward = 0.0

        current_price = self.market_data.get_price(self.current_step)
        if action == 0:  # Buy
            if self.wallet_total >= current_price:
                self.shares_held += 1
                self.wallet_total -= current_price
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.wallet_total += self.shares_held * current_price
                self.shares_held = 0

        self.net_worth = self.wallet_total + self.shares_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        reward = self.net_worth - self.wallet_total  # Profit as reward

        self.current_step += 1
        if self.current_step >= self.market_data.length - 1:
            done = True

        obs = self._get_observation()
        info = {}

        return obs, reward, done, False, info


class RLProcessor(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    rl_action = pyqtSignal(int)
    
    def __init__(self, mode="live"):
        super().__init__()
        self.mode = mode
        self.repo = appy.get_repo()
        self.symbol = resolve_stock_name(self.repo) 
        self.policy = self.repo.get_value("RLworker/active", "PPO")
        self.algo_cls = {"PPO": PPO, "DQN": DQN, "A2C": A2C, "SAC": SAC, "TD3": TD3}.get(self.policy)
        self.mode = self.repo.get_value("RLworker/mode", "live")
        self.do_training = self.repo.get_value("RLworker/do_training", False)

        self.save_dir = get_model_folder(self.repo)
        os.makedirs(self.save_dir, exist_ok=True)
        self.model_path = os.path.join(self.save_dir, f"{self.symbol}_{self.policy}_latest.zip")
        self.buffer_path = os.path.join(self.save_dir, f"{self.symbol}_replay.pkl")

        self.env = BaseTradingEnv(symbol=self.symbol)
        self.model = None
        self.replay_buffer = ReplayBuffer()

        self.load_model()
        self.load_replay_buffer()
        self.running = True
        self.market_data = self.env.market_data
        self.progress.emit(f"RLProcessor initialized for {self.symbol} in {self.mode} mode.")
        
    def predict_from_dom(self, dom):
        obs = self.build_observation_from_dom(dom)
        obs = obs.flatten()
        action, _ = self.model.predict(obs, deterministic=True)
        label = ["BUY", "HOLD", "SELL"][action] if action in [0, 1, 2] else "ΑΓΝΩΣΤΟ"
        self.progress.emit(f"RLProcessor predict {label} in ")
        self.rl_action.emit(int(action))
        return int(action)


    def build_observation_from_dom(self, data):
        """
        Converts full DOM data from WebSocket into a 10x10 observation vector.
        Uses top 5 buys and 5 sells, each with price + quantity.
        """
        try:
            upd = data.get("upd", [])
            ins = data.get("ins", [])
            dom_rows = upd + ins

            buys = sorted([r for r in dom_rows if r.get("s") == "B"], key=lambda x: -x["p"])[:5]
            sells = sorted([r for r in dom_rows if r.get("s") == "S"], key=lambda x: x["p"])[:5]

            obs = []
            for row in buys:
                obs.extend([row.get("p", 0), row.get("q", 0)])
            for row in sells:
                obs.extend([row.get("p", 0), row.get("q", 0)])

            # Fill up to 10x2 (10 entries: 5 buys, 5 sells)
            while len(obs) < 10 * 10:
                obs.append(0.0)

            return np.array(obs, dtype=np.float32)

        except Exception as e:
            print(f"Failed to build observation from DOM: {e}")
            return None

    def create_model(self, auto_train=True):
        if not self.algo_cls:
            raise ValueError(f"Unknown policy: {self.policy}")

        if os.path.exists(self.model_path):
            self.progress.emit(f"Loading existing model from {self.model_path}")
            return self.algo_cls.load(self.model_path, env=self.env)

        self.progress.emit(f"Creating new model ({self.policy}) for {self.symbol}")
        model = self.algo_cls("MlpPolicy", self.env, verbose=0)
        if auto_train and self.mode != "live":
            steps = int(self.repo.get_value("RLworker/episodes", 5000))
            self.progress.emit(f"Training new model for {steps} steps")
            model.learn(total_timesteps=steps)
            model.save(self.model_path)
            self.progress.emit(f"Model saved to {self.model_path}")
        return model

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = self.algo_cls.load(self.model_path, env=self.env)
                self.progress.emit(f"Model loaded from {self.model_path}")
            except Exception as e:
                self.progress.emit(f"Failed to load model: {e}")
                self.model = self.create_model(auto_train=True)
        else:
            self.model = self.create_model(auto_train=True)

    def save_model(self):
        if self.model:
            self.model.save(self.model_path)
            self.progress.emit(f"Model saved to {self.model_path}")

    def load_replay_buffer(self):
        if os.path.exists(self.buffer_path):
            try:
                self.replay_buffer.load(self.buffer_path)
                self.progress.emit(f"Replay buffer loaded with {len(self.replay_buffer)} entries")
            except Exception as e:
                self.progress.emit(f"Failed to load replay buffer: {e}")

    def save_replay_buffer(self):
        if self.replay_buffer:
            self.replay_buffer.save(self.buffer_path)
            self.progress.emit(f"Replay buffer saved with {len(self.replay_buffer)} entries")

    def train_loop(self, steps=1000):
        self.progress.emit("Starting training loop...")
        self.model.learn(total_timesteps=steps)
        self.save_model()
        self.save_replay_buffer()
        self.progress.emit("Training complete.")

    def live_loop(self):
        self.progress.emit("Starting live prediction loop...")
        obs, _ = self.env.reset()
        while self.running:
            action, _ = self.model.predict(obs, deterministic=True)
            self.rl_action.emit(int(action))
            obs, reward, done, truncated, info = self.env.step(action)
            self.progress.emit(f"Step {self.env.current_step}: Action={action}, Reward={reward:.4f}, NetWorth={self.env.net_worth:.2f}")
            if done:
                obs, _ = self.env.reset()
            #time.sleep(0.1)
            
    def run_live_loop(self):
        while self.running:
            try:
                date, obs = self.market_data.next_obs()
                if date is None:  # No more obs — reached today
                    self.progress.emit("Reached latest available data, resetting…")
                    self.market_data.reset()  # Start over from beginning
                    continue
                obs = np.expand_dims(obs, axis=0)
                action, _ = self.model.predict(obs, deterministic=True)
                self.progress.emit(f"{date} → Action: {action}")
                self.rl_action.emit(int(action))
            except Exception as e:
                self.progress.emit(f"Error in live loop: {e}")
                break
            
            
    def stop(self):
        self.running = False
        self.progress.emit("RLProcessor stopped.")

    def process(self):
        self.env.reset()
        self.progress.emit("✅ Environment reset.")
        if self.mode == "live":
            self.progress.emit("✅ Environment reset.")
            self.run_live_loop()
        elif self.mode in ("training", "backtest"):
            self.progress.emit("✅ training")
            self.run_training_loop(steps=1000)
        self.finished.emit()
# Example usage:
# rl_proc = RLProcessor(symbol="AAPL")
# rl_proc.train_loop(steps=5000)
# rl_proc.live_loop()

# -*- coding: utf-8 -*-
"""Module implementing fin. by alkis amanatidis"""
from Main.Functions.core import appy
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
#from dataclasses import dataclass
#import numpy as np
#from Main.Functions.core import appy
#from pathlib import Path

class MarketData:
    def __init__(self, symbol, start_index=0):
        self.symbol = symbol
        self.start_index = start_index
        self.current_idx = start_index
        
        self.repo = appy.get_repo()
        self.history_df = None
        self.ws_df = None
        self.signals_df = None
        self.scaler = MinMaxScaler()

        self.folder_export =os.path.expanduser(self.repo.get_value("FOLDER/EXPORT"))
        os.makedirs(self.folder_export, exist_ok=True)
        self.TENSORBOARD = os.path.join((self.folder_export),self.repo.get_value("FOLDER/TENSORBOARD"))        
        self.GENERAL =  os.path.join((self.folder_export), self.repo.get_value("FOLDER/GENERAL"))        
        self.SPYDATA =  os.path.join((self.folder_export), self.repo.get_value("FOLDER/SPYDATA"))
        self.AI =  os.path.join((self.folder_export), self.repo.get_value("FOLDER/AI"))
        self.AI_models=f"{self.AI}/models"
        
        self.raw_path =os.path.join(self.SPYDATA, f"RD_{self.symbol}.csv")          
        self.ws_path = os.path.join(self.SPYDATA, f"WS_{self.symbol}.csv")
        self.signals_path =os.path.join(self.SPYDATA,  f"SIGNALS_{self.symbol}.csv")
        
        # Always initialize as empty DataFrames, never None
        self.history_df = pd.DataFrame()
        self.ws_df = pd.DataFrame()
        self.signals_df = pd.DataFrame()

        self.load_all()

    def _safe_load_csv(self, path, enrich_fn=None):
        """Loads CSV if exists, else returns empty DataFrame."""
        if path and os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
            if enrich_fn:
                df = enrich_fn(df)
            return df
        return pd.DataFrame()

    def load_all(self):
        self.history_df = self._safe_load_csv(self.raw_path, self._enrich_indicators)
        self.ws_df = self._safe_load_csv(self.ws_path, self._process_dom)
        self.signals_df = self._safe_load_csv(self.signals_path)

        # Normalize only if we have data
        if not self.history_df.empty:
            numeric_cols = self.history_df.select_dtypes(include=[np.number]).columns
            self.history_df[numeric_cols] = self.scaler.fit_transform(self.history_df[numeric_cols])


    #def load_all(self):
        #if self.raw_path:
            #self.history_df = pd.read_csv(self.raw_path, parse_dates=["Date"], index_col="Date")
            #self.history_df = self._enrich_indicators(self.history_df)
        #if self.ws_path:
            #self.ws_df = pd.read_csv(self.ws_path, parse_dates=["Date"], index_col="Date")
            #print("self.ws_df", self.ws_df)
            #self.ws_df = self._process_dom(self.ws_df)
#
        #if self.signals_path:
            #self.signals_df = pd.read_csv(self.signals_path, parse_dates=["Date"], index_col="Date")
#
        ## Normalize only numeric columns, preserving Date index
        #if self.history_df is not None:
            #numeric_cols = self.history_df.select_dtypes(include=[np.number]).columns
            #self.history_df[numeric_cols] = self.scaler.fit_transform(self.history_df[numeric_cols])

    def _process_dom(self, ws_df):
        """Parse DOM JSON (ins/upd/del), create raw and normalized indicators."""
        features = []
        for raw in ws_df["payload"]:
            try:
                dom = json.loads(raw)
                ins = len(dom.get("ins", []))
                upd = len(dom.get("upd", []))
                dele = len(dom.get("del", []))
                total_orders = ins + upd - dele
                avg_price = np.mean([o.get("p", 0) for o in dom.get("ins", [])]) if dom.get("ins") else 0
                features.append([ins, upd, dele, total_orders, avg_price])
            except Exception:
               features.append([0, 0, 0, 0, 0 ])

        # Create DataFrame from raw features
        df_features = pd.DataFrame(features, columns=["ins_count", "upd_count", "del_count", "net_orders", "avg_price"])
        #print("self.df_features", df_features)
        # Normalize each column to 0-1 range — min-max scaling on the batch
        df_norm = (df_features - df_features.min()) / (df_features.max() - df_features.min())
        df_norm = df_norm.fillna(0)  # in case min==max

        # Rename normalized columns to distinguish
        df_norm.columns = [col + "_norm" for col in df_norm.columns]
        #print("df_norm", df_norm)
        # Concatenate raw + normalized features: total 10 columns
        ws_features = pd.concat([df_features, df_norm], axis=1)

        ws_df = ws_df.drop(columns=["payload"]).reset_index(drop=True)
        ws_features = ws_features.reset_index(drop=True)

        ws_df = pd.concat([ws_df, ws_features], axis=1)
        #print("ws_df", ws_df)
        ws_df.to_csv(self.SPYDATA + f"/newWS{self.symbol}.csv", index=True)
        #print("ws_df", ws_df)
        return ws_df
       
    def reset_pointer(self, start_idx=None):
        max_start = len(self.history_df) - 10  # max start so 10 rows fit (7+2+1)
        if max_start < 0:
            raise ValueError("Not enough data to get a full observation window")

        if start_idx is not None:
            if start_idx > max_start:
                raise ValueError(f"start_idx {start_idx} too close to end, max allowed is {max_start}")
            self.start_index = start_idx
        else:
            self.start_index = np.random.randint(0, max_start + 1)  # inclusive upper bound

        self.current_idx = self.start_index
        
    def get_window_obs(self, idx, window_size=10):
        """
        Returns a numeric observation window of shape (window_size, n_cols)
        starting at index `idx`. Pads with random normalized values if
        there aren’t enough rows.
        """
        df = self.history_df
        n_cols = df.shape[1] if df is not None else 0

        def pad_or_slice(df, start, length, n_cols):
            # Slice available rows, force float32
            part = df.iloc[start:start+length].to_numpy(dtype=np.float32) if df is not None else np.empty((0, n_cols), dtype=np.float32)

            # How many rows missing?
            missing = length - part.shape[0]
            if missing > 0:
                # padding with random normalized values (float32)
                padding = np.random.uniform(low=0.0, high=1.0, size=(missing, n_cols)).astype(np.float32)
                part = np.vstack([part, padding])

            # Sanitize NaN/inf values
            part = np.nan_to_num(part, nan=0.0, posinf=0.0, neginf=0.0)
            return part

        obs_window = pad_or_slice(df, idx, window_size, n_cols)

        # Optional: flatten to 1D if your model expects it
        #obs_window = obs_window.astype(np.float32)
        obs_window = obs_window.flatten().astype(np.float32)
        return obs_window
 
 
    #def get_window_obs(self, idx):
        #def pad_or_slice(df, start, length, n_cols):
            ## Slice available rows
            #part = df.iloc[start:start+length].to_numpy() if df is not None else np.empty((0, n_cols))
            ## How many rows missing?
            #missing = length - part.shape[0]
            #if missing > 0:
                ## Create padding rows (zeros or random normalized values)
                ##padding = np.zeros((missing, n_cols), dtype=part.dtype)
                ## Or for random normalized values:
                #padding = np.random.uniform(low=0.0, high=1.0, size=(missing, n_cols))
                #part = np.vstack([part, padding])
            #return part
#
        #n_cols = self.history_df.shape[1]  # assuming all dfs have same columns count
        #hist_part = pad_or_slice(self.history_df, idx, 7, n_cols)
        #ws_part = pad_or_slice(self.ws_df, idx, 2, n_cols)
        #sig_part = pad_or_slice(self.signals_df, idx, 1, n_cols)
#
        #obs = np.vstack([hist_part, ws_part, sig_part])
        #obs_flat = obs.flatten()
        #
        #
        ##print("obs_flat", obs_flat)
        #return obs_flat
    
    
    
    def next_obs(self):
        obs = self.get_window_obs(self.current_idx)
        date = self.history_df.index[self.current_idx] if self.history_df is not None else None
        self.current_idx += 1
        #print(" date:" ,  date, "obs:",  obs)
        return date, obs
        
        
    def _enrich_indicators(self, df):
        df = df.copy()
        # Convert to numeric safely
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Add indicators (some may produce NaNs at start)
        df['RSI'] = RSIIndicator(df['close']).rsi()
        stoch = StochasticOscillator(df['high'], df['low'], df['close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        macd = MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()

        # Keep only the columns you want to have (exactly 10)
        cols_to_keep = ['open', 'high', 'low', 'close', 'volume', 'RSI', 'Stoch_K', 'Stoch_D', 'MACD', 'MACD_signal']
        df = df[cols_to_keep]

        # Fill NaNs with 0 or use forward fill if you prefer
        df = df.fillna(0)

        # Optionally normalize here (e.g., min-max per column)
        df_norm = (df - df.min()) / (df.max() - df.min())
        df_norm = df_norm.fillna(0)

        # You can return the enriched raw or normalized — 
        # or concat raw + normalized if you want 20 columns total
        #print("df_norm", df_norm)
        return df_norm

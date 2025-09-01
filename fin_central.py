# -*- coding: utf-8 -*-
"""Module implementing fin. by alkis amanatidis"""
from UsersField.Freedom.Freedom import Freedom
from UsersField.Telegrame.telegramme_com import TelegramThread
from UsersField.WebSocket.WebSocketWorker import WebSocketWorker
from UsersField.Logs.logger_setup import log
from UsersField.AiSignals.chatgpt import SignalFetcher
#from UsersField.Agents.Agents import ARanking, DARanking, AStohastic, DAStohastic, ATrade 
from UsersField.DataCenter.data_center import DataCenter
from UsersField.MarketData.marketdata import MarketData  
from UsersField.ProcessorFactory.ProcessorFactory import ProcessorFactory

from Main.Functions.core import appy
#from Main.Functions.Repo import Repo
from Main.Functions.utils import resolve_stock_name, getfilepath
from Main.Functions.softone import *
from Main.Ui.Ui_fin_central import Ui_fin

#from PyQt6.QtCore import *
#from PyQt6.QtCore import Qt
#from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QWidget , QFileDialog, QInputDialog#, QListWidgetItem
from PyQt6.QtGui import QDesktopServices
import configparser, os, schedule,  json, csv#, time, logging, requests, subprocess, sys, , asyncio
import pandas as pd
#import pandas_ta as ta
#from bs4 import BeautifulSoup
from datetime import datetime
#from functools import partial
#from pandas import DataFrame
#from openpyxl.utils.dataframe import dataframe_to_rows
#import asyncio
#import websockets
#from dataclasses import dataclass
#from tradernet import TraderNetWSAPI

now = datetime.now()
date_now = now.strftime("%y%m%d")
date = now.strftime("%Y-%m-%d")
import numpy as np
#import yfinance as yf

from pathlib import Path


class fin(QWidget, Ui_fin):
    def __init__(self, parent=None):
        super(fin, self).__init__(parent)

        self.setupUi(self)

    @pyqtSlot()
    def init_setup(self):
        self.init()
        
        self.init_scheduling()
        
        self.init_RL()
        
    @pyqtSlot()
    def init(self):
        self.log_real = log(mode="REAL DATA")
        self.log_train = log(mode="TRAIN DATA")
        self.log_wallet = log(mode="Wallet")
        self.log_real.info("#######################  Going to the Fin Board   ##############################")
        self.log_train.info("#######################  TRAINING to the Fin Board   ##############################")
          
        self.repo = appy.get_repo()
        self.mpl_widget =appy.get_widget()
        self.freedom_ins =appy.get_freedom_ins()
        
        self.repo.data_changed.connect(self.refresh_central_data)
        self.lw_central_data.itemDoubleClicked.connect(self.edit_value)
        self.refresh_central_data()
        
        self.config = configparser.ConfigParser()
        self.config.read('PrivateData/db.ini')
        
        self.folder_export =os.path.expanduser(self.repo.get_value("FOLDER/EXPORT"))
        os.makedirs(self.folder_export, exist_ok=True)
        self.TENSORBOARD = os.path.join((self.folder_export),self.repo.get_value("FOLDER/TENSORBOARD"))        
        self.GENERAL =  os.path.join((self.folder_export), self.repo.get_value("FOLDER/GENERAL"))        
        self.SPYDATA =  os.path.join((self.folder_export), self.repo.get_value("FOLDER/SPYDATA"))
        self.AI =  os.path.join((self.folder_export), self.repo.get_value("FOLDER/AI"))
        self.AI_models=f"{self.AI}/models"            
        
        
        self.urls=self.config["url"]["general_data"]
        self.index=self.config["output"]["index"]
        
        self.spy=""
        self.lcd_total_wallet.display(float(self.repo.get_value("wallet/total")))      
        self.lcd_total_wallet.setStyleSheet("QLCDNumber {background-color:yellow;color:black}")  
        self.lcd_cash.setStyleSheet("QLCDNumber {background-color:yellow;color:black}")  
        self.lcd_price_now.setStyleSheet("QLCDNumber {background-color:yellow;color:black}")  
        self.lcd_price_bought.setStyleSheet("QLCDNumber {background-color:yellow;color:black}")  
        self.lcd_profit.setStyleSheet("QLCDNumber {background-color:yellow;color:black}")  
        self.lcd_temaxia.setStyleSheet("QLCDNumber {background-color:yellow;color:black}")  
        self.lcd_bought_price.setStyleSheet("QLCDNumber {background-color:orange;color:black}")
        self.lcd_current_price.setStyleSheet("QLCDNumber {background-color:orange;color:black}")
        self.lcd_fees.setStyleSheet("QLCDNumber {background-color:red;color:black}")
        

        self.telegram_bot = None
        self.websocket_worker = None
        self.atrade = None
        self.env = None
        self.mode = self.repo.get_value("RLworker/mode", "live")

        self.get_stock_status()
              
        self.websocket_task = None
        self.training_mode = False
        self.training_thread = None
        self.rb_training_mode.setChecked(self.training_mode)
        self.rb_training_mode.clicked.connect(self.training_pipeline_toggle)
        
        self.dc = DataCenter()
        self.processor_factory = ProcessorFactory()
        
        self.telegram_bot = TelegramThread(self.repo, self) 
        self.telegram_bot.start()  # ✅ Start bot
        self.lw_4.addItem("✅ Enabling Telegram Bot...")
        
        self.cb_spy_buy.clear()
        self.finalist.clear()
        for  candinate in self.repo.get_section_values("candinates"):
            self.finalist.addItem(candinate)
            self.cb_spy_buy.addItem(candinate)
             
        if not schedule.jobs:
            print("No scheduled jobs.")
        else:
            schedule.clear()  
            
        if self.freedom_ins:
             self.lw_4.addItem("✅ connection already exists.")
             print("✅ connection already exists")
        else:
             self.init_fin()
             self.lw_4.addItem("✅ Freedom24 connection Established...")
             print("✅ Freedom24 connection Established...")

        self.get_wallet()
            
    def init_RL(self):
        try:
            if hasattr(self, 'rl_worker') and self.rl_worker:
                self.rl_worker.save_replay_buffer()
                self.rl_worker.save_model()
                self.rl_worker.stop()
                self.rl_worker = None
            if hasattr(self, 'rl_thread') and self.rl_thread:
                if self.rl_thread.isRunning():
                    self.rl_thread.quit()
                    self.rl_thread.wait()
                self.rl_thread.deleteLater()
                self.rl_thread=None
            self.mode = self.repo.get_value("RLworker/mode", "live")
            self.rl_worker, self.rl_thread = self.processor_factory.get_processor(self.mode)
            print(f"✅ RL processor initialized: {type(self.rl_worker)}")
            self.rl_worker.progress.connect(self.update_status)
            self.rl_worker.finished.connect(self.on_rl_worker_ready)
            self.rl_worker.rl_action.connect(self.handle_rl_action)   
            self.rl_thread.finished.connect(self.rl_thread.deleteLater)
            self.rl_thread.start()
        except FileNotFoundError as e:
            print(f"❌ Error initializing RL: {e}")

    @pyqtSlot(int)
    def handle_rl_action(self, action):
        pass
        #print(f"🎯 Action received from RLWorker: {action}")
        # Optionally update UI or send command to execute action

    def on_rl_worker_ready(self):
        self.update_status("✅ RL system ready.")
        self.rl_worker.load_replay_buffer()
        print("🤖 load_replay_buffer.")
        self.rl_worker.load_model()
        print("🤖 load_model.")

    @pyqtSlot(str)
    def update_status(self, message):
        """Update UI status based on RLWorker progress"""
        #print("You can replace this with a QLabel update")  # You can replace this with a QLabel update
        #self.lw_4.clear()
        #self.lw_4.addItem("✅ empty")
        print(message)  # You can replace this with a QLabel update


    def set_training_mode(self, enabled: bool):
        self.training_mode = enabled
        self.rb_training_mode.setChecked(enabled)

        if enabled:
            self.lw_4.addItem("✅ Alive Mode Started.")
            print("Alive Mode Started.")
            
            if self.freedom_ins:
                 self.lw_4.addItem("✅ connection already exists.")
                 print("✅ connection already exists")
            else:
                 self.init_fin()
                 self.lw_4.addItem("✅ Freedom24 connection Established...")
                 print("✅ Freedom24 connection Established...")

            schedule.every(5).minutes.do(self.get_wallet)
            self.lw_4.addItem("✅ Getting wallet every 10 minutes")
   
            schedule.every(5).minutes.do(self.check_status)
            schedule.every(10).minutes.do(self.refresh_status)
            self.lw_4.addItem("✅ so_simple every 5 minute") 

            schedule.every().day.at("23:30").do(self.prepare_all_data)
            print("✅ Get_all_historical_stock_data Started......every().day.at 23:30 ")
            self.lw_4.addItem("✅ Get_all_historical_stock_data Started......every().day.at 23:30 ")
            
            #schedule.every(15).minutes.do(self.save_replay_buffer)
            #self.lw_4.addItem("✅ Every 15 minutes save_replay_buffer." ) 
            #schedule.every(10).minutes.do(self.get_stock_status)
            #self.lw_4.addItem("✅ Every 10 minutes CHECK IF WE HAVE_stocks." ) 
            
            #print("✅ ATrade Started....")
            #self.atrade = ATrade(self.repo)
            #self.atrade.start() 
            #self.lw_4.addItem("✅ ATrade Started....")

            #schedule.every(180).seconds.do(self.check_to_trade)
            #print("✅ Check_to_trade Started.....every(60).seconds")
            #self.lw_4.addItem("✅ Check_to_trade Started.....every(60).seconds")

            #schedule.every().day.at("23:50").do(self.merge_historic_data)
            #print("✅ Merge_historic_data......every().day.at 23:50 ")  
            #self.lw_4.addItem("✅ Merge_historic_data......every().day.at 23:50 ")
            
            #schedule.every().day.at("23:59").do(self.finalized_data_ai)
            #print("✅ Merge_historic_data......every().day.at 23:50 ")  
            #self.lw_4.addItem("✅ Merge_historic_data......every().day.at 23:50 ")

            #schedule.every(3).minutes.do(self.on_bt_f24_2_clicked)
            schedule.run_pending()
            self.telegram_bot.simple_msg("Ειμαστε σε κατασταση λειτουργιας.")
            self.mode = self.repo.get_value("RLworker/mode", "live")
            if self.mode ==  "live":
                self.start_websocket(self.stock_name)
            else:
                pass
        else:
            self.lw_4.addItem("🛑 Alive Mode Stopped.")
            print("Alive Mode Stopped.")
            schedule.clear()
            
            self.mode = self.repo.get_value("RLworker/mode", "live")
            if self.mode ==  "live":
                self.stop_websocket()
            else:
                pass
            
            print("Training stopped.")
            self.telegram_bot.simple_msg("Training stopped.") 




    def refresh_central_data(self):
        """Refresh the QListWidget with all data from Repo."""
        self.lw_central_data.clear()
#        item=QListWidgetItem()
        for key, value in self.repo.get_all_values().items():
            self.lw_central_data.addItem(f"{key}:  {value}")

    def edit_value(self, item):
        """Edit a value in the QListWidget on double-click."""
        key, current_value = item.text().split(":  ", 1)
        new_value, ok = QInputDialog.getText(
            self.lw_central_data, 
            "Edit Value", 
            f"Edit value for {key}:",
            text=current_value)
        if ok:
            self.repo.set_value(key, new_value)

    @pyqtSlot()
    def init_scheduling(self):
            #schedule.every().day.at("23:50").do(self.get_all_historical_stock_data)
            #schedule.every().day.at("00:58").do(self.on_bt_f24_2_clicked)
            #schedule.every(5).minutes.do(self.get_all_historical_stock_data)
            #for hour in range(9, 17):  # Schedule job at every hour from 9 AM to 4 PM
                #schedule.every().day.at(f"{hour}:00").do(job_function)
            #schedule.every(2).minutes.do(self.run_periodic_task2)
            # Create a QTimer to run schedule.run_pending() periodically
            
            schedule.every(10).minutes.do(self.get_stock_status)
            self.lw_4.addItem("✅ Every 10 minutes CHECK IF WE HAVE_stocks." ) 
            
            schedule.every().day.at("09:01").do(self.prepare_api_signals)
            print("✅ Get_from gptapi 10 signals ().day.at 09:01 ")
            self.lw_4.addItem("✅  Get_from gptapi 10 signals.....every().day.at 09:01 ")
            
            self.schedule_timer = QTimer(self)
            self.schedule_timer.timeout.connect(self.run_scheduled_tasks)
            self.schedule_timer.start(6000)  # Check the schedule every second

    def run_scheduled_tasks(self):
        try:    
            schedule.run_pending()
        except Exception as e:
            self.log_real.error(f"Error from schedule occurred: {e}")
           

    @pyqtSlot()
    def refresh_status(self):
        finbotlist=self.freedom_ins.stocklists()
        self.finalist.clear()
        self.cb_spy_buy.clear()
        # Clear previous candidates (optional, if you want to overwrite)
        for key in self.repo.settings.allKeys():
            if key.startswith("candinates/"):
                self.repo.settings.remove(key)

        # Write new values
        for i, symbol in enumerate(finbotlist, start=1):
            key = f"candinates/c{i}"
            self.repo.set_value(key, symbol)
            
        for  candinate in self.repo.get_section_values("candinates"):
            self.finalist.addItem(candinate)
            self.cb_spy_buy.addItem(candinate)
            self.create_ws_file(candinate)

    @pyqtSlot()
    def create_ws_file(self, candinate):
        self.ws_file = os.path.join(self.SPYDATA, f"WS_{candinate}.csv")
        headers = ["Date", "payload"]

        # Create directory if it doesn't exist
        if not os.path.exists(self.SPYDATA):
            os.makedirs(self.SPYDATA)

        # Only create the file with headers if it doesn't exist
        if not os.path.exists(self.ws_file):
            try:
                print("📄 Creating new WebSocket file with headers...")
                with open(self.ws_file, "w", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=headers)
                    writer.writeheader()
            except Exception as e:
                print(f"❌ Failed to create .ws_file: {e}")

  
         
    @pyqtSlot()
    def check_status(self):
        symbol = resolve_stock_name(self.repo) # self.repo.get_value("portfolio/stock", 'TPEIR.GR', str)
        have_stocks = self.repo.get_value("portfolio/have_stocks", 0, str)
        try:
            quote_data = self.freedom_ins.get_stock_quotes(symbol).get(symbol)
            if not quote_data:
                print(f"No quote found for symbol: {symbol}")
                self.repo.set_value("portfolio/stock_price_now", 0)
                return
            price_now = float(quote_data.get("last_price", 0)) 
            self.repo.set_value("portfolio/stock_price_now", price_now)
            
            if have_stocks == "0":
                #action, rl_msg = self.rl_worker.predict_from_dom(self.rl_worker.env.last_dom)
                self.telegram_bot.simple_msg(f"🧐 Υποψηφια Μετοχη: {symbol}💰 Τρεχουσα Τιμη: {price_now}")
                
                #action, rl_msg = self.rl_worker.predict_from_dom(self.rl_worker.env.last_dom)
                #self.telegram_bot.simple_msg(f"🧐 Υποψηφια Μετοχη: {symbol}\n💰 Τρεχουσα Τιμη: {price_now}\n{rl_msg}")
            else:
                self.evaluate_portfolio_position(symbol, price_now)
        except Exception as e:
            print(f"Error retrieving or processing stock quote: {e}")
     
    def evaluate_portfolio_position(self, symbol, price_now):
        quantity = self.repo.get_value("portfolio/q", 0, float)
        price_bought = self.repo.get_value("portfolio/stock_price_bought", 0, float)
        profit_loss = round((price_now - price_bought) * quantity, 2)
        self.repo.set_value("portfolio/profit", profit_loss)
        #action, rl_msg = self.rl_worker.predict_from_dom(self.rl_worker.env.last_dom)
        #self.log_wallet.info(f"Μετοχη: {symbol} | Τιμη: {price_now} | Κερδος: {profit_loss} | Προβλεψη: {rl_msg}")
        #self.telegram_bot.simple_msg(f"💹 Μετοχη: {symbol}\n💰 Τιμη: {price_now}\n📈 Κερδος: {profit_loss}\n{rl_msg}")
        
        self.log_wallet.info(f"Μετοχη: {symbol} | Τιμη: {price_now} | Κερδος: {profit_loss} | Προβλεψη:-")
        self.telegram_bot.simple_msg(f"💹 Μετοχη: {symbol} 💰 Τιμη: {price_now} 📈 Κερδος: {profit_loss}")    
     
    def check_to_trade(self):
        print("Checking for trading actions.........")
        action = self.repo.get_value("portfolio/action", "HOLD", str)  
        symbol = self.repo.get_value("portfolio/stock", 'TPEIR.GR', str)
        quantity = self.repo.get_value("portfolio/q", 1, int)
        
        if action == "SELL":
            profit = self.repo.get_value("portfolio/profit", 0.0, float)
            print(f"Trade detected: SELL {quantity} shares of {symbol}.")
            self.telegram_bot.confirm_trade("SELL", symbol, quantity, profit)

        elif action == "BUY":
            price = self.repo.get_value("portfolio/stock_price_now", 0.0, float)  # Intended buy price
            print(f"Trade detected: BUY {quantity} shares of {symbol}.")
            self.telegram_bot.confirm_trade("BUY", symbol, quantity, price)
        else:
            print("No trade action required. Holding position.")
            
            
    def process_trade_decision(self, decision):
        action = self.repo.get_value("portfolio/action", "HOLD", str)
        if decision == "confirm":
            if action == "BUY":
                print("✅ DEBUG: Executing BUY")
                self.execute_buy()
            elif action == "SELL":
                print("✅ DEBUG: Executing SELL")
                self.execute_sell()
                self.repo.set_value("portfolio/action", "HOLD")  # Reset action
        else:
            print("❌ DEBUG: Trade Canceled")
            self.repo.set_value("portfolio/action", "HOLD")  # Reset action       

    def execute_buy(self):
        """Executes the buy order after confirmation."""
        symbol = self.repo.get_value("portfolio/stock", 'TPEIR.GR', str)
        quantity = self.repo.get_value("portfolio/q", 1, int)
        price = self.repo.get_value("portfolio/stock_price_now", 0.0, float)

        print(f"✅ Buying {quantity} shares of {symbol} at {price:.2f}€.")
        try:
            #self.freedom_ins.buy(symbol, quantity)  # Perform actual buy order
            self.log_wallet.info(f"BOUGHT {quantity} shares of {symbol} at {price:.2f}€.")
            self.repo.set_value("portfolio/stock_price_bought", price)
            self.repo.set_value("portfolio/q", quantity)
            self.repo.set_value("portfolio/profit", 0)  # Reset profit for new buy
            self.repo.set_value("portfolio/action", "HOLD")  # Reset action after execution
        except Exception as e:
            print(f"❌ Error during buy: {e}")

    def execute_sell(self):
        """Executes the sell order after confirmation."""
        symbol = self.repo.get_value("portfolio/stock", 'TPEIR.GR', str)
        quantity = self.repo.get_value("portfolio/q", 1, int)
        profit = self.repo.get_value("portfolio/profit", 0.0, float)
        
        print(f"✅ Selling {quantity} shares of {symbol}. Profit: {profit:.2f}€.")
        try:
            #self.freedom_ins.sell(symbol, quantity)  # Perform actual sell order
            self.log_wallet.info(f"SOLD {quantity} shares of {symbol}. Profit: {profit:.2f}€.")
            self.repo.set_value("portfolio/stock", 0)
            self.repo.set_value("portfolio/stock_price_bought", 0)
            self.repo.set_value("portfolio/stock_price_now", 0)
            self.repo.set_value("portfolio/q", 0)
            self.repo.set_value("portfolio/profit", 0)
            self.repo.set_value("portfolio/action", "HOLD")  # Reset action after execution
        except Exception as e:
            print(f"❌ Error during sell: {e}")
            

        
    @pyqtSlot()
    def on_bt_f24_1_clicked(self):
        self.log_real.info("OK...")
        self.lw_4.clear()
        self.lw_4.addItem("...parseOPQ")
        self.freedom_ins.parseOPQ()
      
        finbotlist=self.freedom_ins.stocklists()  #working great

        self.freedom_ins.market_status("ATHEX")   #working even better
        
        one_quote = self.freedom_ins.get_stock_quotes("TPEIR.GR")  #working
        print("one_quote:\n", one_quote)
        quotes = self.freedom_ins.get_stock_quotes(finbotlist)    #working thats the way i like it
        
        print("quotes:", quotes)

    @pyqtSlot()
    def on_bt_f24_2_clicked(self):
        self.lw_4.clear()
        self.log_real.info("make_market_data...")
        #self.prepare_api_signals()
        
 
        
    @pyqtSlot()
    def on_bt_f24_3_clicked(self):
        self.lw_4.clear()
        self.lw_4.addItem("get_wallet") 
        self.get_wallet()   
 
    def on_reward(self, reward):
        print(f"🎯 Received reward: {reward}")
        # Optional: Add to training data, logging, or retraining trigger
        self.lw_4.addItem(f"🎯 Reward received: {reward:.2f}")         
        
    def start_websocket(self, stock):
        self.lw_4.addItem("Starting WebSocket's...")#?api_key=xxx&access_token=xxxx"
        print("WebSocketWorker stock:", stock)
        self.websocket_worker = WebSocketWorker(self.freedom_ins, stock)
        #self.websocket_worker = WebSocketWorker(f"wss://wss.tradernet.com/?SID={self.sid}", self.freedom_ins)
        self.websocket_worker.websocket_data.connect(self.websocket_data_received)
        
        #self.websocket_worker.websocket_data.connect(self.rl_worker.handle_dom_update)
        
        self.websocket_worker.websocket_error.connect(self.websocket_handle_error)
        self.websocket_worker.websocket_close.connect(self.websocket_handle_close)       
        self.websocket_worker.start()

    def stop_websocket(self):
        self.lw_4.clear()
        self.lw_4.addItem("Stopping WebSocket's...")
        self.websocket_worker.stop()

    def websocket_handle_error(self, error):
        print("Error received:", error)
        self.lw_4.addItem(f"Error: {error}")

    def websocket_handle_close(self, close_msg):
        print("Connection closed:", close_msg)
        self.lw_4.addItem(f"Closed: {close_msg}")
        
    def websocket_data_received(self, data):
        print("📡Websocket data:" , data)
        # Καταγραφή raw WebSocket
        symbol = data["i"]
        self.save_ws_data_full(symbol, data)
        
        if hasattr(self, "env"):
            self.rl_worker.env.last_dom = data
        if not data or "i" not in data:
            print("⚠️ Invalid WebSocket payload.")
            return
  
        if not self.rl_worker or self.rl_worker.model is None:
            print("❌ RL system not ready.")
            return
        #action, label = self.rl_worker.predict_from_dom(data)
        #print("🤖 Prediction:", label)


    def save_ws_data_full(self, symbol, data):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            filename = Path(self.SPYDATA) / f"WS_{symbol}.csv"
            is_new = not filename.exists()

            row = {
                "Date": timestamp,
                "payload": json.dumps(data)}

            with open(filename, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=row.keys())
                if is_new:
                    writer.writeheader()
                writer.writerow(row)
            print(f"📝 Full WebSocket data saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save full WebSocket data for {symbol}: {e}")

    @pyqtSlot()
    def on_bt_conn_to_f24_clicked(self): # on Initialized MODE
        self.init_fin()
        
 
    @pyqtSlot()
    def on_bt_initialize_clicked(self):
        filepath = self.load_csv()
        if filepath:
            parsed_text = self.parse_market_csv(filepath)
            self.save_output(parsed_text)

    def load_csv(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        return filepath  # Return the path if selected


    def parse_market_csv(self, filepath):
        parsed_lines = []

        with open(filepath, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue

                timestamp = row[0]
                try:
                    data = json.loads(row[1])
                except json.JSONDecodeError:
                    continue

                lines = [f"\nTimestamp: {timestamp}",
                         f"Record #: {data.get('n')}",
                         f"Instrument: {data.get('i')}",
                         f"Count: {data.get('cnt')}, Extra: {data.get('x')}"]

                if data.get('del'):
                    lines.append("Deleted orders:")
                    for item in data['del']:
                        lines.append(f"  - Price: {item['p']}, Key: {item['k']}")

                if data.get('ins'):
                    lines.append("Inserted orders:")
                    for item in data['ins']:
                        lines.append(f"  - Price: {item['p']}, Side: {item['s']}, Qty: {item['q']}, Key: {item['k']}")

                if data.get('upd'):
                    lines.append("Updated orders:")
                    for item in data['upd']:
                        lines.append(f"  - Price: {item['p']}, Side: {item['s']}, Qty: {item['q']}, Key: {item['k']}")

                parsed_lines.extend(lines)

        return "\n".join(parsed_lines)

    def save_output(self, text):
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Output", "parsed_output.txt", "Text Files (*.txt)")
        if save_path:
            with open(save_path, 'w') as f:
                f.write(text)
                
    @pyqtSlot()
    def on_bt_stop_clicked(self): # on Initialized MODE
        self.rl_worker.stop()
        
    @pyqtSlot()
    def on_bt_train_clicked(self): # on Initialized MODE
        self.rl_worker.set_mode()
        
    @pyqtSlot()
    def on_bt_new_index_clicked(self):
        """Slot documentation goes here."""
        self.lw_4.clear()
        self.lw_4.addItem("Empty")
        
    @pyqtSlot()
    def on_bt_spy_update_clicked(self):
        """Slot documentation goes here."""
        self.lw_4.clear()
        self.lw_4.addItem("Empty")
    
    @pyqtSlot()
    def on_bt_1_clicked(self):
        """Slot documentation goes here. """
        self.lw_4.clear()
        self.lw_4.addItem("Empty")

 
    @pyqtSlot()
    def on_bt_2_clicked(self):
#        from main import BASE_DIR, SETTINGS
#        print("BASE_DIR=", BASE_DIR)
        
        test1=getfilepath("log")
        print("test1=", test1)
        
        test2=getfilepath("log", "xxxx.txt")
        print("test2=", test2)
        
        test3=getfilepath("spy", "xxxx.txt")
        print("test3=", test3)
        
        test4=getfilepath("spy", "xxxx.txt", new=True)
        print("test4 file=", test4)
        
        test5=getfilepath("ai", "ai___xxxx.txt", new=True)
        print("test4 file=", test5)

    @pyqtSlot()
    def on_bt_3_clicked(self):
        #stock_name = self.repo.get_value("portfolio/stock")  
        if self.repo.get_value("portfolio/stock") =="0":
            stock_name =  self.repo.get_section_values("candinates")[0]
            print("stock_name_1", stock_name)
        model_type="PPO"
        self.predict(model_type=model_type, stock_name=stock_name, save_dir=self.AI_models)
        
    @pyqtSlot()
    def on_bt_4_clicked(self):
        self.rl_worker.load_model() 
   
    @pyqtSlot()
    def on_bt_5_clicked(self):
        self.show_tensorboard()    
        
    @pyqtSlot()
    def on_bt_6_clicked(self):
        #self.lw_4.addItem("make_prediction. from loading the model")
       #self.make_prediction()
       self.lw_4.addItem("✅ Predict from the Trained Policy. from last_shoot")
       #self.predict() 
       self.last_shoot()
       
    @pyqtSlot()
    def last_shoot(self):
        md = MarketData(
            symbol=self.stock_name)
        md.reset_pointer()
        
        self.rl_worker.market_data=md
        self.rl_worker.run_live_loop()

    @pyqtSlot()
    def make_prediction(self):
        """Predict the best action using the trained model"""
        stock_name = self.repo.get_value("portfolio/stock")
        if stock_name == "0":
            candidates = self.repo.get_section_values("candinates")
            stock_name = candidates[0] if candidates else "TPEIR.GR"
            print("stock_name_1", stock_name)

        # ✅ Use the environment from the active RL processor
        if hasattr(self, "rl_worker") and hasattr(self.rl_worker, "env") and self.rl_worker.env is not None:
            try:
                action = self.rl_worker.predict()  # Your env must have a .predict() method
            except Exception as e:
                print(f"❌ Prediction error: {e}")
                return

            if action == 0:
                print(f"🤖 Predicted Action for {stock_name}: BUY ({action})")
            elif action == 1:
                print(f"🤖 Predicted Action for {stock_name}: HOLD ({action})")
            elif action == 2:
                print(f"🤖 Predicted Action for {stock_name}: SELL ({action})")
            else:
                print(f"🤖 Unknown Action: {action}")
        else:
            print("❌ No trained model or environment found! Train the model first.")
               
               
    def predict(self, obs=None, model_type="PPO", stock_name="unknown_stock", save_dir="models"):
        """Predict action using the last saved model for a given stock."""
        stock_name = self.repo.get_value("portfolio/stock")
        if self.repo.get_value("portfolio/stock") =="0":
            stock_name =  self.repo.get_section_values("candinates")[0]
            print("stock_name_1", stock_name)
        
        model_path = f"{save_dir}/{stock_name}_{model_type}_latest.zip"
        print(f" we are using {model_path}.")
        
        if not os.path.exists(model_path):
            print(f"❌ No saved model found for {stock_name}. Train a model first.")
            return None
        print(f"📥 Loading model from {model_path}...")
        # Load the model dynamically based on model type
        if model_type == "PPO":
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
        elif model_type == "A2C":
            from stable_baselines3 import A2C
            model = A2C.load(model_path)
        elif model_type == "DQN":
            from stable_baselines3 import DQN
            model = DQN.load(model_path)
        else:
            raise ValueError("❌ Unsupported model type!")
        print(f"📥 using model  {model_type}")

        #obs = self.prepare_observation()
        #if obs is None:
            #print("❌ Could not prepare observation. Prediction aborted.")
            #return None
        # Αν δεν δόθηκε obs, προετοίμασέ το από CSV
        if obs is None:
            obs = self.prepare_observation()
            if obs is None:
                print("❌ Αδυναμία δημιουργίας παρατήρησης.")
                return None
            
        obs = self.normalize_observation(obs)  
        action, _ = model.predict(obs)
        if action==0:
            print(f"🤖 Predicted Action for {stock_name}: BUY {action}")
        elif action==1:
            print(f"🤖 Predicted Action for {stock_name}: HOLD {action}")
        elif action==2:
            print(f"🤖 Predicted Action for {stock_name}: SELL {action}")
        return action
        
    def prepare_observation(self):
        """Prepares the observation either from the env or from the raw DataFrame."""
        stock_name = self.repo.get_value("portfolio/stock")
        if self.repo.get_value("portfolio/stock") =="0":
            stock_name =  self.repo.get_section_values("candinates")[0]
            print("stock_name_1", stock_name)
        try:
            if hasattr(self, "env") and self.env is not None:
                # ✅ If we have a live environment
                obs = self.env._next_observation()
                print("📈 Using live environment observation!")
            else:
                # ❗ If we don't have live env, fallback to manual construction
                print("📂 No environment found! Preparing observation from raw data...")
                df = pd.read_csv(f"{self.AI}/models/{stock_name}.csv").copy()
                #df = self.df.copy()
                
                # Use last 'window_size' rows like in training
                window_size = 10
                if len(df) < window_size:
                    raise ValueError("Not enough data for observation!")
                frame = df.iloc[-window_size:]
                obs = np.concatenate([
                    frame["close"].values,
                    frame["SMA_10"].values,
                    frame["EMA_10"].values,
                    frame["volume"].values,
                    frame["RSI"].values,
                    frame["MACD_12_26_9"].values,
                    frame["MACD_Signal"].values,
                    frame["VWAP"].values,
                    frame["STOCHk_14_3_3"].values,
                    frame["STOCH_SIGNAL"].values
                ]).astype(np.float32)
                obs = obs.flatten()
            return obs
        except Exception as e:
            print(f"❌ Error preparing observation: {e}")
            return None

    @pyqtSlot()
    def on_bt_7_clicked(self): 
      self.lw_4.addItem("✅ empty")
      pass
       
    @pyqtSlot()
    def on_bt_8_clicked(self):
        self.lw_4.addItem("✅ empty")
        pass 



        
    @pyqtSlot(object)
    def training_done(self, trained_env):
        """Save the trained environment and allow predictions"""
        self.env = trained_env  
        print("✅ RL Training Finished! You can now make predictions.")
        
    @pyqtSlot()
    def on_bt_9_clicked(self):
        #data visualization
        self.mpl_widget.canvas.axes.cla() 
        self.mpl_widget.canvas.axes.axis([0, 10, 0, 10]) 
        self.mpl_widget.canvas.axes.text(1, 5, 'Finance calculation wait a moment.', style='italic',
        bbox={'facecolor': 'blue', 'alpha': 0.4, 'pad': 10})
        self.mpl_widget.canvas.draw() 

        data_to_plt, selectedFilter = QFileDialog.getOpenFileName(
            self,
            self.tr("choose stock"),
            "C:/Users/User/Desktop/Fin_Folder/FIN_DATA/spy", 
            "",
            None
        )
        print("data_to_plt=", data_to_plt)

        data = pd.read_excel(data_to_plt)
        print("DATA=", data)
        data=data.iloc[::-1]
        data = data.reset_index(drop=True)
        print("DATA=", data)
        prices = data.iloc[:, 1]
        name=os.path.basename(data_to_plt)[:-5]

        self.mpl_widget.canvas.axes.cla() 
        self.mpl_widget.canvas.axes.plot(prices)
        self.mpl_widget.canvas.axes.set_xlabel('Index')
        self.mpl_widget.canvas.axes.set_ylabel('Price')
        self.mpl_widget.canvas.axes.set_title(name+'  Plot')
        self.mpl_widget.canvas.draw() 

    #@pyqtSlot()
    #def import_csv_file(self, file_path):
        ## Read the CSV file
        #df = pd.read_csv(file_path)
        #try:
                ## Create a new column for the modified date format
##            df['Modified_Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
##            df['Modified_Date'] = df['Modified_Date'].dt.strftime('%Y-%m-%d')
            ## Convert date column to the desired format
            #df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
            #df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            #print("date")
        ## Remove commas from the 'Volume' column
##            df['Volume'] = df['Volume'].str.replace(',', '')
            #df['Volume'] = pd.to_numeric(df['Volume'].str.replace(',', ''), errors='coerce')
        #
            ## Save the modified DataFrame as a CSV file
            #df.to_csv(f'{file_path[:-4]}_ai_.csv', index=False)
        #except:
            #pass
        #return df
 
    @pyqtSlot()
    def on_bt_buy_clicked(self):
        self.lw_4.clear()
        self.lw_4.addItem("✅ empty button buy")
        print("empty.....we have a buy worker")

    @pyqtSlot()
    def on_bt_sell_clicked(self):
        self.lw_4.clear()
        self.lw_4.addItem("✅ empty button sell")
        print("empty.....we have a sell worker")

    def on_task_completed(self, future):
        result = future.result()
        #if isinstance(result, DataFrame):
                ## Convert the DataFrame to a string and add it as an item to the QListWidget.
                #data_string = result.to_string()
                #self.lw_4.addItem(data_string)
        if isinstance(result, dict):
            self.lw_4.addItem("its a dictionary ....")
            #print("result_dictionary.... =", result )
            data_string = json.dumps(result, indent=4)
            #print("data_string =", data_string )
            self.lw_4.addItem(data_string)
        elif isinstance(result, tuple):
               self.lw_4.addItem("It's a tuple ....")
               try:
                   current_money_str, pos_value_str = result
                   current_money = float(current_money_str)
                   pos_value = float(pos_value_str)
                   total=float(current_money_str)+float(pos_value_str) 
                   self.lcd_1.display(total)
                   self.lcd_2.display(current_money)
                   self.lcd_3.display(pos_value)
                   self.lcd_4.display(pos_value)
                   self.lcd_5.display(pos_value)
                   self.lcd_6.display(pos_value)

               except (ValueError, TypeError):
                   print("Invalid current_money or pos_value. Cannot convert to numeric values.")     

        else:
                # Handle the case when the result is not a DataFrame (optional).
                self.lw_4.addItem(result)
                self.lw_4.addItem("Task completed ....")
        print("on_task_completed!!!!!")
        
    @pyqtSlot()
    def on_bt_open_data_clicked(self):
        """Slot documentation goes here."""        
        os.startfile(self.folder_export)
        
    @pyqtSlot()
    def on_bt_home_clicked(self):
        """Slot documentation goes home"""
        self.telegram_bot.stop()
        from Main.Ui.m_body import MainWindow
        self.home=MainWindow()
        self.home.show()
        self.close()
        
    @pyqtSlot()
    def init_fin(self):
        self.log_real.info("#######################  initialization of fin   ##############################")        
        if self.freedom_ins:
             self.lw_4.addItem("OK..Freedom24_connection...already exist.")
             self.log_real.debug("OK..Freedom24_connection....already exist.")
             self.bt_conn_to_f24.setStyleSheet("background-color: green;")
        else:
             self.freedom_ins=Freedom(self.repo)
             self.lw_4.addItem("Connection failed with Freedom24.and retry to init the instance")
             self.log_real.debug("Connection failed with Freedom24.and retry to init the instance.")
             
        
    @pyqtSlot()
    def get_wallet(self):  # on Initialized MODE
        self.lw_4.clear()
        self.lw_4.addItem("Task in progress...We are getting your wallet from Freedom24.")
        task_runner = TaskRunner()
        task_runner.run_hard(self.freedom_ins.get_wallet_position_status, self.wallet_data) 


    def wallet_data(self, future):
        """Handle wallet and position data once received."""
        result = future.result()
        account_info = result["result"]["ps"]["acc"][0]
        self.w_total = float(account_info["s"])
        
        # Display total wallet value
        self.lcd_total_wallet.display(self.w_total)
        self.lw_4.addItem(f"Έχεις {self.w_total} ευρώ συνολο.")
        self.repo.set_value("wallet/total", self.w_total)

        positions = result["result"]["ps"].get("pos", [])
        if not positions:
            self._clear_position_data()
            self.lw_4.addItem("Not working money")
            print("⚠️ No active positions found.")
            return

        # Position present
        pos = positions[0]
        self._update_position_data(pos, account_info)
        self.lw_4.addItem(f"Έχεις {self.total_wallet} ευρώ συνολο, και {self.price_now} σε μετοχές.")
        
        self.lw_4.addItem("✅ OK...Wallet Task completed.")
        self.log_real.info("OK...Wallet initialization, getting current status.")

    def _clear_position_data(self):
        """Reset values when no position exists."""
        self.repo.set_value("wallet/free", 0)
        self.repo.set_value("portfolio/stock", 0)
        self.repo.set_value("portfolio/stock_price_now", 0)
        self.repo.set_value("portfolio/stock_price_bought", 0)
        self.repo.set_value("portfolio/q", 0)
        self.repo.set_value("portfolio/market", 0)
        self.repo.set_value("portfolio/profit", 0)
        self.repo.set_value("portfolio/have_stocks", 0)

    def _update_position_data(self, pos, account_info):
        """Update local state and UI with position data."""
        self.cash = float(account_info.get("s", 0))
        self.price_now = float(pos.get("market_value", 0))
        self.price_bought = float(pos.get("s", 0))
        self.temaxia = float(pos.get("q", 0))
        self.bought_price = float(pos.get("bal_price_a", 0))
        self.current_price = float(pos.get("mkt_price", 0))
        self.Closing = float(pos.get("close_price", 0))
        self.profit = float(pos.get("currval", 0))

        self.total_wallet = round(self.price_now + self.cash + self.profit, 2)
        self.fees = round((self.temaxia * 0.02 + 2) * 2, 2)

        self.w_spy = pos.get("i", "")
        self.acc_pos_id = pos.get("acc_pos_id", "")
        self.instr_id = pos.get("instr_id", "")

        # Update UI
        self.lcd_total_wallet.display(self.total_wallet)
        self.lcd_cash.display(self.cash)
        self.lcd_price_now.display(self.price_now)
        self.lcd_price_bought.display(self.price_bought)
        self.lcd_profit.display(self.profit)
        self.lcd_temaxia.display(self.temaxia)
        self.lcd_bought_price.display(self.bought_price)
        self.lcd_current_price.display(self.current_price)
        self.lcd_fees.display(self.fees)
        self.txt_w_spy.setText(self.w_spy)
        self.txt_Instr_ID.setText(str(self.instr_id))
        self.txt_ID.setText(str(self.acc_pos_id))
        self.txt_Closing.setText(str(self.Closing))

        # Save to repo
        self.repo.set_value("wallet/total", self.total_wallet)
        self.repo.set_value("wallet/free", self.cash)
        self.repo.set_value("portfolio/stock", self.w_spy)
        self.repo.set_value("portfolio/stock_price_bought", self.bought_price)
        self.repo.set_value("portfolio/q", self.temaxia)
        self.repo.set_value("portfolio/market", pos.get("ltr"))
        self.repo.set_value("portfolio/have_stocks", 1)

    
    def wallet_data2(self, future):   # when the data is back 
        result = future.result()
        self.w_total=result["result"]["ps"]["acc"][0]["s"]
        self.lcd_total_wallet.display(float(self.w_total))
        self.lw_4.addItem(f"Εχεις {self.w_total} ευρω συνολο.")
        self.repo.set_value("wallet/total", self.w_total)
 
        if result["result"]["ps"]["pos"]:
            pos_data = result["result"]["ps"]["pos"][0]
            print("pos_data", pos_data)   
            self.profit = pos_data["currval"]         
            
            total_wallet = float(pos_data["market_value"])+float(result["result"]["ps"]["acc"][0]["s"]+float(self.profit))

            self.total_wallet=round(total_wallet, 2)
            
            self.cash =   result["result"]["ps"]["acc"][0]["s"]
            self.price_now = pos_data["market_value"]
            self.price_bought= pos_data["s"]
            self.temaxia = pos_data["q"]
            self.bought_price = pos_data["bal_price_a"]
            self.current_price = pos_data["mkt_price"]
            self.Closing = pos_data["close_price"]
            
            self.w_spy= pos_data["i"]
            self.acc_pos_id = pos_data["acc_pos_id"]
            self.instr_id = pos_data["instr_id"]

            self.fees=(int(self.temaxia)*0.02+2)*2
            
            self.lcd_total_wallet.display(float(self.total_wallet))
            self.lcd_cash.display(float(self.cash))
            self.lcd_price_now.display(float(self.price_now))
            self.lcd_price_bought.display(float(self.price_bought))
            self.lcd_profit.display(float(self.profit))
            self.lcd_temaxia.display(float(self.temaxia))
            self.lcd_bought_price.display(float(self.bought_price))
            self.lcd_current_price.display(float(self.current_price))
            self.lcd_fees.display(float(self.fees))
            self.txt_w_spy.setText(self.w_spy)
            self.txt_Instr_ID.setText(str(self.instr_id))
            self.txt_ID.setText(str(self.acc_pos_id))
            self.txt_Closing.setText(str(self.Closing))
            self.lw_4.addItem(f"Εχεις {self.total_wallet} ευρω συνολο, και {self.price_now} σε μετοχες.")

            self.repo.set_value("wallet/total", self.total_wallet)
            self.repo.set_value("wallet/free", self.cash)
            self.repo.set_value("portfolio/stock", pos_data.get("i"))
            self.repo.set_value("portfolio/stock_price_bought", float(pos_data.get("bal_price_a")))
            self.repo.set_value("portfolio/q", float(pos_data.get("q")))
            self.repo.set_value("portfolio/market", pos_data.get("ltr"))
            self.repo.set_value("portfolio/have_stocks", 1)
        else:
             self.repo.set_value("wallet/free", 0)
             self.repo.set_value("portfolio/stock",0)
             self.repo.set_value("portfolio/stock_price_now", 0)
             self.repo.set_value("portfolio/stock_price_bought", 0)
             self.repo.set_value("portfolio/q", 0)
             self.repo.set_value("portfolio/market", 0)
             self.repo.set_value("portfolio/profit", 0)
             
             self.lw_4.addItem("Not working money")
             print("Failed...Invalid current_money or pos_value.... or  Not working money.") 

        self.lw_4.addItem(" ✅ OK...Wallet Task completed.")
        self.log_real.info("OK..Wallet initialition ,getting current status.")
        
    @pyqtSlot()
    def get_stock_status(self): 
        # Get the currently held stock (if any)
        self.bstock_name = self.repo.get_value("portfolio/stock", "0")
        # Determine the active stock to use
        if self.bstock_name == "0" or not self.bstock_name.strip():
            # If no stock is currently held, use the first candidate
            self.log_real.info(f"Checking if we have bought stocks self.bstock_name = {self.bstock_name}")
            candidates = self.repo.get_section_values("candinates")
            self.stock_name = candidates[0] if candidates else "TPEIR.GR"  # fallback
        else:
            self.stock_name = self.bstock_name
            
    @pyqtSlot()
    def save_replay_buffer(self):
        if hasattr(self, "rl_worker") and hasattr(self.rl_worker, "save_replay_buffer"):
            self.rl_worker.save_replay_buffer()
        else:
            print("⚠️ No active RL worker or method missing.")



    def toggle_mode(self):
        new_mode = self.repo.get_value("RLworker/mode", "live")
        self.rl_mode.setText(new_mode)
        self.init_RL() 

    @pyqtSlot()
    def show_tensorboard(self):
        """Launch TensorBoard inside PyQt"""
        try:
            self.tensorboard_proc = QProcess(self)
            #python -m tensorboard.main --logdir=C:\Users\User\Desktop\Fin_Folder\05.TENSORBOARD\logs\fit
            self.tensorboard_proc.setProgram(r"C:/02.FINTECH/FINBOT_ENV/Scripts/tensorboard.exe")
            self.tensorboard_proc.setArguments([f"--logdir={self.TENSORBOARD}/logs/fit"])
            self.tensorboard_proc.setWorkingDirectory(self.TENSORBOARD)
            #self.tensorboard_proc.setProgram("python")
            #self.tensorboard_proc.setArguments(["-m", "tensorboard.main", f"--logdir={self.TENSORBOARD}/logs/fit", "--port=6006"])
            self.tensorboard_proc.start()
            self.lw_4.addItem("✅ TensorBoard started at: http://localhost:6006/")
            self.log_real.info("TensorBoard started at: http://localhost:6006/")
            QDesktopServices.openUrl(QUrl("http://localhost:6006/"))
        except Exception as e:
            self.log_real.info(f"TensorBoard Error: {e}")
            self.lw_4.addItem(f"TensorBoard Error: {e}")
            
    @pyqtSlot()   
    def training_pipeline_toggle(self):
        self.set_training_mode(self.rb_training_mode.isChecked()) 
        
    @pyqtSlot()
    def prepare_all_data(self):            
        self.lw_4.clear()
        self.lw_4.addItem("...Getting_stock_data....")
        self.log_real.info("OK...Getting_stock_data...")
        task_runner = TaskRunner2()
        task_runner.run_hard(self.dc.get_all_historical_data())
        self.lw_4.addItem("Ok getting the data...")

    @pyqtSlot()
    def prepare_api_signals(self):
        self.lw_4.clear()
        self.lw_4.addItem("📡 Getting GPT signals...")
        candidates = self.repo.get_section_values("candinates")
        task_runner = TaskRunner2()
        for symbol in candidates:
            self.log_real.info(f"Getting GPT signals...for {symbol}")
            fetcher = SignalFetcher(symbol, test_mode=False)
            task_runner.run_hard(fetcher.save_signals())
        self.lw_4.addItem("✅ Submitted tasks to fetch GPT signals.")
        
    def closeEvent(self, event):
        print("🚪 App closing. Stopping all threads...")
        self.stop_all_threads()
        event.accept()
        
    def stop_all_threads(self):
        if self.telegram_bot:
            print("🛑 Stopping Telegram thread...")
            self.telegram_bot.stop()
        if self.websocket_worker:
            self.websocket_worker.stop()
        if hasattr(self, 'schedule_timer') and self.schedule_timer.isActive():
            self.schedule_timer.stop()
            print("⏹️ Schedule timer stopped.")
            
        # 🧠 Σταμάτησε RL worker αν τρέχει
        if hasattr(self, 'rl_worker') and self.rl_worker:
            print("🧠 Stopping RL worker...")
            self.rl_worker.save_replay_buffer()
            self.rl_worker.save_model()
            self.rl_worker.stop()
            self.rl_worker = None

        if hasattr(self, 'rl_thread') and self.rl_thread:
            if self.rl_thread.isRunning():
                self.rl_thread.quit()
                self.rl_thread.wait()
            self.rl_thread.deleteLater()
            self.rl_thread = None
            
        if schedule.jobs:
            schedule.clear()
            print("🧼 Schedule jobs cleared.")
            

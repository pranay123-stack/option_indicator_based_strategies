import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
from datetime import datetime, timedelta




# Set up logging
logging.basicConfig(level=logging.INFO, filename='/Users/pranaygaurav/Downloads/AlgoTrading/Option_Trading_Strategies/EMA_RSI/Logs/ems_rsi_backtest_log.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')









def calculate_sma(df, column):
    """
    Calculate Simple Moving Average (SMA) for the selected rows.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    column (str): Name of the column to calculate SMA for.

    Returns:
    float: SMA value.
    """
    return df[column].mean()

def initial_ema(df, column, period, time_str):
    """
    Calculate the initial EMA value using the SMA of the previous data.

    Args:
    df (pd.DataFrame): DataFrame containing the previous data.
    column (str): Name of the column to calculate EMA for.
    period (int): Number of periods for EMA calculation.
    time_str (str): Time in 'HH:MM:SS' format to start filtering from.

    Returns:
    float: Initial EMA value.
    """
    filtered_df = filter_df_by_time_and_period(df, time_str, period)
    ema = calculate_sma(filtered_df, column)
    return round(ema,2)

def calculate_rsi(df, column='Close', periods=14):
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    rsi_series = pd.Series(index=df.index, dtype=float)
    
    avg_gain = gain.iloc[:periods].mean()
    avg_loss = loss.iloc[:periods].mean()
    
    if avg_loss == 0:
        rsi_series.iloc[periods-1] = 100
    else:
        rs = avg_gain / avg_loss
        rsi_series.iloc[periods-1] = round(100 - (100 / (1 + rs)),2)

    for i in range(periods, len(df)):
        avg_gain = (avg_gain * (periods - 1) + gain.iloc[i]) / periods
        avg_loss = (avg_loss * (periods - 1) + loss.iloc[i]) / periods
        
        if avg_loss == 0:
            rsi_series.iloc[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi_series.iloc[i] = round(100 - (100 / (1 + rs)),2)

    return rsi_series

def calculate_ema(df, column, period, previous_ema):
    
    
    alpha = 2 / (period + 1)
    
    ema_series = pd.Series(index=df.index, dtype=float)
    ema_series.iloc[0] = previous_ema

    for i in range(1, len(df)):
        current_close = df.iloc[i][column]
        ema_series.iloc[i] = round((current_close - ema_series.iloc[i - 1]) * alpha + ema_series.iloc[i - 1],2)

    return ema_series

def extract_symbol_and_expiry(ticker):
    import re
    match = re.match(r'([A-Z]+)(\d{2}[A-Z]{3}\d{2})', ticker)
    if match:
        symbol = match.group(1)
        backtest_config['Underlying_Symbol'] = symbol
        expiry = match.group(2)
        return symbol + expiry, symbol
    return None, None



def filter_df_by_time_and_period(df, time_str, period):
    """
    Filter the DataFrame from a specific time backwards for a given period.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    time_str (str): Time in 'HH:MM:SS' format to start filtering from.
    period (int): Number of periods to go back from the specified time.

    Returns:
    pd.DataFrame: Filtered DataFrame.
    """
    # Ensure the 'Time' column is in datetime format
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
    # Convert the specified time string to datetime.time
    target_time = pd.to_datetime(time_str, format='%H:%M:%S').time()
    
    # Find the index of the row with Time equal to the target time
    target_index = df[df['Time'] == target_time].index[0]
    
    # Filter the DataFrame for the specified time range
    filtered_df = df.iloc[max(0, target_index - period + 1): target_index + 1]
    return filtered_df

def update_entrylegs_details(csv_file_path, entrylegs_details):
    df = pd.read_csv(csv_file_path)
    if not df.empty:
        sample_ticker = df.iloc[0]['Ticker']
        symbol_expiry, symbol = extract_symbol_and_expiry(sample_ticker)
        if symbol_expiry and symbol:
            for leg in entrylegs_details:
                leg['Ticker_Symbol'] = symbol_expiry
                if symbol == 'NIFTY':
                    leg['lot_size'] = 50
                elif symbol == 'BANKNIFTY':
                    leg['lot_size'] = 15
    return entrylegs_details


def atm_at_time_t(ticker_df, entry_time, symbol):
    # Ensure the 'Time' column is of type datetime.time
    ticker_df['Time'] = pd.to_datetime(ticker_df['Time'], format='%H:%M:%S').dt.time
    
    # Ensure entry_time is of type datetime.time
    entry_time = datetime.strptime(entry_time, '%H:%M:%S').time()

    # print("TICKER DF", ticker_df)  # Show the first few rows for verification
    # print("ENTRY TIME", entry_time)
    # print("SYMBOL", symbol)

    # # Debug: Check unique times in the DataFrame
    # print("Unique times in ticker_df:", ticker_df['Time'].unique())
    
    # Filter the DataFrame based on 'Time' and 'Ticker'
    atm_row = ticker_df[(ticker_df['Time'] == entry_time) & (ticker_df['Ticker'].str.contains(symbol))]

    # print("ATM ROW", atm_row)
    
    if atm_row.empty:
        raise ValueError(f"No data available for symbol {symbol} at time {entry_time}")
    
    atm = int(round(atm_row.iloc[0]['ATM_close'] / 100) * 100)
    return atm

def ticker_row_at_time(ticker_df, symbol_to_search, time):
    ticker_df['Time'] = pd.to_datetime(ticker_df['Time'], format='%H:%M:%S').dt.time
    time = datetime.strptime(time, '%H:%M:%S').time()
    
    # print(f"Searching for Ticker: {symbol_to_search} at Time: {time}")
    result = ticker_df[(ticker_df['Ticker'] == symbol_to_search) & (ticker_df['Time'] == time)]
    
    # print("Ticker Row Entry:", result)
    return result


def get_ohlc_ticker_row_at_time_t(df, ticker_row_at_time,ohlc_column):
    if ticker_row_at_time.empty:
        return None  # Or some other indication that no data was found

    if ohlc_column == 'High':
       return ticker_row_at_time.iloc[0]['High']
    elif ohlc_column == 'Low':
        return ticker_row_at_time.iloc[0]['Low']
    elif ohlc_column == 'Close':
        return ticker_row_at_time.iloc[0]['Close']
    elif ohlc_column == 'Open':
        return ticker_row_at_time.iloc[0]['Open']
    else:
        return None

def place_order_at_atm_price(ticker_df,symbol,entry_time, leg):


    atm = atm_at_time_t(ticker_df,entry_time,symbol)
    # print("ATM VALUE ===",atm)
    strike = atm+ leg['offset']
    # print("strike ===",strike)
   

   
    symbol_to_search = f"{leg['Ticker_Symbol']}{strike}{leg['type']}.NFO"

    
    # print("symbol_to_search ===",symbol_to_search)
    ticker_row_entry = ticker_row_at_time(ticker_df, symbol_to_search, entry_time)
    # print("ticker_row_entry ===",ticker_row_entry)
    entry_price =  get_ohlc_ticker_row_at_time_t(ticker_df,  ticker_row_entry ,"Close")
    # print("entry_price ===",entry_price)
    
    sl_multiplier = 1+(backtest_config['stoploss_percentage'] /100)
   



    tl_multiplier = backtest_config['target_percentage'] / 100
    # value = entry_price * tl_multiplier
    # target_price = entry_price - value
    # print("target_price ===",target_price)
  
    
  
    
    return {
        'symbol_to_search': symbol_to_search,
        'entry_type':"Buy",
        'entry_time': entry_time,
        'entry_price': entry_price,
        'Stoploss_price': None
    }


def buy_atm_option_and_check_sl_hit_or_squareoff(legs,indexentryprice,Underlying_symbol,ticker_df, option_type,entry_time,stoploss_percentage):
  
   
   
    
    # Define square off time
    square_off_time = backtest_config['squareoff_time']

    if isinstance( square_off_time , str):
        square_off_time = datetime.strptime(square_off_time, '%H:%M:%S').time()
    
  
    if option_type == 'CE':
        entry_order_details =place_order_at_atm_price(ticker_df,Underlying_symbol,entry_time,legs[0])
  

        trade_details = {
            'TickerSymbol': entry_order_details['symbol_to_search'],
            'EntryTime': entry_order_details['entry_time'],
            'ExitTime': None,
            'EntryType': entry_order_details['entry_type'],
            'ExitType': "",
            'EntryPrice': entry_order_details['entry_price'],
            'Initial_Stoploss': entry_order_details['entry_price']*0.5,
            'Modified_Stoploss': entry_order_details['entry_price']*0.5,
            'ExitPrice': None,
            'pnl': None
        }

        logging.info(f"CE position took{trade_details} ")

        if isinstance(trade_details['EntryTime'], str):
                trade_details['EntryTime'] = datetime.strptime(trade_details['EntryTime'], '%H:%M:%S').time()
            

        ticker_df['Time'] = pd.to_datetime(ticker_df['Time'], format='%H:%M:%S').dt.time
         
        entry_time = datetime.strptime(entry_time, '%H:%M:%S').time()
                # Initialize the previous price
        
        
    
        old_price = entry_order_details['entry_price']
       
        filtered_ticker_df = ticker_df[(ticker_df['Time'] > entry_time) & (ticker_df['Ticker'] == entry_order_details['symbol_to_search'])]

        for i in range(len(filtered_ticker_df)):
            current_time = filtered_ticker_df.iloc[i]['Time']
            current_price = filtered_ticker_df.iloc[i]['Close']
            current_high = filtered_ticker_df.iloc[i]['High']
            current_low = filtered_ticker_df.iloc[i]['Low']

            # print(f"Current time: {current_time}, Current price: {current_price}")

            # Update trailing stop-loss if the current price moves favorably by 50%
            if current_price > old_price:
                       # Calculate the percentage change from the old price
                percentage_change = (current_price - old_price) / old_price
                

                # Adjust the stop loss by the same percentage
                trade_details['Modified_Stoploss'] = trade_details['Initial_Stoploss']+ trade_details['Initial_Stoploss']* (percentage_change)
                # trade_details['Modified_Stoploss'] *= 1.5
                # old_price = current_price
                logging.info(f"New trailing stop-loss: {trade_details['Modified_Stoploss']} at time: {current_time}")

          
                    # Check if the current price has declined to the stop-loss level
            if current_high <= trade_details['Modified_Stoploss']:
                trade_details['ExitTime'] = current_time
                trade_details['ExitPrice'] = current_low
                trade_details['ExitType'] = "Sell"
                trade_details['pnl'] = trade_details['ExitPrice'] - trade_details['EntryPrice']
                logging.info(f"Stop-loss hit at time: {current_time}, price: {current_price}")
                break

            if current_time >= square_off_time:
                trade_details['ExitTime'] = current_time
                trade_details['ExitPrice'] = current_price
                trade_details['ExitType'] = "Sell"
                trade_details['pnl'] = trade_details['ExitPrice'] - trade_details['EntryPrice']
                logging.info(f"Squared off at time: {current_time}, price: {current_price}")
                break

        return trade_details

        
    if option_type == 'PE':
        entry_order_details =place_order_at_atm_price(ticker_df,Underlying_symbol,entry_time,legs[1])
  

        trade_details = {
            'TickerSymbol': entry_order_details['symbol_to_search'],
            'EntryTime': entry_order_details['entry_time'],
            'ExitTime': None,
            'EntryType': entry_order_details['entry_type'],
            'ExitType': "",
            'EntryPrice': entry_order_details['entry_price'],
            'Initial_Stoploss': entry_order_details['entry_price']*0.5,
            'Modified_Stoploss': entry_order_details['entry_price']*0.5,
            'ExitPrice': None,
            'pnl': None
        }

        logging.info(f"PE position took{trade_details} ")

      
        if isinstance(trade_details['EntryTime'], str):
                trade_details['EntryTime'] = datetime.strptime(trade_details['EntryTime'], '%H:%M:%S').time()
            

        ticker_df['Time'] = pd.to_datetime(ticker_df['Time'], format='%H:%M:%S').dt.time
         
        entry_time = datetime.strptime(entry_time, '%H:%M:%S').time()
       
        
       
        old_price = entry_order_details['entry_price']
   
        filtered_ticker_df = ticker_df[(ticker_df['Time'] > entry_time) & (ticker_df['Ticker'] == entry_order_details['symbol_to_search'])]
        for i in range(len(filtered_ticker_df)):
            current_time = filtered_ticker_df.iloc[i]['Time']
            current_price = filtered_ticker_df.iloc[i]['Close']
            current_high = filtered_ticker_df.iloc[i]['High']
            current_low = filtered_ticker_df.iloc[i]['Low']


            # print(f"Current time: {current_time}, Current price: {current_price}")

            # Update trailing stop-loss if the current price moves favorably by 50%
            if current_price > old_price:
                    # Calculate the percentage change from the old price
                percentage_change = (current_price - old_price) / old_price
                
                # Adjust the stop loss by the same percentage
                trade_details['Modified_Stoploss'] = trade_details['Initial_Stoploss']+ trade_details['Initial_Stoploss']* (percentage_change)
            
                # trade_details['Modified_Stoploss'] *= 1.5
                # old_price = current_price
                
                logging.info(f"New trailing stop-loss: {trade_details['Modified_Stoploss']} at time: {current_time}")

           
                    # Check if the current price has declined to the stop-loss level
            if current_high <= trade_details['Modified_Stoploss']:
                trade_details['ExitTime'] = current_time
                trade_details['ExitPrice'] = current_low
                trade_details['ExitType'] = "Sell"
                trade_details['pnl'] = trade_details['ExitPrice'] - trade_details['EntryPrice']
                logging.info(f"Stop-loss hit at time: {current_time}, price: {current_price}")
                break

            if current_time >= square_off_time:
                trade_details['ExitTime'] = current_time
                trade_details['ExitPrice'] = current_price
                trade_details['ExitType'] = "Sell"
                trade_details['pnl'] = trade_details['ExitPrice'] - trade_details['EntryPrice']
                logging.info(f"Squared off at time: {current_time}, price: {current_price}")
                break

        return trade_details


def filter_by_time(df, start_time="09:15:59", end_time="15:30:59"):
    start_time = datetime.strptime(start_time, "%H:%M:%S").time()
    end_time = datetime.strptime(end_time, "%H:%M:%S").time()
    
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
    filtered_df = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]
    return filtered_df

def process_data(file_path, legs):
    df = pd.read_csv(file_path)
    df0 = pd.read_csv(backtest_config['index_csv_file_path_0'])

    df0['DateTime'] = pd.to_datetime(df0['Date'] + ' ' + df0['Time'])
    df0['Time'] = pd.to_datetime(df0['Time'], format='%H:%M:%S').dt.time
    
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time

    start_time = "09:15:59"
    end_time = "15:30:59"

    ticker_df = pd.read_csv(backtest_config['ticker_csv_file_path'])
        
    filtered_df = filter_by_time(df, start_time=start_time, end_time=end_time).copy()
    filtered_df0 = filter_by_time(df0, start_time=start_time, end_time=end_time).copy()
    ticker_filtered_df = filter_by_time(ticker_df, start_time=start_time, end_time=end_time).copy()
    
    previous_EMA_9 = initial_ema(filtered_df0, 'Close', backtest_config['Technical_Indicator_Details']['EMA_RSI']['ema9_period'], '15:30:59')
    previous_EMA_26 = initial_ema(filtered_df0, 'Close', backtest_config['Technical_Indicator_Details']['EMA_RSI']['ema26_period'], '15:30:59')

    # previous_EMA_9 = calculate_ema(filtered_df0, 'Close', backtest_config['Technical_Indicator_Details']['EMA_RSI']['ema9_period']).iloc[-1]
    # previous_EMA_26 = calculate_ema(filtered_df0, 'Close', backtest_config['Technical_Indicator_Details']['EMA_RSI']['ema26_period']).iloc[-1]

    # print("previous_EMA_9 ===", previous_EMA_9)
    # print("previous_EMA_26 ===", previous_EMA_26)
    
    filtered_df.loc[:, 'EMA_9'] = calculate_ema(filtered_df, 'Close', backtest_config['Technical_Indicator_Details']['EMA_RSI']['ema9_period'], previous_EMA_9)
    filtered_df.loc[:, 'EMA_26'] = calculate_ema(filtered_df, 'Close', backtest_config['Technical_Indicator_Details']['EMA_RSI']['ema26_period'], previous_EMA_26)
    filtered_df.loc[:, 'RSI_14'] = calculate_rsi(filtered_df, column='Close', periods=14)
    filtered_df.loc[:, 'Signal'] = ''
    
    trades = []
    skip_periods = 0
    
    for i in range(1, len(filtered_df)):
        if skip_periods > 0:
            skip_periods -= 1
            # filtered_df.at[filtered_df.index[i], 'Signal'] = 'Hold'
            continue

        cross_up = (filtered_df.iloc[i]['EMA_9'] > filtered_df.iloc[i]['EMA_26']) and (filtered_df.iloc[i-1]['EMA_9'] < filtered_df.iloc[i-1]['EMA_26']) and (filtered_df.iloc[i]['RSI_14'] > 50)
        cross_down = (filtered_df.iloc[i]['EMA_9'] < filtered_df.iloc[i]['EMA_26']) and (filtered_df.iloc[i-1]['EMA_9'] > filtered_df.iloc[i-1]['EMA_26']) and (filtered_df.iloc[i]['RSI_14'] < 50)

        if cross_up:
            filtered_df.at[filtered_df.index[i], 'Signal'] = 'Crossup_Buy'
            current_price = filtered_df.iloc[i]['Close']
            current_time = str(filtered_df.iloc[i]['Time'])
            # logging.info('Current time_crossup: %s', current_time)
            # logging.info('Current price_crossup: %s', current_price)
            trade = buy_atm_option_and_check_sl_hit_or_squareoff(legs, current_price, backtest_config['Underlying_Symbol'], ticker_filtered_df, 'CE', current_time, 50)
            trades.append(trade)
            skip_periods = 20
        elif cross_down:
            filtered_df.at[filtered_df.index[i], 'Signal'] = 'Crossdown_Buy'
            current_price = filtered_df.iloc[i]['Close']
            current_time = str(filtered_df.iloc[i]['Time'])
            # logging.info('Current time_crossdown: %s', current_time)
            # logging.info('Current price_crossdown: %s', current_price)
            trade = buy_atm_option_and_check_sl_hit_or_squareoff(legs, current_price, backtest_config['Underlying_Symbol'], ticker_filtered_df, 'PE', current_time, 50)
            trades.append(trade)
            skip_periods = 20
        else:
           pass
    
       # save_trade_in_csv()
    
    
    
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(backtest_config['trades_df_csv_path'], index=False)
    # Save final DataFrame to CSV
    filtered_df.to_csv(backtest_config['signals_df_csv_path'], index=False)
    return trades_df

 



# Manually create the list of date pairs, excluding specific dates except January 20
dates_to_process = [
    ("2023-12-29", "2024-01-01"),
    ("2024-01-01", "2024-01-02"),
    ("2024-01-02", "2024-01-03"),
    ("2024-01-03", "2024-01-04"),
    ("2024-01-04", "2024-01-05"),
    ("2024-01-05", "2024-01-08"),  # Skip 06, 07
    ("2024-01-08", "2024-01-09"),
    ("2024-01-09", "2024-01-10"),
    ("2024-01-10", "2024-01-11"),
    ("2024-01-11", "2024-01-12"),
    ("2024-01-12", "2024-01-15"),  # Skip 13, 14
    ("2024-01-15", "2024-01-16"),
    ("2024-01-16", "2024-01-17"),
    ("2024-01-17", "2024-01-18"),
    ("2024-01-18", "2024-01-19"),
    ("2024-01-19", "2024-01-20"),
    ("2024-01-20", "2024-01-23"),  # Skip 21, 22
    ("2024-01-23", "2024-01-24"),
    ("2024-01-24", "2024-01-25"),
    ("2024-01-25", "2024-01-29"),  # Skip 26, 27, 28
    ("2024-01-29", "2024-01-30"),
    ("2024-01-30", "2024-01-31"),
    ("2024-01-31", "2024-02-01")
]

backtest_path = "/Users/pranaygaurav/Downloads/AlgoTrading/Option_Trading_Strategies/EMA_RSI"
data_path = "/Users/pranaygaurav/Downloads/AlgoTrading/Option_Trading_Strategies/EMA_RSI/DATA"
trades_output_dir = os.path.join(backtest_path, "TRADES")
signals_output_dir = os.path.join(backtest_path, "SIGNALS")

# Ensure the output directory exists
os.makedirs(trades_output_dir, exist_ok=True)
os.makedirs(signals_output_dir, exist_ok=True)

# List to hold different backtest configurations
backtest_configs = []

for previous_date, current_date in dates_to_process:
    # Extract month and day for the filenames
    previous_date_obj = datetime.strptime(previous_date, "%Y-%m-%d")
    current_date_obj = datetime.strptime(current_date, "%Y-%m-%d")
    
    prev_month = previous_date_obj.strftime("%b").upper()
    curr_month = current_date_obj.strftime("%b").upper()
    
    prev_day = previous_date_obj.strftime("%d")
    curr_day = current_date_obj.strftime("%d")
    
    prev_year = previous_date_obj.strftime("%Y")
    curr_year = current_date_obj.strftime("%Y")
    
    # Create a new configuration dictionary
    config = {
        "index_csv_file_path_0": os.path.join(data_path, f"INDEX_{prev_month}_{prev_year}", f"BANKNIFTY_GFDLCM_INDICES_{previous_date_obj.strftime('%d%m%Y')}.csv"),
        "index_csv_file_path": os.path.join(data_path, f"INDEX_{curr_month}_{curr_year}", f"BANKNIFTY_GFDLCM_INDICES_{current_date_obj.strftime('%d%m%Y')}.csv"),
        "ticker_csv_file_path": os.path.join(data_path, f"TICKER_{curr_month}_{curr_year}", f"ATM_BANKNIFTY_{curr_day}{curr_month}{curr_year[2:]}".upper() + ".csv"),
        "trades_df_csv_path": os.path.join(trades_output_dir, f"trades_{current_date_obj.strftime('%d%m%Y')}.csv"),
        "signals_df_csv_path": os.path.join(signals_output_dir, f"signals_{current_date_obj.strftime('%d%m%Y')}.csv"),
        "start_time": "09:15:59",
        "squareoff_time": "15:30:59",
        "stoploss_percentage": 50,
        "target_percentage": 50,
        "Underlying_Symbol": "",
        "entrylegs_details": [
            {'id': 1, 'Ticker_Symbol': '', 'type': 'CE', 'EntryType': "Buy", 'ExitType': "Sell", 'lot_size': 0, 'offset': 0},
            {'id': 2, 'Ticker_Symbol': '', 'type': 'PE', 'EntryType': "Buy", 'ExitType': "Sell", 'lot_size': 0, 'offset': 0},
        ],
        "Technical_Indicator_Details": {
            "EMA_RSI": {
                "rsi_period": 14,
                "ema9_period": 9,
                "ema26_period": 26,
            }
        }
    }
    
    if previous_date == "2023-12-29":
        config["index_csv_file_path_0"] = "/Users/pranaygaurav/Downloads/AlgoTrading/Backtest_Option_Trading/month_wise_backtest/EMA_RSI/DATA/INDEX_JAN_2024/0.BANKNIFTY_GFDLCM_INDICES_29122023.csv"
    
    backtest_configs.append(config)

# Process each backtest configuration
for iteration, backtest_config in enumerate(backtest_configs, start=1):
    try:
       
        # Log the paths being processed
        logging.info(f"Iteration: {iteration}")
        logging.info(f"index_csv_file_path_0: {backtest_config['index_csv_file_path_0']}")
        logging.info(f"index_csv_file_path: {backtest_config['index_csv_file_path']}")
        logging.info(f"ticker_csv_file_path: {backtest_config['ticker_csv_file_path']}")
        
        
        
        # Update entrylegs_details
        entrylegs_details = update_entrylegs_details(backtest_config['ticker_csv_file_path'], backtest_config['entrylegs_details'])
        
        # Process the data
        trades_df = process_data(backtest_config['index_csv_file_path'], entrylegs_details)
        
        # Save the trades_df to CSV
        trades_df.to_csv(backtest_config['trades_df_csv_path'], index=False)

        logging.info(f"Processed trades for {backtest_config['index_csv_file_path']}. Output saved to {backtest_config['trades_df_csv_path']}")
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        continue
    except Exception as e:
        logging.error(f"Error processing {backtest_config['index_csv_file_path']}: {e}")
        continue

logging.info("Backtesting completed for all dates.")

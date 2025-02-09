import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import pandas_ta as ta
# Set up logging
logging.basicConfig(level=logging.INFO, filename='/Users/pranaygaurav/Downloads/AlgoTrading/Option_Trading_Strategies/SUPERTREND/Logs/supertrend-backtest_log.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# def calculate_supertrend(df0,df, period=10, multiplier=3):
#     # Log the filtered data
#     # logger.info("Filtered data is:\n%s" % df)
#     df["ATR"] = ta.atr(df.High,df.Low,df.Close, length=14)
#     sti= ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=3)
#     sti = round(sti, 2)
#     df = pd.merge(df, sti, left_index=True,right_index=True)
#     df = df[["Time","High","Low","Close","SUPERT_10_3.0"]]
#     logging.info("ATR data is:\n%s" % df)
   

#     return df



import pandas as pd
import pandas_ta as ta
import logging

def calculate_supertrend(df0, df, period=10, multiplier=3):
    # Initialize an empty DataFrame to store the results
    result_df = pd.DataFrame(columns=["Time", "High", "Low", "Close", "SUPERT_10_3.0"])

 
    # Merge filtered df0 with df
    df_merged = pd.concat([df0, df]).reset_index(drop=True)

    # logging.info(f"MERGED DF \n {df_merged.head(25)}")

    # Calculate ATR using pandas_ta library
    df_merged["ATR"] = ta.atr(df_merged['High'], df_merged['Low'], df_merged['Close'], length=period)

    # Calculate Supertrend using pandas_ta library
    supertrend = ta.supertrend(df_merged['High'], df_merged['Low'], df_merged['Close'], length=period, multiplier=multiplier)
    df_merged["SUPERT_10_3.0"] = supertrend['SUPERT_10_3.0']



    # Select final columns
    result_df = df_merged[["Time", "High", "Low", "Close", "SUPERT_10_3.0"]]

    logging.info(f"RESULT DF \n {df_merged.tail(25)}")

    return result_df

   





def filter_by_time(df, start_time="09:15:59", end_time="15:30:59"):
    start_time = datetime.strptime(start_time, "%H:%M:%S").time()
    end_time = datetime.strptime(end_time, "%H:%M:%S").time()
    
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
    filtered_df = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]
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

def extract_symbol_and_expiry(ticker):
    import re
    match = re.match(r'([A-Z]+)(\d{2}[A-Z]{3}\d{2})', ticker)
    if match:
        symbol = match.group(1)
        backtest_config['Underlying_Symbol'] = symbol
        expiry = match.group(2)
        return symbol + expiry, symbol
    return None, None

def place_order_at_atm_price(ticker_df, symbol, entry_time, leg):
    atm = atm_at_time_t(ticker_df, entry_time, symbol)
    strike = atm + leg['offset']
    symbol_to_search = f"{leg['Ticker_Symbol']}{strike}{leg['type']}.NFO"
    ticker_row_entry = ticker_row_at_time(ticker_df, symbol_to_search, entry_time)
    entry_price = get_ohlc_ticker_row_at_time_t(ticker_df, ticker_row_entry, "Close")
    
    return {
        'symbol_to_search': symbol_to_search,
        'entry_type': "Buy",
        'entry_time': entry_time,
        'entry_price': entry_price,
        'Stoploss_price': None
    }

def atm_at_time_t(ticker_df, entry_time, symbol):
    ticker_df['Time'] = pd.to_datetime(ticker_df['Time'], format='%H:%M:%S').dt.time
    entry_time = datetime.strptime(entry_time, '%H:%M:%S').time()
    atm_row = ticker_df[(ticker_df['Time'] == entry_time) & (ticker_df['Ticker'].str.contains(symbol))]
    
    if atm_row.empty:
        raise ValueError(f"No data available for symbol {symbol} at time {entry_time}")
    
    atm = int(round(atm_row.iloc[0]['ATM_close'] / 100) * 100)
    return atm

def ticker_row_at_time(ticker_df, symbol_to_search, time):
    ticker_df['Time'] = pd.to_datetime(ticker_df['Time'], format='%H:%M:%S').dt.time
    time = datetime.strptime(time, '%H:%M:%S').time()
    return ticker_df[(ticker_df['Ticker'] == symbol_to_search) & (ticker_df['Time'] == time)]

def get_ohlc_ticker_row_at_time_t(df, ticker_row_at_time, ohlc_column):
    if ticker_row_at_time.empty:
        return None
    return ticker_row_at_time.iloc[0][ohlc_column]

def buy_atm_option_and_check_sl_hit_or_squareoff(legs, indexentryprice, Underlying_symbol, ticker_df, option_type, entry_time, stoploss_percentage):
    square_off_time = datetime.strptime(backtest_config['squareoff_time'], '%H:%M:%S').time()

    if option_type == 'CE':
        entry_order_details = place_order_at_atm_price(ticker_df, Underlying_symbol, entry_time, legs[0])
        trade_details = {
            'TickerSymbol_______________': entry_order_details['symbol_to_search'],
            'EntryTime': entry_order_details['entry_time'],
            'ExitTime': None,
            'EntryType': entry_order_details['entry_type'],
            'ExitType': "",
            'EntryPrice': entry_order_details['entry_price'],
            'Initial_Stoploss': entry_order_details['entry_price'] * 0.5,
            'Modified_Stoploss': entry_order_details['entry_price'] * 0.5,
            'ExitPrice': None,
            'pnl': None
        }

        ticker_df['Time'] = pd.to_datetime(ticker_df['Time'], format='%H:%M:%S').dt.time
        entry_time = datetime.strptime(entry_time, '%H:%M:%S').time()
        old_price = entry_order_details['entry_price']
        filtered_ticker_df = ticker_df[(ticker_df['Time'] > entry_time) & (ticker_df['Ticker'] == entry_order_details['symbol_to_search'])]

        for i in range(len(filtered_ticker_df)):
            current_time = filtered_ticker_df.iloc[i]['Time']
            current_price = filtered_ticker_df.iloc[i]['Close']
            current_high = filtered_ticker_df.iloc[i]['High']
            current_low = filtered_ticker_df.iloc[i]['Low']

            if current_price > old_price:
                percentage_change = (current_price - old_price) / old_price
                trade_details['Modified_Stoploss'] = trade_details['Initial_Stoploss']+ trade_details['Initial_Stoploss']* (percentage_change)
                trade_details['Modified_Stoploss'] = round(trade_details['Modified_Stoploss'],3)

            if current_high <= trade_details['Modified_Stoploss']:
                trade_details['ExitTime'] = current_time
                trade_details['ExitPrice'] = current_low
                trade_details['ExitType'] = "Sell"
                trade_details['pnl'] = round(trade_details['ExitPrice'] - trade_details['EntryPrice'],3)
                logging.info(f"Stop-loss hit at time: {current_time}, price: {current_price}")
                break

            if current_time >= square_off_time:
                trade_details['ExitTime'] = current_time
                trade_details['ExitPrice'] = current_price
                trade_details['ExitType'] = "Sell"
                trade_details['pnl'] = round(trade_details['ExitPrice'] - trade_details['EntryPrice'],3)
                logging.info(f"Squared off at time: {current_time}, price: {current_price}")
                break

        return trade_details

    if option_type == 'PE':
        entry_order_details = place_order_at_atm_price(ticker_df, Underlying_symbol, entry_time, legs[1])
        trade_details = {
            'TickerSymbol_______________': entry_order_details['symbol_to_search'],
            'EntryTime': entry_order_details['entry_time'],
            'ExitTime': None,
            'EntryType': entry_order_details['entry_type'],
            'ExitType': "",
            'EntryPrice': entry_order_details['entry_price'],
            'Initial_Stoploss': entry_order_details['entry_price'] * 0.5,
            'Modified_Stoploss': entry_order_details['entry_price'] * 0.5,
            'ExitPrice': None,
            'pnl': None
        }

        ticker_df['Time'] = pd.to_datetime(ticker_df['Time'], format='%H:%M:%S').dt.time
        entry_time = datetime.strptime(entry_time, '%H:%M:%S').time()
        old_price = entry_order_details['entry_price']
        filtered_ticker_df = ticker_df[(ticker_df['Time'] > entry_time) & (ticker_df['Ticker'] == entry_order_details['symbol_to_search'])]

        for i in range(len(filtered_ticker_df)):
            current_time = filtered_ticker_df.iloc[i]['Time']
            current_price = filtered_ticker_df.iloc[i]['Close']
            current_high = filtered_ticker_df.iloc[i]['High']
            current_low = filtered_ticker_df.iloc[i]['Low']

            if current_price > old_price:
                percentage_change = (current_price - old_price) / old_price
                trade_details['Modified_Stoploss'] = trade_details['Initial_Stoploss']+ trade_details['Initial_Stoploss']* (percentage_change)
                trade_details['Modified_Stoploss'] = round(trade_details['Modified_Stoploss'],3)

            if current_high <= trade_details['Modified_Stoploss']:
                trade_details['ExitTime'] = current_time
                trade_details['ExitPrice'] = current_low
                trade_details['ExitType'] = "Sell"
                trade_details['pnl'] = round(trade_details['ExitPrice'] - trade_details['EntryPrice'],3)
                logging.info(f"Stop-loss hit at time: {current_time}, price: {current_price}")
                break

            if current_time >= square_off_time:
                trade_details['ExitTime'] = current_time
                trade_details['ExitPrice'] = current_price
                trade_details['ExitType'] = "Sell"
                trade_details['pnl'] = round(trade_details['ExitPrice'] - trade_details['EntryPrice'],3)
                logging.info(f"Squared off at time: {current_time}, price: {current_price}")
                break

        return trade_details

def process_data(file_path, legs):


    start_time = "09:15:59"
    end_time = "15:30:59"



    df = pd.read_csv(file_path)
    df0 = pd.read_csv(backtest_config['index_csv_file_path_0'])

    df0['DateTime'] = pd.to_datetime(df0['Date'] + ' ' + df0['Time'])
    df0['Time'] = pd.to_datetime(df0['Time'], format='%H:%M:%S').dt.time

    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time




    filtered_df0 = filter_by_time(df0, start_time=start_time, end_time=end_time).copy()
    filtered_df = filter_by_time(df, start_time=start_time, end_time=end_time).copy()

    ticker_df = pd.read_csv(backtest_config['ticker_csv_file_path'])
    ticker_filtered_df = filter_by_time(ticker_df, start_time=start_time, end_time=end_time).copy()
    
    merged_df = calculate_supertrend(filtered_df0,filtered_df, backtest_config['Technical_Indicator_Details']['SUPERTREND']['atr_period'], 
                                        backtest_config['Technical_Indicator_Details']['SUPERTREND']['multiplier'])
    
    merged_df.loc[:, 'Signal'] = ''
 
           
    trades = []
   
    
   
    for i in range(1, len(merged_df)):
        latest = merged_df.iloc[i]
        previous = merged_df.iloc[i - 1]

        if latest['Close'] > latest['SUPERT_10_3.0'] and previous['Close'] < previous['SUPERT_10_3.0']:
            signal = 'UPTREND_BUY'
            merged_df.at[merged_df.index[i], 'Signal'] = signal
            current_price = latest['Close']
            current_time = str(merged_df.iloc[i]['Time'])
            logging.debug(f"Buy CE Signal at time: {current_time}, price: {current_price}")
            trade = buy_atm_option_and_check_sl_hit_or_squareoff(legs, current_price, backtest_config['Underlying_Symbol'], ticker_filtered_df, 'CE', current_time, 50)
            trades.append(trade)

        elif latest['Close'] < latest['SUPERT_10_3.0'] and previous['Close'] > previous['SUPERT_10_3.0']:
            signal = 'DOWNTREND_BUY'
            merged_df.at[merged_df.index[i], 'Signal'] = signal
            current_price = latest['Close']
            current_time = str(merged_df.iloc[i]['Time'])
            logging.debug(f"Buy PE Signal at time: {current_time}, price: {current_price}")
            trade = buy_atm_option_and_check_sl_hit_or_squareoff(legs, current_price, backtest_config['Underlying_Symbol'], ticker_filtered_df, 'PE', current_time, 50)
            trades.append(trade)
        
        else:
            pass


    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(backtest_config['trades_df_csv_path'], index=False)
    merged_df.to_csv(backtest_config['signals_df_csv_path'], index=False)
    return trades_df

# # Generate the list of dates to process, excluding specific dates
# dates_to_process = []
# start_date = datetime(2024, 1, 1)
# end_date = datetime(2024, 1, 31)
# excluded_dates = [datetime(2024, 1, 6), datetime(2024, 1, 7),
#                   datetime(2024, 1, 13), datetime(2024, 1, 14),
#                   datetime(2024, 1, 20), datetime(2024, 1, 21),
#                   datetime(2024, 1, 22), datetime(2024, 1, 26), 
#                   datetime(2024, 1, 27), datetime(2024, 1, 28)]



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


# current_date = start_date
# while current_date <= end_date:
#     if current_date not in excluded_dates:
#         dates_to_process.append(current_date)
#     current_date += timedelta(days=1)

backtest_path = "/Users/pranaygaurav/Downloads/AlgoTrading/Option_Trading_Strategies/SUPERTREND/"
data_path = "/Users/pranaygaurav/Downloads/AlgoTrading/Option_Trading_Strategies/SUPERTREND/DATA"
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
            "SUPERTREND": {
                "atr_period": 10,
                "multiplier": 3,
            }
        }
    }

        
    if previous_date == "2023-12-29":
        config["index_csv_file_path_0"] = "/Users/pranaygaurav/Downloads/AlgoTrading/Backtest_Option_Trading/month_wise_backtest/SUPERTREND/DATA/INDEX_JAN_2024/0.BANKNIFTY_GFDLCM_INDICES_29122023.csv"
    
    
    backtest_configs.append(config)

for iteration, backtest_config in enumerate(backtest_configs, start=1):
    try:
   
  
        
          # Log the paths being processed
        logging.info(f"Iteration: {iteration}")
        logging.info(f"index_csv_file_path_0: {backtest_config['index_csv_file_path_0']}")
        logging.info(f"index_csv_file_path: {backtest_config['index_csv_file_path']}")
        logging.info(f"ticker_csv_file_path: {backtest_config['ticker_csv_file_path']}")
        
        
        
    
     
        
        entrylegs_details = update_entrylegs_details(backtest_config['ticker_csv_file_path'], backtest_config['entrylegs_details'])
        
        trades = process_data(backtest_config['index_csv_file_path'], entrylegs_details)



        trades_df = pd.DataFrame(trades, columns=[
        'TickerSymbol_______________', 'EntryTime', 'ExitTime', 'EntryType', 'ExitType',
        'EntryPrice', 'Initial_Stoploss', 'Modified_Stoploss', 'ExitPrice', 'pnl'
        ])
    
        # Ensure all columns are aligned by reordering them if necessary
        trades_df = trades_df[['TickerSymbol_______________', 'EntryTime', 'ExitTime', 'EntryType', 'ExitType',
                            'EntryPrice', 'Initial_Stoploss', 'Modified_Stoploss', 'ExitPrice', 'pnl']]

        # trades_df.to_csv(backtest_config['trades_df_csv_path'], index=False)


            # Calculate the maximum width for each column based on the column name and the values in that column
        column_widths = {col: max(len(str(col)), trades_df[col].astype(str).map(len).max()) for col in trades_df.columns}

        # Pad each entry in the DataFrame to match the column name's width
        for col in trades_df.columns:
            trades_df[col] = trades_df[col].astype(str).map(lambda x: x.ljust(column_widths[col]))

        
        trades_df.to_csv(backtest_config['trades_df_csv_path'], index=False)

        logging.info(f"Processed trades for {backtest_config['index_csv_file_path']}. Output saved to {backtest_config['trades_df_csv_path']}")
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        continue
    except Exception as e:
        logging.error(f"Error processing {backtest_config['index_csv_file_path']}: {e}")
        continue

logging.info("Backtesting completed for all dates.")

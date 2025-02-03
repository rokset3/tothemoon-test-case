import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def get_data(data_path):
    df = pd.read_csv(data_path).drop(columns=['Unnamed: 0'])
    df['date'] = pd.to_datetime(df['date'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def compute_pnl(df, pos_col="pos_size", price_col="price"):
    """
    A simpler, mark-to-market P&L calculation:
      PnL(t) = Position(t-1) * (Price(t) - Price(t-1))

    Arguments:
        df: A pandas DataFrame with columns [pos_col, price_col]
        pos_col: Name of the column containing position sizes (long=+, short=-)
        price_col: Name of the column containing prices

    Returns:
        The same DataFrame with new columns:
            'pnl': Instantaneous P&L at each row
            'cumulative_pnl': Cumulative sum of P&L
    """
    # Safety checks
    if pos_col not in df.columns or price_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain '{pos_col}' and '{price_col}' columns."
        )

    # Convert to NumPy arrays
    pos_array = df[pos_col].to_numpy(dtype=float)
    price_array = df[price_col].to_numpy(dtype=float)
    n = len(df)

    # 1) Shift the position by 1 (so row t uses the position from t-1)
    #    For the first row, we assume position(t-1) = 0
    pos_shifted = np.roll(pos_array, shift=1)
    pos_shifted[0] = 0.0  # at row 0, there's no "prior" position in this scheme

    # 2) Compute price differences: price[t] - price[t-1]
    #    For row 0, we can define diff=0 or skip entirely. We'll set row 0's diff=0.
    price_diff = np.diff(np.insert(price_array, 0, price_array[0]))
    # Explanation:
    #  - np.insert(..., 0, price_array[0]) duplicates the first price at the start
    #  - np.diff(...) then yields differences across the new array
    #  - The first difference (row 0) will be 0, because price_array[0] - price_array[0] = 0.

    # 3) Multiply to get instantaneous PnL
    pnl_array = pos_shifted * price_diff

    # 4) Cumulative sum
    cum_pnl_array = np.cumsum(pnl_array)

    # Assign results back to DataFrame
    df[f"{price_col}_pnl"] = pnl_array
    df[f"{price_col}_cumulative_pnl"] = cum_pnl_array

    return df

def ema(series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def compute_macd(df, short_span: int, long_span: int, signal_span: int,
                 price_col="close", macd_col="pos_macd"):
    """
    Compute MACD with user-defined EMA spans, then assign a position:
      +1 if MACD > Signal, else -1.
    Returns a copy of df with columns: macd_col, and intermediate columns [macd, signal].
    """
    df = df.copy()
    
    df["ema_short"] = ema(df[price_col], short_span)
    df["ema_long"] = ema(df[price_col], long_span)
    df["macd"] = df["ema_short"] - df["ema_long"]
    df["signal"] = ema(df["macd"], signal_span)
    
    # Position: +1 if MACD>Signal else -1
    df[macd_col] = (df["macd"] > df["signal"]).astype(int) * 2 - 1
    # Explanation: True->1 => 2*1-1=+1, False->0 => 2*0-1=-1
    
    return df

def evaluate_macd_on_fold(train_df, val_df,
                          short_span, long_span, signal_span,
                          price_col="close"):
    """
    - Concatenate train + val (so the MACD rolling calculations can include past data).
    - Compute MACD signals and PnL on the combined set.
    - Then measure PnL only on the validation period (val_df) to avoid lookahead.
    Returns the final cumulative PnL on val_df.
    """
    # 1) Combine train + val for indicator computation
    combined = pd.concat([train_df, val_df], ignore_index=True)
    combined = compute_macd(combined, short_span, long_span, signal_span,
                            price_col=price_col, macd_col="pos_macd")
    
    # 2) Compute PnL on combined
    combined = compute_pnl(combined, pos_col="pos_macd", price_col=price_col)
    
    # 3) Extract the validation slice from combined, measure cumulative PnL difference
    #    We identify the exact index range that corresponds to val_df
    val_start_idx = len(train_df)  # first row of val is right after train
    val_end_idx = len(train_df) + len(val_df)  # exclusive
    val_slice = combined.iloc[val_start_idx:val_end_idx]
    if val_slice.empty:
        return 0.0
    
    # 4) The final cumulative PnL in the validation slice
    final_cum_pnl = val_slice[f"{price_col}_cumulative_pnl"].iloc[-1]
    # Alternatively, you could measure the difference from the start of val to end of val
    start_cum_pnl = val_slice[f"{price_col}_cumulative_pnl"].iloc[0]
    val_pnl = final_cum_pnl - start_cum_pnl
    
    return val_pnl

def evaluate_macd_grid(df, val_windows, param_grid, date_col="timestamp", price_col="close"):
    """
    For each (short_span, long_span, signal_span) in param_grid:
      1) Create folds with create_expanding_splits(df, val_windows).
      2) Evaluate each fold's final val PnL.
      3) Average across folds.
    Returns a DataFrame summarizing results for each parameter set.
    """
    from statistics import mean

    # 1) Create the folds
    folds = create_expanding_splits(df, val_windows, date_col=date_col)

    results = []
    for (short_span, long_span, signal_span) in param_grid:
        fold_pnls = []
        
        for (train_df, val_df) in folds:
            if val_df.empty or train_df.empty:
                # skip if there's no data to validate
                continue
            
            val_pnl = evaluate_macd_on_fold(
                train_df, val_df,
                short_span=short_span,
                long_span=long_span,
                signal_span=signal_span,
                price_col=price_col
            )
            fold_pnls.append(val_pnl)
        
        avg_pnl = mean(fold_pnls) if fold_pnls else 0
        results.append({
            "short_span": short_span,
            "long_span": long_span,
            "signal_span": signal_span,
            "avg_val_pnl": avg_pnl
        })
    
    return pd.DataFrame(results)

def clean_text(text):
    """
    Lowercase text, remove URLs, mentions, hashtags, non-alphabetical characters, and stopwords.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)   # remove mentions
    text = re.sub(r'#', '', text)      # remove hashtag symbols
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters and spaces
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def add_text_sentiment(df_text, text_col='post_text'):
    """
    Clean the raw text and compute a compound sentiment score for each post.
    """
    analyzer = SentimentIntensityAnalyzer()
    df_text['clean_text'] = df_text[text_col].apply(clean_text)
    df_text['sentiment'] = df_text['clean_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    return df_text

def create_lag_features(df, col, lags=[1, 2, 3, 5]):
    """
    Create lag features for the specified column.
    """
    for lag in lags:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

def create_rolling_features(df, col, window=5):
    """
    Create rolling mean and standard deviation features for the specified column.
    """
    df[f'{col}_roll_mean'] = df[col].rolling(window).mean()
    df[f'{col}_roll_std'] = df[col].rolling(window).std()
    return df

def pnl(df, pos_col="pos_size", price_col="price"):
    """
    Calculate instantaneous and cumulative PnL.
    
    PnL(t) = Position(t-1) * (Price(t) - Price(t-1))
    """
    pos_array = df[pos_col].to_numpy(dtype=float)
    price_array = df[price_col].to_numpy(dtype=float)
    pos_shifted = np.roll(pos_array, shift=1)
    pos_shifted[0] = 0.0  # first period has no previous position
    price_diff = np.diff(np.insert(price_array, 0, price_array[0]))
    pnl_array = pos_shifted * price_diff
    cum_pnl_array = np.cumsum(pnl_array)
    df["pnl"] = pnl_array
    df["cumulative_pnl"] = cum_pnl_array
    return df

def create_expanding_splits(df, val_windows, date_col="timestamp", gap_months=2):
    """
    Creates a list of (train_df, val_df) folds using an expanding window with a gap.
    
    For each validation window (val_start, val_end), the training data includes only
    data strictly before (val_start - gap). This gap helps reduce data leakage from
    running/lagging features.
    """
    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)
    folds = []
    for (val_start, val_end) in val_windows:
        gap_start = val_start - pd.DateOffset(months=gap_months)
        train_mask = df_sorted[date_col] < gap_start
        train_df = df_sorted.loc[train_mask].copy()
        val_mask = (df_sorted[date_col] >= val_start) & (df_sorted[date_col] < val_end)
        val_df = df_sorted.loc[val_mask].copy()
        folds.append((train_df, val_df))
    return folds
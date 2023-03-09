STOCK_PRICES = "orats_price_history_to_2020-09-28.csv"
DIVIDENDS = "orats_dividends.csv"
EARNINGS = "orats_earnings.csv"
WINDOW_LIST = [20]
MATURITY_BUCKETS = [0.0, 5.0, 15.0, 30.0, 60.0, 120.0, 250.0]
MONEYNESS = [-2.0, -1.0, 0.0, 1.0, 2.0]
QUANTILE_LIST = [0.5]
RETURNS_FOR_STATISTICS = [1, 2, 4, 5, 6, 19, 20, 21, 252]

DATASETS_DIRECTORIES = ["ivs", "volumes", "open_interests", "other"]

DO_NOT_NORMALIZE = [
    "sigma_for_moneyness",
    "number_days_to_earnings",
    "number_days_to_dividends",
    "close",
    "dividend_amount",
    "dividend_adjusted_prices",
    "earnings_flag",
    "ticker",
    "business_days_to_maturity",
]

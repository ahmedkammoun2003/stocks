import investpy
try:
    df = investpy.get_stock_historical_data(stock='SFBT', country='tunisia', from_date='01/01/2010', to_date='01/01/2023')
    print("Success")
    print(df.head())
except Exception as e:
    print(f"Failed: {e}")

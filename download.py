import yfinance as yf

ticker = "RELIANCE.NS"   # change stock symbol if needed
start_date = "2023-01-01"
end_date = "2025-12-29"

df = yf.download(ticker, start=start_date, end=end_date)

df.to_csv("RELIANCE.csv")

print("CSV downloaded successfully!")

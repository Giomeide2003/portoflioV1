-- Example SQL screening query.
-- Adapt the table and column names to the target market-data warehouse.

SELECT
    ticker,
    date,
    close,
    sector,
    market_cap
FROM market_prices
WHERE date >= :start_date
  AND date <= :end_date
  AND market_cap >= :minimum_market_cap
  AND sector IS NOT NULL
ORDER BY ticker, date;

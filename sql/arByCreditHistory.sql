SELECT 
  "Credit_History",
  COUNT(*) AS total,
  SUM(CASE WHEN "Loan_Status" = 'Y' THEN 1 ELSE 0 END) AS approved,
  ROUND(100.0 * SUM(CASE WHEN "Loan_Status" = 'Y' THEN 1 ELSE 0 END)/COUNT(*), 2) AS approval_rate_pct
FROM loan_data
GROUP BY "Credit_History"
ORDER BY approval_rate_pct DESC;

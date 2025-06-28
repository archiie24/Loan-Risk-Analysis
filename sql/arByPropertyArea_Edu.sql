SELECT 
  "Property_Area",
  "Education",
  COUNT(*) AS total,
  ROUND(100.0 * SUM(CASE WHEN "Loan_Status" = 'Y' THEN 1 ELSE 0 END)/COUNT(*), 2) AS approval_rate_pct
FROM loan_data
GROUP BY "Property_Area", "Education"
ORDER BY approval_rate_pct DESC;

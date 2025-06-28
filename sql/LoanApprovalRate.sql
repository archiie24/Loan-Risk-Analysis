SELECT
    COUNT(*) AS total_applications,
    SUM(CASE WHEN "Loan_Status" = 'Y' THEN 1 ELSE 0 END) AS approved,
    ROUND(
        100.0 * SUM(CASE WHEN "Loan_Status" = 'Y' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS approval_rate_pct
FROM loan_data;

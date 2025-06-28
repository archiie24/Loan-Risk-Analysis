SELECT 
  "Education",
  "Self_Employed",
  ROUND(AVG("ApplicantIncome"), 2) AS avg_income,
  ROUND(AVG("LoanAmount"), 2) AS avg_loan
FROM loan_data
GROUP BY "Education", "Self_Employed"
ORDER BY avg_income DESC;

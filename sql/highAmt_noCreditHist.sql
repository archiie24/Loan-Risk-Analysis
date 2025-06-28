SELECT 
  "Loan_ID",--
  "ApplicantIncome",
  "LoanAmount",
  "Credit_History",
  "Loan_Status"
FROM loan_data
WHERE "Credit_History" = 0 AND "LoanAmount" > 150;

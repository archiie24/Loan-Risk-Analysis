SELECT 
  "Loan_ID",
  "Property_Area",
  "ApplicantIncome",
  "LoanAmount",
  RANK() OVER (PARTITION BY "Property_Area" ORDER BY "LoanAmount" DESC) AS loan_rank_in_area
FROM loan_data;

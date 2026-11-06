-- ==============================================================================
-- PROJECT: CONSUMER LOAN & CREDIT RISK SQL ANALYTICS
-- AUTHOR: Candra Savira (Business Statistics, ITS)
-- DATABASE: SQLite / PostgreSQL Compatible Relational Schema
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- QUERY 1: EXECUTIVE PORTFOLIO SUMMARY & NON-PERFORMING LOAN (NPL) METRICS
-- Calculates total loan volume, active borrowers, average DTI, and NPL (Kol-3 to Kol-5)
-- ------------------------------------------------------------------------------
WITH loan_metrics AS (
    SELECT 
        COUNT(DISTINCT l.loan_id) AS total_loans,
        COUNT(DISTINCT l.borrower_id) AS total_borrowers,
        SUM(l.principal_amount) AS total_principal_disbursed,
        AVG(l.principal_amount) AS avg_loan_size,
        AVG(l.interest_rate) AS avg_interest_rate,
        SUM(CASE WHEN l.loan_status IN ('Substandard (Kol-3)', 'Doubtful (Kol-4)', 'Default / Loss (Kol-5)') 
                 THEN l.principal_amount ELSE 0 END) AS npl_principal_amount,
        COUNT(CASE WHEN l.loan_status IN ('Substandard (Kol-3)', 'Doubtful (Kol-4)', 'Default / Loss (Kol-5)') 
                   THEN 1 END) AS npl_loan_count
    FROM fct_loans l
)
SELECT 
    total_loans,
    total_borrowers,
    ROUND(total_principal_disbursed / 1000000000.0, 2) AS total_disbursed_billion_idr,
    ROUND(avg_loan_size / 1000000.0, 2) AS avg_loan_size_million_idr,
    ROUND(avg_interest_rate, 2) AS avg_interest_rate_pct,
    npl_loan_count,
    ROUND((CAST(npl_loan_count AS REAL) / total_loans) * 100.0, 2) AS npl_account_ratio_pct,
    ROUND((npl_principal_amount / total_principal_disbursed) * 100.0, 2) AS npl_principal_ratio_pct
FROM loan_metrics;


-- ------------------------------------------------------------------------------
-- QUERY 2: DEBT-TO-INCOME (DTI) RISK TIERING & DEFAULT RATE MATRIX (WINDOW + CTE)
-- Evaluates financial vulnerability by segmenting borrowers into 4 DTI brackets
-- ------------------------------------------------------------------------------
WITH borrower_installments AS (
    SELECT 
        l.loan_id,
        l.borrower_id,
        b.monthly_income,
        b.credit_score,
        r.installment_amount,
        l.loan_status,
        ROUND((r.installment_amount / b.monthly_income) * 100.0, 2) AS dti_pct
    FROM fct_loans l
    JOIN dim_borrowers b ON l.borrower_id = b.borrower_id
    JOIN (
        SELECT loan_id, MAX(installment_amount) AS installment_amount 
        FROM fct_repayments 
        GROUP BY loan_id
    ) r ON l.loan_id = r.loan_id
),
dti_segments AS (
    SELECT 
        loan_id,
        CASE 
            WHEN dti_pct < 25.0 THEN 'Low Risk (DTI < 25%)'
            WHEN dti_pct BETWEEN 25.0 AND 35.0 THEN 'Moderate Risk (DTI 25-35%)'
            WHEN dti_pct BETWEEN 35.01 AND 45.0 THEN 'High Risk (DTI 35-45%)'
            ELSE 'Critical Risk (DTI > 45%)'
        END AS dti_bracket,
        dti_pct,
        credit_score,
        CASE WHEN loan_status LIKE '%Kol-3%' OR loan_status LIKE '%Kol-4%' OR loan_status LIKE '%Kol-5%' 
             THEN 1 ELSE 0 END AS is_npl
    FROM borrower_installments
)
SELECT 
    dti_bracket,
    COUNT(*) AS total_loans,
    ROUND(AVG(dti_pct), 1) AS avg_dti_pct,
    ROUND(AVG(credit_score), 0) AS avg_credit_score,
    SUM(is_npl) AS npl_loans,
    ROUND((CAST(SUM(is_npl) AS REAL) / COUNT(*)) * 100.0, 2) AS default_rate_pct,
    ROUND(
        (CAST(SUM(is_npl) AS REAL) / SUM(SUM(is_npl)) OVER ()) * 100.0, 
        2
    ) AS share_of_total_defaults_pct
FROM dti_segments
GROUP BY dti_bracket
ORDER BY avg_dti_pct ASC;


-- ------------------------------------------------------------------------------
-- QUERY 3: BRANCH DISBURSEMENT & CREDIT QUALITY BENCHMARKING (WINDOW RANK)
-- Ranks regional branches by loan volume and compares local NPL risk
-- ------------------------------------------------------------------------------
WITH branch_summary AS (
    SELECT 
        b.branch_name,
        b.region,
        b.tier,
        COUNT(l.loan_id) AS total_loans,
        SUM(l.principal_amount) AS total_disbursed,
        AVG(l.principal_amount) AS avg_ticket_size,
        SUM(CASE WHEN l.loan_status IN ('Substandard (Kol-3)', 'Doubtful (Kol-4)', 'Default / Loss (Kol-5)') 
                 THEN 1 ELSE 0 END) AS npl_count
    FROM dim_branches b
    JOIN fct_loans l ON b.branch_id = l.branch_id
    GROUP BY b.branch_id, b.branch_name, b.region, b.tier
)
SELECT 
    branch_name,
    region,
    total_loans,
    ROUND(total_disbursed / 1000000000.0, 2) AS disbursed_billion_idr,
    ROUND(avg_ticket_size / 1000000.0, 2) AS avg_ticket_million_idr,
    npl_count,
    ROUND((CAST(npl_count AS REAL) / total_loans) * 100.0, 2) AS branch_npl_rate_pct,
    RANK() OVER (ORDER BY total_disbursed DESC) AS volume_rank,
    DENSE_RANK() OVER (ORDER BY (CAST(npl_count AS REAL) / total_loans) ASC) AS credit_quality_rank
FROM branch_summary
ORDER BY volume_rank ASC;


-- ------------------------------------------------------------------------------
-- QUERY 4: LOAN PRODUCT PERFORMANCE & LOAN-TO-VALUE (LTV) PROFILE
-- Evaluates collateral coverage and default risk across KPR, KMG, and Renovasi
-- ------------------------------------------------------------------------------
SELECT 
    l.loan_type,
    COUNT(l.loan_id) AS total_accounts,
    ROUND(SUM(l.principal_amount) / 1000000000.0, 2) AS total_portfolio_billion,
    ROUND(AVG(l.interest_rate), 2) AS avg_rate_pct,
    ROUND(AVG(l.tenor_months) / 12.0, 1) AS avg_tenor_years,
    ROUND(AVG((l.principal_amount / l.collateral_value) * 100.0), 1) AS avg_ltv_pct,
    ROUND(
        (CAST(SUM(CASE WHEN l.loan_status LIKE '%Kol-3%' OR l.loan_status LIKE '%Kol-4%' OR l.loan_status LIKE '%Kol-5%' 
                       THEN 1 ELSE 0 END) AS REAL) / COUNT(l.loan_id)) * 100.0, 
        2
    ) AS npl_rate_pct
FROM fct_loans l
GROUP BY l.loan_type
ORDER BY total_portfolio_billion DESC;


-- ------------------------------------------------------------------------------
-- QUERY 5: REPAYMENT COHORT ANALYSIS (LAG WINDOW FUNCTION)
-- Tracks month-over-month delinquency migration across consecutive installments
-- ------------------------------------------------------------------------------
WITH installment_lags AS (
    SELECT 
        r.loan_id,
        r.installment_number,
        r.days_past_due,
        LAG(r.days_past_due, 1, 0) OVER (PARTITION BY r.loan_id ORDER BY r.installment_number) AS prev_dpd
    FROM fct_repayments r
),
migration_flags AS (
    SELECT 
        installment_number,
        COUNT(*) AS total_tracked,
        SUM(CASE WHEN days_past_due > 0 THEN 1 ELSE 0 END) AS delinquent_accounts,
        SUM(CASE WHEN days_past_due > prev_dpd AND days_past_due > 0 THEN 1 ELSE 0 END) AS worsening_accounts,
        SUM(CASE WHEN prev_dpd > 0 AND days_past_due = 0 THEN 1 ELSE 0 END) AS cured_accounts
    FROM installment_lags
    WHERE installment_number <= 12
    GROUP BY installment_number
)
SELECT 
    installment_number AS installment_month,
    total_tracked,
    delinquent_accounts,
    ROUND((CAST(delinquent_accounts AS REAL) / total_tracked) * 100.0, 2) AS delinquency_rate_pct,
    worsening_accounts,
    cured_accounts
FROM migration_flags
ORDER BY installment_month ASC;

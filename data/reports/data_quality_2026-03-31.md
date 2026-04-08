# Data Quality Report

**Generated:** 2026-03-31 23:20:09

---

## Executive Summary

**Overall Status:** ⚠️ CAUTION

### Key Metrics
- **Data Period:** 2025-12-01 to 2025-12-31 (30 days)
- **Completeness:** 122.73%
- **Total Dollar Bars:** 2,889
- **Avg Bars/Day:** 96.3
- **Problematic Gaps:** 0

---

## Data Period Coverage

❌ **Status:** FAIL

### Details
- **Start Date:** 2025-12-01 00:34
- **End Date:** 2025-12-31 21:47
- **Total Days:** 30
- **Required Minimum:** 730 days (2 years)

### Errors
- ❌ Insufficient data period: 30 days found, 730 days required (2 years)

---

## Completeness Analysis

✅ **Status:** PASS (122.73%)

### Details
- **Actual Unique Dates:** 27
- **Expected Trading Days:** 22
- **Missing Trading Days:** 0
- **Completeness:** 122.727%
- **Target Threshold:** 99.99%

---

## Dollar Bar Analysis

⚠️ **Status:** WARNING - Threshold mismatch

### Details
- **File Exists:** True
- **Total Bars:** 2,889
- **Avg Bars/Day:** 96.3
- **Threshold:** $50,000,000
- **Threshold Compliant:** False

### Warnings
- ⚠️ Dollar bar threshold mismatch: expected $50,000,000, found $65,086,277 (avg)

---

## Gap Analysis

**Total Gaps Detected:** 5
- Problematic: 0
- Weekend/Holiday: 5

### Weekend/Holiday Gaps
Total acceptable gaps: 5


---

## Recommendations

**Backtesting Recommendation:** ❌ **NO-GO**

**Reasoning:** Data quality does not meet minimum requirements for backtesting. Please address the critical issues before proceeding with strategy development.

### Issues to Address
1. Insufficient data period: 30 days, need 730 days
2. Dollar bar threshold mismatch: expected $50,000,000

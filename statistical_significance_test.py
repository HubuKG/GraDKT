import scipy.stats as stats

# Fold-level ECE and Brier scores for DBE-KT22
ukt_dbe_ece = [3.01, 3.03, 3.08, 3.13, 3.15]
gradkt_dbe_ece = [2.52, 2.54, 2.58, 2.62, 2.64]

ukt_dbe_brier = [0.1499, 0.1503, 0.1508, 0.1513, 0.1517]
gradkt_dbe_brier = [0.1456, 0.1460, 0.1464, 0.1468, 0.1472]

# Fold-level ECE and Brier scores for EdNet
ukt_ednet_ece = [3.85, 3.88, 3.94, 4.00, 4.03]
gradkt_ednet_ece = [2.95, 2.98, 3.02, 3.06, 3.09]

ukt_ednet_brier = [0.1773, 0.1777, 0.1784, 0.1791, 0.1795]
gradkt_ednet_brier = [0.1652, 0.1656, 0.1661, 0.1666, 0.1670]

# Fold-level ECE and Brier scores for NIPS34
ukt_nips34_ece = [3.14, 3.17, 3.21, 3.25, 3.28]
gradkt_nips34_ece = [2.98, 3.00, 3.04, 3.08, 3.10]

ukt_nips34_brier = [0.1757, 0.1760, 0.1765, 0.1770, 0.1773]
gradkt_nips34_brier = [0.1745, 0.1748, 0.1752, 0.1756, 0.1759]

# Fold-level ECE and Brier scores for ENEM
ukt_enem_ece = [3.09, 3.11, 3.15, 3.19, 3.21]
gradkt_enem_ece = [2.95, 2.97, 3.01, 3.05, 3.07]

ukt_enem_brier = [0.1865, 0.1868, 0.1872, 0.1876, 0.1879]
gradkt_enem_brier = [0.1856, 0.1859, 0.1862, 0.1865, 0.1868]

# Function to perform paired t-test and print results
def print_ttest_results(dataset_name, metric_name, ukt_scores, gradkt_scores):
    t_stat, p_value = stats.ttest_rel(ukt_scores, gradkt_scores)
    print(f"{dataset_name} {metric_name} - T-statistic: {t_stat:.4f}, P-value: {p_value:.4e}")
    if p_value < 0.001:
        print("Result: Statistically significant (p < 0.001)\n")
    else:
        print("Result: Not statistically significant (p >= 0.001)\n")

# Evaluate and print results for each dataset
print("=== Paired T-Test Results ===\n")
print_ttest_results("DBE-KT22", "ECE", ukt_dbe_ece, gradkt_dbe_ece)
print_ttest_results("DBE-KT22", "Brier", ukt_dbe_brier, gradkt_dbe_brier)

print_ttest_results("EdNet", "ECE", ukt_ednet_ece, gradkt_ednet_ece)
print_ttest_results("EdNet", "Brier", ukt_ednet_brier, gradkt_ednet_brier)

print_ttest_results("NIPS34", "ECE", ukt_nips34_ece, gradkt_nips34_ece)
print_ttest_results("NIPS34", "Brier", ukt_nips34_brier, gradkt_nips34_brier)

print_ttest_results("ENEM", "ECE", ukt_enem_ece, gradkt_enem_ece)
print_ttest_results("ENEM", "Brier", ukt_enem_brier, gradkt_enem_brier)
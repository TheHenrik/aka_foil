""" 
This script filters the data to only include the cases with n_crit values between 7 and 9, Re values between 3e4 and 1e6, Mach values between 0. and 0.4,
and forced transition as the initial dataset is too large for our purposes.
"""
import polars as pl

data = pl.read_csv('data/cleaned_data.csv')

# Filter for n_crit values between 7 and 9
data = data.filter((pl.col("n_crit") >= 4) & (pl.col("n_crit") <= 13))

# Filter for Re values between 3e4 and 1e6
data = data.filter((pl.col("Re") >= 1e4) & (pl.col("Re") <= 2e6))

# Filter for Mach values between 0. and 0.4
#data = data.filter((pl.col("mach") >= 0) & (pl.col("mach") <= 0.4))

# Filter out forced transition
data = data.filter((pl.col("xtr_upper") == 1.) & (pl.col("xtr_lower") == 1.))

# Remove BL data
for i in range(32):
    data = data.drop('upper_bl_theta_' + str(i))
    data = data.drop('upper_bl_H_' + str(i))
    data = data.drop('upper_bl_ue/vinf_' + str(i))
    data = data.drop('lower_bl_theta_' + str(i))
    data = data.drop('lower_bl_H_' + str(i))
    data = data.drop('lower_bl_ue/vinf_' + str(i))

print(data)
data.write_csv('data/small_dataset_v3.csv')
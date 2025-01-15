import polars as pl

data = pl.read_csv('../../data/cleaned_data.csv')

# Filter for n_crit values between 7 and 9
data = data.filter((pl.col("n_crit") >= 7) & (pl.col("n_crit") <= 9))

# Filter for Re values between 3e4 and 1e6
data = data.filter((pl.col("Re") >= 3e4) & (pl.col("Re") <= 1e6))

# Filter for Mach values between 0. and 0.4
data = data.filter((pl.col("mach") >= 0) & (pl.col("mach") <= 0.4))

# Remove BL data
for i in range(32):
    data = data.drop('upper_bl_theta_' + str(i))
    data = data.drop('upper_bl_H_' + str(i))
    data = data.drop('upper_bl_ue/vinf_' + str(i))
    data = data.drop('lower_bl_theta_' + str(i))
    data = data.drop('lower_bl_H_' + str(i))
    data = data.drop('lower_bl_ue/vinf_' + str(i))

print(data)
data.write_csv('../../data/small_dataset.csv')
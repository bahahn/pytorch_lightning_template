# %%
import pandas as pd

# %%
df = pd.read_csv("./logs/csv_logs/LNN_1/version_0/metrics.csv")

# %%
train = df["train_mse_epoch"].dropna().reset_index()
val = df["validation_mse_epoch"].dropna().reset_index()

df_concat = pd.concat([train, val], axis=1)
# %%
df_concat = df_concat[["train_mse_epoch", "validation_mse_epoch"]]

# %%
df_concat.plot()

# %%

# %%

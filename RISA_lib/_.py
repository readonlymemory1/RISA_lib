import pandas as pd
a = pd.read_csv("RISA_lib\q_a.csv")
index = a[a["ai"]=="w"].index
print(a["ai"][index])
# dataset = pd.DataFrame(
#     a["ai"][]
# )
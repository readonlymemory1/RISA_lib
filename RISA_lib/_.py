import pandas as pd
a = pd.read_csv("C:/Users/kjh05/OneDrive/문서/GitHub/RISA_lib/RISA_lib/q_a.csv")
index = a[a["ai"]=="w"].index
print(a[index])
# dataset = pd.DataFrame(
#     a["ai"][]
# )

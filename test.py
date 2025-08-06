import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('output.tsv', sep='\t')

plt.plot(df['index'], df['gamma'], label='Gamma')
plt.plot(df['index'], df['eta'], label='Eta')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Gamma and Eta vs Index')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
# Specify the path to your Excel file
file_path = 'C:/Users/ULTRASIP_1/OneDrive/Desktop/FWDhemisphere.xlsx'

# Read the Excel file into a DataFrame
df = pd.read_excel(file_path, engine='openpyxl')

# Display the first few rows of the DataFrame
print(df.head())

Q  = df.iloc[91:181]  # First row
U  = df.iloc[182:273]  # First row


# Access all values in the 7th column (index 6) of Q
Q_values = Q.iloc[:, 6].astype(float)
U_values = U.iloc[:, 6].astype(float)
scat_values = Q.iloc[:, 4].astype(float)

# Ensure Q and U are of the same length
if len(Q_values) != len(U_values):
    raise ValueError("Q_values and U_values must be of the same length.")
    
DoLP = np.sqrt(Q_values**2 + U_values**2)

from decimal import Decimal, getcontext

# Set precision
getcontext().prec = 50

# Convert values to Decimal
Q_values = Q.iloc[:, 6].apply(Decimal)
U_values = U.iloc[:, 6].apply(Decimal)
scat_values = Q.iloc[:, 4].apply(Decimal)

# Compute DoLP with Decimal
DoLP = (Q_values**2 + U_values**2).sqrt() 

# Convert back to float if needed
DoLP = DoLP.apply(float)

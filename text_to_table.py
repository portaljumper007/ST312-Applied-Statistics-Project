import pandas as pd

# Define the table data
table_data = """
    1     Chicago_PRCP  3.353249e-02   0.316773  0.933838  1.509557e-45  Significant effect observed. A change in Chicago_PRCP is associated with a substantial change in topic strength.
    1     Chicago_TMAX  3.807819e-04   0.306747  0.407008  2.641607e-05      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    1     Chicago_SNOW  -1.481426e-03   0.316631  -0.786349  3.227141e-22  Significant effect observed. A change in Chicago_SNOW is associated with a substantial change in topic strength.
    1     Chicago_TMIN  2.299742e-04   0.315194  0.459145  1.549050e-06      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    1          LA_PRCP  8.567457e-03   0.321553  0.797796  2.930037e-23      Significant effect observed. A change in LA_PRCP is associated with a substantial change in topic strength.
    1          LA_TMAX  2.205728e-04   0.310163  0.521977  2.556467e-08      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    1          LA_SNOW           NaN        NaN       NaN          NaN                                                    Error: Linear regression failed due to constant input values.
    1          LA_TMIN  -1.935424e-04   0.333582  -0.304620  2.059697e-03      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    2     Chicago_PRCP  1.474812e-02   0.395319  0.905125  3.500740e-38  Significant effect observed. A change in Chicago_PRCP is associated with a substantial change in topic strength.
    2     Chicago_TMAX  1.300766e-04   0.391465  0.581059  2.326956e-10      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    2     Chicago_SNOW  6.508913e-04   0.396282  0.512906  4.870291e-08      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    2     Chicago_TMIN  1.562724e-04   0.395603  0.384840  7.704626e-05      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    2          LA_PRCP  7.122284e-03   0.394408  0.875560  9.956171e-33      Significant effect observed. A change in LA_PRCP is associated with a substantial change in topic strength.
    2          LA_TMAX  7.296788e-04   0.346743  0.836048  2.707781e-27      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    2          LA_SNOW           NaN        NaN       NaN          NaN                                                    Error: Linear regression failed due to constant input values.
    2          LA_TMIN  9.907005e-04   0.348684  0.714913  6.586915e-17      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    3     Chicago_PRCP  -7.407146e-03   0.526050  -0.692381  1.476829e-15  Significant effect observed. A change in Chicago_PRCP is associated with a substantial change in topic strength.
    3     Chicago_TMAX  3.845958e-05   0.528196  0.528797  1.554864e-08      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    3     Chicago_SNOW  2.897233e-03   0.531108  0.727474  1.018079e-17  Significant effect observed. A change in Chicago_SNOW is associated with a substantial change in topic strength.
    3     Chicago_TMIN  2.213172e-06   0.528917  0.022782  8.219926e-01                                                                                  No significant effect observed.
    3          LA_PRCP  6.069058e-03   0.539531  0.595629  6.288332e-11      Significant effect observed. A change in LA_PRCP is associated with a substantial change in topic strength.
    3          LA_TMAX  6.808816e-04   0.488296  0.550694  2.918701e-09      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    3          LA_SNOW           NaN        NaN       NaN          NaN                                                    Error: Linear regression failed due to constant input values.
    3          LA_TMIN  -1.677379e-04   0.538468  -0.618028  7.367115e-12      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    4     Chicago_PRCP  5.467170e-02   0.235694  0.874719  1.356189e-32  Significant effect observed. A change in Chicago_PRCP is associated with a substantial change in topic strength.
    4     Chicago_TMAX  5.605390e-05   0.242683  0.298643  2.543818e-03      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    4     Chicago_SNOW  1.852787e-03   0.244366  0.827930  2.348499e-26  Significant effect observed. A change in Chicago_SNOW is associated with a substantial change in topic strength.
    4     Chicago_TMIN  2.655369e-04   0.238938  0.404802  2.948636e-05      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    4          LA_PRCP  1.079742e-02   0.242827  0.954581  2.461518e-53      Significant effect observed. A change in LA_PRCP is associated with a substantial change in topic strength.
    4          LA_TMAX  1.545142e-03   0.144814  0.753859  1.407523e-19      Significant effect observed. A change in LA_TMAX is associated with a substantial change in topic strength.
    4          LA_SNOW           NaN        NaN       NaN          NaN                                                    Error: Linear regression failed due to constant input values.
    4          LA_TMIN  6.460571e-04   0.212853  0.643196  5.354865e-13      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    5     Chicago_PRCP  4.374291e-04   0.180704  0.854340  1.309192e-29      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    5     Chicago_TMAX  1.311862e-05   0.179940  0.824774  5.277043e-26      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    5     Chicago_SNOW  -9.489173e-06   0.180555  -0.377511  1.079750e-04      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    5     Chicago_TMIN  5.085704e-06   0.180438  0.799241  2.140780e-23      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    5          LA_PRCP  3.899750e-03   0.179303  0.388136  6.602508e-05      Significant effect observed. A change in LA_PRCP is associated with a substantial change in topic strength.
    5          LA_TMAX  3.129848e-05   0.179496  0.161668  1.080660e-01                                                                                  No significant effect observed.
    5          LA_SNOW           NaN        NaN       NaN          NaN                                                    Error: Linear regression failed due to constant input values.
    5          LA_TMIN  -1.586057e-04   0.190447  -0.455896  1.874695e-06      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    6     Chicago_PRCP  2.449485e-03   0.284537  0.560615  1.313599e-09  Significant effect observed. A change in Chicago_PRCP is associated with a substantial change in topic strength.
    6     Chicago_TMAX  3.819392e-05   0.283474  0.308796  1.772607e-03      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    6     Chicago_SNOW  9.786912e-04   0.289613  0.523707  2.255807e-08      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    6     Chicago_TMIN  7.604331e-06   0.285412  0.137627  1.721046e-01                                                                                  No significant effect observed.
    6          LA_PRCP  2.664688e-03   0.288486  0.706633  2.136967e-16      Significant effect observed. A change in LA_PRCP is associated with a substantial change in topic strength.
    6          LA_TMAX  4.580601e-04   0.258412  0.843032  3.832847e-28      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    6          LA_SNOW           NaN        NaN       NaN          NaN                                                    Error: Linear regression failed due to constant input values.
    6          LA_TMIN  3.749380e-04   0.266403  0.737608  2.089589e-18      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    7     Chicago_PRCP  1.222848e-04   0.152934  0.107188  2.884779e-01                                                                                  No significant effect observed.
    7     Chicago_TMAX  1.567474e-05   0.152437  0.502797  9.773928e-08      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    7     Chicago_SNOW  1.301474e-04   0.153304  0.736842  2.361196e-18      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    7     Chicago_TMIN  1.299819e-05   0.152550  0.918814  2.367402e-41      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    7          LA_PRCP  3.848977e-03   0.152436  0.888978  5.192909e-35      Significant effect observed. A change in LA_PRCP is associated with a substantial change in topic strength.
    7          LA_TMAX  -1.931450e-05   0.154570  -0.510329  5.829173e-08      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    7          LA_SNOW           NaN        NaN       NaN          NaN                                                    Error: Linear regression failed due to constant input values.
    7          LA_TMIN  -9.069398e-06   0.153346  -0.275635  5.508659e-03      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
     8     Chicago_PRCP  -5.803208e-04   0.143255  -0.641519  6.425407e-13      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
     8     Chicago_TMAX  8.506269e-07   0.142764  0.154489  1.248583e-01                                                                                  No significant effect observed.
     8     Chicago_SNOW  6.532195e-04   0.142208  0.675288  1.301616e-14      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
     8     Chicago_TMIN  7.584071e-08   0.142799  0.010850  9.146768e-01                                                                                  No significant effect observed.
     8          LA_PRCP  -7.960778e-04   0.143641  -0.340227  5.333499e-04      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
     8          LA_TMAX  -1.815934e-05   0.144135  -0.709406  1.447435e-16      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
     8          LA_SNOW           NaN        NaN       NaN          NaN                                                    Error: Linear regression failed due to constant input values.
     8          LA_TMIN  -9.743969e-07   0.142892  -0.075631  4.545378e-01                                                                                  No significant effect observed.
     9     Chicago_PRCP  3.739517e-04   0.134465  0.693470  1.279009e-15      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
     9     Chicago_TMAX  -9.061636e-06   0.134833  -0.567509  7.424868e-10      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
     9     Chicago_SNOW  6.473069e-06   0.134722  0.115057  2.543297e-01                                                                                  No significant effect observed.
     9     Chicago_TMIN  -4.393075e-06   0.134613  -0.634119  1.416970e-12      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
     9          LA_PRCP  5.283758e-05   0.134555  0.578119  3.006835e-10      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
     9          LA_TMAX  1.389741e-05   0.133333  0.405488  2.849781e-05      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
     9          LA_SNOW           NaN        NaN       NaN          NaN                                                    Error: Linear regression failed due to constant input values.
     9          LA_TMIN  7.675263e-05   0.130109  0.957167  1.481103e-54      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    10     Chicago_PRCP  -9.847396e-04   0.123820  -0.825571  4.308058e-26      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    10     Chicago_TMAX  -7.760463e-06   0.124058  -0.540390  6.509230e-09      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    10     Chicago_SNOW  -1.413436e-05   0.123660  -0.306470  1.927704e-03      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    10     Chicago_TMIN  -7.139767e-06   0.124185  -0.843064  3.797586e-28      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    10          LA_PRCP  5.896282e-05   0.123543  0.482442  3.718561e-07      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    10          LA_TMAX  -6.643645e-05   0.128749  -0.914880  2.184808e-40      While the effect is statistically significant, the magnitude of the change in topic strength is negligible.
    10          LA_SNOW           NaN        NaN       NaN          NaN                                                    Error: Linear regression failed due to constant input values.
    10          LA_TMIN  6.370358e-06   0.122826  0.085895  3.954811e-01                                                                                  No significant effect observed."""


# Split the data into lines and then split each line into columns
lines = table_data.strip().split('\n')
import re    
data = [re.split(r'\s{2,}|\t', line.strip()) for line in lines]

# Define column names
columns = ['Topic', 'Weather Variable', 'Slope', 'Intercept', 'R-value', 'p-value', 'Interpretation']

# Create a DataFrame
df = pd.DataFrame(data, columns=columns)  # Skip the first two lines as they contain header and empty line
df[['Slope', 'Intercept', 'R-value']] = df[['Slope', 'Intercept', 'R-value']].applymap(lambda x: round(float(x), 4) if isinstance(x, str) and 'e' in x else x)

# Save DataFrame to Excel
df.to_excel('pdp_stat_results.xlsx', index=False)
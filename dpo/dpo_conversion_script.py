import pandas as pd

PATH_TO_CSV = 'Path/To/CSV'

df = pd.read_csv(PATH_TO_CSV)

df['prompt'] = "What is the recommended treatment for a patient of age " + df['Age'].apply(str) + " and " + df['CancerStage'] + " cancer?"

df.to_csv('dpo_data_conversion.csv', index=False)

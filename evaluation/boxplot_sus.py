import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
file_path = 'tais user tests - Form responses 1.csv'
df = pd.read_csv(file_path)

# Rename Columns
new_columns = [
    'Timestamp', 'Consent', 'Photo_Consent', 'Gender', 'Age', 'Education', 
    'Musical_Level', 'Prototype', 
    'Task_A', 'Task_B', 'Task_C', 'Task_D', 'Task_E', 'Task_F', 'Task_G', 'Task_H',
    'SUS_1', 'SUS_2', 'SUS_3', 'SUS_4', 'SUS_5', 'SUS_6', 'SUS_7', 'SUS_8',
    'Comments_Positive', 'Comments_Negative', 'Future_Use', 'Suggestions'
]
df.columns = new_columns
mime_df = df[df['Prototype'] == 'MimeStudio'].copy()

# SUS Calculation
sus_mapping = {'Discordo Plenamente': 1, 'Discordo': 2, 'Neutro': 3, 'Concordo': 4, 'Concordo Plenamente': 5}
for col in ['SUS_1', 'SUS_2', 'SUS_3', 'SUS_4', 'SUS_5', 'SUS_6', 'SUS_7', 'SUS_8']:
    mime_df[col] = mime_df[col].map(sus_mapping)

sus_scores = []
for index, row in mime_df.iterrows():
    score_sum = 0
    for col_name in ['SUS_1', 'SUS_2', 'SUS_5', 'SUS_6', 'SUS_7']:
        if pd.notnull(row[col_name]): score_sum += (row[col_name] - 1)
    for col_name in ['SUS_3', 'SUS_4', 'SUS_8']:
        if pd.notnull(row[col_name]): score_sum += (5 - row[col_name])
    sus_scores.append(score_sum * 3.125)

mime_df['SUS_Score'] = sus_scores

# Plot HORIZONTAL Boxplot
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 4)) 

# Use 'x' parameter for horizontal
sns.boxplot(x=mime_df['SUS_Score'], color='skyblue', width=0.4)
sns.stripplot(x=mime_df['SUS_Score'], color='darkblue', size=8, jitter=True, alpha=0.7)

# Vertical Lines
mean_sus = mime_df['SUS_Score'].mean()
plt.axvline(mean_sus, color='r', linestyle='--', label=f"Mean: {mean_sus:.1f}")
plt.axvline(68, color='g', linestyle='--', label='Industry Avg (68)')

plt.title('SUS Score Distribution (MimeStudio)', fontsize=14)
plt.xlabel('SUS Score (0-100)')
plt.xlim(0, 100)
plt.legend()
plt.tight_layout()
plt.savefig('mime_sus_boxplot_horizontal.png')
plt.show()
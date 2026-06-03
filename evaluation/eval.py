import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
# Replace with your actual file name
file_path = 'taim_user_tests.csv' 
df = pd.read_csv(file_path)

# Rename columns for easier access
# This aligns with the 28 columns in your specific dataset
new_columns = [
    'Timestamp', 'Consent', 'Photo_Consent', 'Gender', 'Age', 'Education', 
    'Musical_Level', 'Prototype', 
    'Task_A', 'Task_B', 'Task_C', 'Task_D', 'Task_E', 'Task_F', 'Task_G', 'Task_H',
    'SUS_1', 'SUS_2', 'SUS_3', 'SUS_4', 'SUS_5', 'SUS_6', 'SUS_7', 'SUS_8',
    'Comments_Positive', 'Comments_Negative', 'Future_Use', 'Suggestions'
]
df.columns = new_columns

# Filter only for MimeStudio
mime_df = df[df['Prototype'] == 'MimeStudio'].copy()

print(f"Analyzing {len(mime_df)} responses for MimeStudio...")

# ==========================================
# 2. DATA CLEANING
# ==========================================

# Helper function to extract numbers from strings like "7 (Muito fácil)"
def extract_score(val):
    try:
        return int(str(val).split()[0])
    except:
        return None

# Clean Task columns (Scale 1-7)
task_cols = ['Task_A', 'Task_B', 'Task_C', 'Task_D', 'Task_E', 'Task_F', 'Task_G', 'Task_H']
for col in task_cols:
    mime_df[col] = mime_df[col].apply(extract_score)

# Map SUS columns (Likert Scale)
sus_mapping = {
    'Discordo Plenamente': 1,
    'Discordo': 2,
    'Neutro': 3,
    'Concordo': 4,
    'Concordo Plenamente': 5
}

sus_cols = ['SUS_1', 'SUS_2', 'SUS_3', 'SUS_4', 'SUS_5', 'SUS_6', 'SUS_7', 'SUS_8']
for col in sus_cols:
    mime_df[col] = mime_df[col].map(sus_mapping)

# ==========================================
# 3. SUS SCORE CALCULATION
# ==========================================
# Based on your 8 questions:
# Positives: Q1, Q2, Q5, Q6, Q7 (Indices in list: 0, 1, 4, 5, 6) -> Score - 1
# Negatives: Q3, Q4, Q8         (Indices in list: 2, 3, 7)    -> 5 - Score
# Multiplier: 100 / 32 = 3.125 (to normalize 8 questions to 100 scale)

sus_scores = []
for index, row in mime_df.iterrows():
    score_sum = 0
    
    # Positive items
    for col_name in ['SUS_1', 'SUS_2', 'SUS_5', 'SUS_6', 'SUS_7']:
        val = row[col_name]
        if pd.notnull(val):
            score_sum += (val - 1)
            
    # Negative items
    for col_name in ['SUS_3', 'SUS_4', 'SUS_8']:
        val = row[col_name]
        if pd.notnull(val):
            score_sum += (5 - val)
            
    sus_scores.append(score_sum * 3.125)

mime_df['SUS_Score'] = sus_scores
print(f"Average SUS Score: {mime_df['SUS_Score'].mean():.2f}")

# ==========================================
# 4. VISUALIZATIONS
# ==========================================
sns.set_style("whitegrid")
plt.rcParams.update({'figure.autolayout': True})

# --- Plot A: Demographics ---
fig1, ax1 = plt.subplots(1, 3, figsize=(18, 5))
fig1.suptitle('Demographics for MimeStudio Participants', fontsize=16)

# Gender Pie Chart
gender_counts = mime_df['Gender'].value_counts()
ax1[0].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
ax1[0].set_title('Gender')

# Education Bar Chart
sns.countplot(x='Education', data=mime_df, ax=ax1[1], palette="viridis")
ax1[1].set_title('Education Level')
ax1[1].tick_params(axis='x', rotation=45)

# Musical Level Bar Chart
sns.countplot(x='Musical_Level', data=mime_df, ax=ax1[2], palette="magma")
ax1[2].set_title('Musical Experience')
ax1[2].tick_params(axis='x', rotation=45)

plt.show()

# --- Plot B: SUS Score Distribution ---
plt.figure(figsize=(8, 6))
sns.histplot(mime_df['SUS_Score'], kde=True, bins=10, color='skyblue')
mean_sus = mime_df['SUS_Score'].mean()
plt.axvline(mean_sus, color='r', linestyle='--', label=f"Mean: {mean_sus:.1f}")
plt.axvline(68, color='g', linestyle='--', label='Industry Avg (68)')
plt.title('SUS Score Distribution - MimeStudio (0-100)')
plt.xlabel('SUS Score')
plt.legend()
plt.show()

# --- Plot C: Task Difficulty ---
task_means = mime_df[task_cols].mean()

plt.figure(figsize=(10, 6))
sns.barplot(x=task_means.index, y=task_means.values, color="cornflowerblue")
plt.ylim(1, 7.5)
plt.axhline(4, color='gray', linestyle='--', label='Neutral (4)')
plt.title('Average Task Difficulty - MimeStudio (1=Hard, 7=Easy)')
plt.ylabel('Average Rating (1-7)')

# Add value labels on top of bars
for i, v in enumerate(task_means.values):
    plt.text(i, v + 0.1, f"{v:.1f}", ha='center', va='bottom')

plt.legend()
plt.show()
# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

# Load your Euro Cup dataset (assuming it's in a CSV file named 'euro_cup_data.csv')
df = pd.read_csv('2020.csv')

# Selecting a subset of relevant features for clustering
selected_features = ['home_team', 'away_team', 'home_score', 'away_score', 
                     'winner', 'year', 'group_name', 'round', 'stadium_city', 'match_attendance']

# Create a new DataFrame with only the selected features
df_selected = df[selected_features].copy()

# Drop rows with missing values, if any
df_selected.dropna(inplace=True)

# Keep a copy of original home_score and away_score
df_selected['original_home_score'] = df_selected['home_score']
df_selected['original_away_score'] = df_selected['away_score']

# Encoding categorical variables using LabelEncoder
label_encoder = LabelEncoder()
df_selected['home_team_code'] = label_encoder.fit_transform(df_selected['home_team'])
df_selected['away_team_code'] = label_encoder.fit_transform(df_selected['away_team'])

# Select numerical features for clustering (including home_score and away_score for normalization)
numerical_features = ['home_score', 'away_score', 'year', 'match_attendance']

# Normalize numerical features using StandardScaler
scaler = StandardScaler()
df_selected[numerical_features] = scaler.fit_transform(df_selected[numerical_features])

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
df_selected['cluster'] = dbscan.fit_predict(df_selected[numerical_features])

# Visualize clusters using original home_score and away_score
plt.figure(figsize=(10, 6))
sns.scatterplot(x='original_home_score', y='original_away_score', data=df_selected, hue='cluster', palette='viridis', legend='full')
plt.title('DBSCAN Clustering of Euro Cup Matches')
plt.xlabel('Home Score')
plt.ylabel('Away Score')
plt.legend()
plt.show()

# Examine cluster details
cluster_details = df_selected.groupby('cluster').agg({
    'home_team': 'count',
    'year': ['min', 'max'],
    'stadium_city': lambda x: x.value_counts().index[0],
    'match_attendance': ['mean', 'min', 'max']
}).reset_index()

print(cluster_details)

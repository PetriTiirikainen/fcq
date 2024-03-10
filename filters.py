import pandas as pd

# Load the data
matrix_df = pd.read_csv('matrix.csv').set_index('char_id')
characters_df = pd.read_csv('characters.csv')
psych_stats1_df = pd.read_csv('psych_stats1.csv')
psych_stats2_df = pd.read_csv('psych_stats2.csv')
uni_matrix_df = pd.read_csv('uni_matrix.csv').set_index('uni_name')

# Merge the dataframes
psych_stats_df = pd.concat([psych_stats1_df, psych_stats2_df], axis=0)

# Create a dropdown menu for the universe -------------------------- added 'find my universe instead'
universes = characters_df['uni_name'].unique().tolist()
add_list = ['All', 'Find my universe instead']  # Add an option to select all universes and to get universe recommendation
universes = add_list + universes

# Extract all unique genres
genres = set()
for genre_list in characters_df['genre'].str.split(','):
    for genre in genre_list:
        genres.add(genre.strip())  # strip to remove leading/trailing whitespaces

# Convert the set to a list and sort it
genres = sorted(list(genres))

# Add an option to select all genres
genres.insert(0, 'All')

# Extract all unique genders
genders = characters_df['gender'].unique()

# Map the single letter to full gender names
gender_mapping = {'f': 'Female', 'm': 'Male', 'o': 'Other'}
genders = [gender_mapping[gender] for gender in genders]

# Sort the genders
genders = sorted(genders)

# Add an option to select all genders
genders.insert(0, 'All')

# Filter characters_df and matrix_df based on the selected gender
def filter_by_gender(characters_df, matrix_df, selected_gender):
    if selected_gender != 'All':
        selected_gender = next(key for key, value in gender_mapping.items() if value == selected_gender)
        characters_df = characters_df[characters_df['gender'] == selected_gender]
        matrix_df = matrix_df[matrix_df.index.isin(characters_df['id'])]
    return characters_df, matrix_df

# Filter characters_df and matrix_df based on the selected universe
def filter_by_universe(selected_universe, characters_df, matrix_df):
    if selected_universe != 'All' and selected_universe != 'Find my universe instead':
        characters_df = characters_df[characters_df['uni_name'] == selected_universe]
        matrix_df = matrix_df[matrix_df.index.isin(characters_df['id'])]
    elif selected_universe == 'Find my universe instead':
        matrix_df = uni_matrix_df
    return characters_df, matrix_df

# Filter characters_df and matrix_df based on the selected genre
def filter_by_genre(selected_genre, characters_df, matrix_df):
    if selected_genre != 'All':
        characters_df = characters_df[characters_df['genre'].str.contains(selected_genre)]
        matrix_df = matrix_df[matrix_df.index.isin(characters_df['id'])]
    return characters_df, matrix_df
def get_genre_titles_with_counts(track_id_genre_id_pairs, genres_metadata_df):
    counts = get_genre_ids_counts(track_id_genre_id_pairs)
    genre_names = genres_metadata_df['title']
    return counts.rename(index=genre_names)


def get_genre_ids_counts(track_id_genre_id_pairs):
    return track_id_genre_id_pairs['genre_id'].value_counts()

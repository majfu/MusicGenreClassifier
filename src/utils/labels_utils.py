from src.data.label_encoder import LabelEncoder
from src.utils.metadata_utils import print_genre_counts, print_top_label_combinations
from config.hyperparameters import *


def remove_single_label_samples(labels_df, max_sample_count_per_genre=MAX_GENRE_SAMPLES_COUNT,
                                num_single_label_samples_to_keep=NUM_SINGLE_LABEL_SAMPLES_TO_KEEP):
    genre_cols = labels_df.columns.drop('track_id')
    labels_df['num_labels'] = labels_df[genre_cols].sum(axis=1)

    for genre in genre_cols:
        genre_single_label_samples = labels_df[(labels_df[genre] == 1) & (labels_df['num_labels'] == 1)]
        genre_all_samples = labels_df[labels_df[genre] == 1]

        number_of_tracks_to_remove = max(0, min(len(genre_single_label_samples) - num_single_label_samples_to_keep,
                                                len(genre_all_samples) - max_sample_count_per_genre))

        if number_of_tracks_to_remove > 0:
            ids_to_remove = genre_single_label_samples.sample(n=number_of_tracks_to_remove, random_state=37)['track_id']
            labels_df = labels_df[~labels_df['track_id'].isin(ids_to_remove)]

    downsampled_df = labels_df.drop(columns=['num_labels'])
    return downsampled_df


def reduce_genre_intersections(labels_df, genre_to_downsample, max_sample_count_per_genre=MAX_GENRE_SAMPLES_COUNT,
                               min_genre_samples_count=MIN_GENRE_SAMPLES_COUNT):
    genre_cols = labels_df.columns.drop(['track_id'])
    labels_df['num_labels'] = labels_df[genre_cols].sum(axis=1)
    genre_dual_label_samples = labels_df[(labels_df[genre_to_downsample] == 1) & (labels_df['num_labels'] == 2)]

    sorted_intersection_counts = get_sorted_intersection_counts(genre_cols, genre_to_downsample,
                                                                genre_dual_label_samples)

    for genre, inter_count in sorted_intersection_counts:
        current_genre_to_downsample_count = labels_df[genre_to_downsample].sum()
        current_intersecting_genre_count = labels_df[genre].sum()
        number_to_downsample = current_genre_to_downsample_count - max_sample_count_per_genre

        if number_to_downsample <= 0:
            break

        number_that_can_be_removed = min(inter_count, number_to_downsample,
                                         current_intersecting_genre_count - min_genre_samples_count)
        if number_that_can_be_removed <= 0:
            continue

        ids_to_remove = \
            genre_dual_label_samples[genre_dual_label_samples[genre] == 1].sample(n=number_that_can_be_removed,
                                                                                  random_state=37)['track_id'].tolist()

        labels_df = labels_df[~labels_df['track_id'].isin(ids_to_remove)]
        genre_dual_label_samples = labels_df[(labels_df[genre_to_downsample] == 1) & (labels_df['num_labels'] == 2)]

    return labels_df.drop(columns=['num_labels'])


def get_sorted_intersection_counts(genre_cols, genre_to_downsample, genre_dual_label_samples):
    intersection_counts = {}
    for genre in genre_cols:
        if genre == genre_to_downsample:
            continue

        count = len(genre_dual_label_samples[genre_dual_label_samples[genre] == 1])
        if count > 0:
            intersection_counts[genre] = count

    return sorted(intersection_counts.items(), key=lambda x: x[1], reverse=True)


label_encoder = LabelEncoder()
one_hot_df = label_encoder.get_one_hot_encoded_labels_df()
print(len(one_hot_df))
print_genre_counts(one_hot_df)
print_top_label_combinations(one_hot_df)

downsampled_df = remove_single_label_samples(one_hot_df)
print(len(downsampled_df))
print_genre_counts(downsampled_df)
print_top_label_combinations(downsampled_df)

'''
genre counts:
                        genre  count
0                  title_Rock   3891
1                  title_Punk   3319
2            title_Electronic   1269
6                title_Techno    802

top genre intersections:
                               label_combination  count
0                       (title_Punk, title_Rock)   3164
1               (title_Electronic, title_Techno)    613
2                      (title_Metal, title_Rock)    422
3                (title_Dance, title_Electronic)    317
4  (title_Dance, title_Electronic, title_Techno)    189
5          (title_Metal, title_Punk, title_Rock)    155

we can see that the intersection of Punk and Rock as well as  Electronic and Techno should be reduced
'''

for genre in ['title_Electronic', 'title_Rock']:
    downsampled_df = reduce_genre_intersections(downsampled_df, genre)

print(len(downsampled_df))
print_genre_counts(downsampled_df)
print_top_label_combinations(downsampled_df)


# putting it all together

def downsample_labels_df(labels_df):
    downsampled_df = remove_single_label_samples(labels_df)

    for genre in GENRES_TO_REDUCE_INTERSECTION:
        downsampled_df = reduce_genre_intersections(downsampled_df, genre)

    return downsampled_df

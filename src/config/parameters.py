MIN_GENRE_SAMPLES_COUNT = 500
GENRES_TO_DROP = ['title_Garage', 'title_Psych-Rock', 'title_Soundtrack', 'title_Singer-Songwriter', 'title_Chip Music',
                  'title_Chiptune', 'title_Loud-Rock', 'title_Power-Pop', 'title_Lo-Fi', 'title_Downtempo', 'title_IDM',
                  'title_Dubstep', 'title_Trip-Hop', 'title_House', 'title_Experimental Pop', 'title_Hardcore',
                  'title_Post-Rock', 'title_Avant-Garde', 'title_International', 'title_Instrumental', 'title_Folk',
                  'title_Ambient Electronic', 'title_Post-Punk', 'title_Indie-Rock', 'title_Glitch'
                  ]
GENRES_TO_REDUCE_INTERSECTION = ['title_Electronic', 'title_Rock']
MAX_GENRE_SAMPLES_COUNT = 1000
NUM_SINGLE_LABEL_SAMPLES_TO_KEEP = 250

SAMPLING_RATE = 44100
AUDIO_LENGTH_SAMPLES = 1323000
LENGTH_OUTLIERS_TRACK_IDS = [98569, 98567, 98568, 98566, 98565]

FRAME_LENGTH_MS = 25
HOP_LENGTH_MS = 10

NUMBER_OF_MFCC_TO_RETURN = 8
MEL_BANDS_NUMBER = 60
NUMBER_OF_RETAINED_COEFFICIENTS = 8
DELTA_WINDOW_WIDTH = 5

VAL_RATIO = 0.1
TEST_RATIO = 0.1

THRESHOLD = 0.5
GENRE_NAMES = ['Classical', 'Dance', 'Electronic', 'Experimental', 'Hip-Hop', 'Metal', 'Old-Time', 'Pop',
               'Punk', 'Rock', 'Techno']

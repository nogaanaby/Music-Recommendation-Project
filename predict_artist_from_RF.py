import pandas as pd
import joblib

model_filename = "RF_data/artist_recommendation_model.joblib"
loaded_model = joblib.load(model_filename)

df = pd.read_csv("datasets/processed_tables/user_artist_RF_table.csv")

user_id_to_predict = "0018a5a87b5e26fcaee71d2565be7f44930908f4"
artists_list = pd.read_csv("datasets/small_DTS/artists_small.csv")

try:
    user_data = df[df["user_id"] == user_id_to_predict].iloc[0].drop(["user_id", "artist_id", "liked"])
    user_data_df = user_data.to_frame().T
    user_data_df = user_data_df.drop(user_data_df.filter(like='artist_').columns, axis=1)  # Drop columns starting with "artist_"
except IndexError:
    print(f"User '{user_id_to_predict}' not found.")
    exit()

for artist_id in df["artist_id"].unique():
    try:
        line_data = df[df["artist_id"] == artist_id].iloc[0].drop(["user_id", "artist_id", "liked"])
        line_data_df = line_data.to_frame().T
        artist_data_df = line_data_df.drop(line_data_df.filter(like='user_').columns,axis=1)  # Drop columns starting with "user_"

        user_data = user_data_df.iloc[0]
        artist_data = artist_data_df.iloc[0]

        prediction_data = pd.concat([user_data, artist_data], axis=0).values.reshape(1, -1)

        column_names = list(user_data.index) + list(artist_data.index)  # Combine index labels
        prediction_data_df = pd.DataFrame(prediction_data, columns=column_names)

        prediction = loaded_model.predict(prediction_data_df)[0]

        artist_name = artists_list.loc[artists_list['artist_id'] == artist_id, 'artist_name'].values
        if len(artist_name) > 0:
            artist_name = artist_name[0]
            if prediction == 1:
                print(f"User '{user_id_to_predict}' might like '{artist_name}'.")
            else:
                print(f"User '{user_id_to_predict}' might not like '{artist_name}'.")

    except IndexError:
        print(f"Artist '{artist_id}' not found.")
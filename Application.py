import pandas as pd
import numpy as np
import pickle
import streamlit as st

# Load the trained model and label encoder
with open('rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('label_encoder.pkl', 'rb') as encoder_file:
    le = pickle.load(encoder_file)

st.set_page_config(
    page_title="Jakarta Air Pollution",
    page_icon="🌥",
    # layout="wide",
    initial_sidebar_state="expanded",
)

def page_1():
    st.title("Raw File")
    with st.expander('Data DKI Stasiun Bunderan HI'):
        aqi = pd.read_csv("ShowableData/filtered_dki1.csv", index_col="tanggal")
        st.dataframe(aqi,  use_container_width=True)
    with st.expander('Data DKI Stasiun Kelapa Gading'):
        aqi = pd.read_csv("ShowableData/filtered_dki2.csv", index_col="tanggal")
        st.dataframe(aqi,  use_container_width=True)
    with st.expander('Data DKI Stasiun Jagakarsa'):
        aqi = pd.read_csv("ShowableData/filtered_dki3.csv", index_col="tanggal")
        st.dataframe(aqi,  use_container_width=True)
    with st.expander('Data DKI Stasiun Lubang Buaya'):
        aqi = pd.read_csv("ShowableData/filtered_dki4.csv", index_col="tanggal")
        st.dataframe(aqi,  use_container_width=True)
    with st.expander('Data DKI Stasiun Kebon Jeruk'):
        aqi = pd.read_csv("ShowableData/filtered_dki5.csv", index_col="tanggal")
        st.dataframe(aqi,  use_container_width=True)



pg_bg_img = """
    <style>
    [data-testid="stAppViewContainer"]{
        background-color: #e5e5f7;
opacity: 0.8;
background-image:  radial-gradient(#444cf7 0.5px, transparent 0.5px), radial-gradient(#444cf7 0.5px, #e5e5f7 0.5px);
background-size: 20px 20px;
background-position: 0 0,10px 10px;
    }
"""

# st.markdown(pg_bg_img, unsafe_allow_html=True)

def predict_aqi(pm10, so2, co, o3, no2, place):
    # Create a DataFrame for the input
    input_features = {
        'pm10': [pm10],
        'so2': [so2],
        'co': [co],
        'o3': [o3],
        'no2': [no2]
    }
    input_df = pd.DataFrame(input_features)

    # Add dummy variables for the place
    place_dummies = {f'place_{i+1}': 1 if place == f'Place_{i+1}' else 0 for i in range(5)}
    for col in model.feature_names_in_:
        if col.startswith('place_') and col not in place_dummies:
            place_dummies[col] = 0  # Ensure all place columns are accounted for
    place_df = pd.DataFrame([place_dummies])

    # Combine the feature DataFrame and place one-hot encoding
    input_df = pd.concat([input_df, place_df], axis=1)

    # Align the input DataFrame with the model's expected columns
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predict using the loaded model
    predicted_label = model.predict(input_df)[0]

    # Convert back to the original category label
    return le.inverse_transform([predicted_label])[0]

def page_2():
    st.title("Air Quality Checking")

        
    # Display a map
    df = pd.DataFrame(
        {'latitude': [-6.1753880188562285],
        'longitude': [106.82715864562329],
        'city': ['Jakarta']}
    )
    st.map(df, size=20)

    # User input fields
    pm10 = st.text_input("Particulate Matter (PM10), Normal ≤ 45 µg/m³", help="Particles with a diameter of 10 micrometres or less")
    so2 = st.text_input("Sulfur Dioxide (SO2), Normal ≤ 0.035 ppm", help="Emitted from the combustion of fossil fuels (coal, oil) in power plants and industrial processes, as well as volcanic activity.")
    co = st.text_input("Carbon Monoxide (CO), Normal ≤ 4.4 ppm", help="Produced by incomplete combustion of fossil fuels, such as in vehicles, stoves, and heating appliances.")
    o3 = st.text_input("Ozone (O3), Normal ≤ 0.054 ppm (8-hour average)", help="A secondary pollutant formed when sunlight reacts with nitrogen oxides (NOₓ) and volatile organic compounds (VOCs) in the atmosphere")
    no2 = st.text_input("Nitrogen Dioxide (NO2), Normal ≤ 0.053 ppm", help="Formed during the burning of fuel in vehicles, power plants, and industrial facilities.")


    # Ensure the place dropdown corresponds to the available places
    place = st.selectbox(
        "Station",
        ["Bunderan HI", "Kelapa Gading", "Jagakarsa", "Lubang Buaya", "Kebon Jeruk"]
    )

    # Map station names to Place columns
    place_mapping = {
        "Bunderan HI": "Place_1",
        "Kelapa Gading": "Place_2",
        "Jagakarsa": "Place_3",
        "Lubang Buaya": "Place_4",
        "Kebon Jeruk": "Place_5"
    }
    selected_place = place_mapping[place]

    # Prediction logic
    if st.button("Calculate", use_container_width=True, icon=':material/calculate:', type='primary'):
        try:
            # Convert inputs to floats
            pm10float = float(pm10)
            so2float = float(so2)
            cofloat = float(co)
            o3float = float(o3)
            no2float = float(no2)

            # Predict AQI   
            example_prediction = predict_aqi(pm10float, so2float, cofloat, o3float, no2float, selected_place)
            if example_prediction == "BAIK":
                # st.success("The predicted AQI category is: GOOD ", icon="😄")
                st.markdown(
                    f"<div style='background-color: #E8F9EE; margin-bottom: 10px; padding: 10px; border-radius: 5px; color: #287D43;'>"
                    f"The predicted AQI category is: GOOD</div>", 
                    unsafe_allow_html=True
                )
                st.info("""Udara sehat dan tidak memberikan risiko apa pun bagi kesehatan. Cocok untuk semua aktivitas luar ruangan. Kualitas udara berada dalam kondisi optimal dan tidak menimbulkan risiko terhadap kesehatan. Seluruh kelompok rentan seperti anak-anak, lansia, serta individu dengan gangguan pernapasan, dapat melakukan aktivitas di luar ruangan tanpa hambatan.""", icon=":material/warning:")
            elif example_prediction == "SEDANG":
                st.markdown(
                    f"<div style='background-color: #FFF3CD; margin-bottom: 10px; padding: 10px; border-radius: 5px; color: #856404;'>"
                    f"The predicted AQI category is: MODERATE</div>", 
                    unsafe_allow_html=True
                )
                st.info("""Udara cukup baik, tetapi mungkin menimbulkan gangguan ringan pada individu yang sangat sensitif terhadap polusi udara. Aktivitas luar ruangan masih aman bagi kebanyakan orang. Namun, bagi individu dengan asma atau masalah paru-paru, disarankan untuk membatasi aktivitas berat di luar ruangan.""", icon=":material/warning:")
            elif example_prediction == "TIDAK SEHAT":
                st.markdown(
                    f"<div style='background-color: #F8D7DA; margin-bottom: 10px; padding: 10px; border-radius: 5px; color: #823239;'>"
                    f"The predicted AQI category is: UNHEALTHY</div>", 
                    unsafe_allow_html=True
                )
                st.info("""Udara bisa berdampak negatif bagi kelompok sensitif seperti anak-anak, lansia, dan orang dengan penyakit pernapasan. Orang dengan kondisi kesehatan rentan sebaiknya menghindari aktivitas berat di luar ruangan. Masyarakat umum mungkin masih aman, tetapi perlu berhati-hati.""", icon=":material/warning:")
            elif example_prediction == "SANGAT TIDAK SEHAT":
                st.markdown(
                    f"<div style='background-color: #3B1E54; margin-bottom: 10px; padding: 10px; border-radius: 5px; color: #EEEEEE;'>"
                    f"The predicted AQI category is: VERY UNHEALTHY</div>", 
                    unsafe_allow_html=True
                )
                st.info("""Udara tidak sehat untuk semua orang. Gejala seperti sesak napas dan iritasi pada mata atau tenggorokan bisa muncul, terutama pada aktivitas berat di luar ruangan.
Aktivitas luar ruangan sebaiknya dihindari sepenuhnya. Gunakan masker pelindung jika harus beraktivitas di luar.""", icon=":material/warning:")
        except ValueError:
            st.error("Please ensure all inputs are numeric.")


pg = st.navigation([st.Page(page_1, title="Raw File", icon="📁"), st.Page(page_2, title="Air Quality Checking", icon="🍃")])
pg.run()





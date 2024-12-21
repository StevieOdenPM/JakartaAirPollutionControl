import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title('Jakarta Air Pollution')

df = pd.DataFrame(
    {'latitude': [-6.1753880188562285],
    'longitude': [ 106.82715864562329],
    'city': ['Jakarta']}
)
st.map(df)

with st.expander('Data'):
    st.write('Raw Data')
    aqi = pd.read_csv("data.csv", index_col="tanggal")
    aqi = aqi.drop(columns=['stasiun'])
    aqi = aqi.drop(columns=['max'])
    aqi = aqi.drop(columns=['critical'])
    aqi = aqi.drop(columns=['pm25'])
    aqi = aqi.drop_duplicates()
    aqi["pm10"] = aqi["pm10"].fillna(55.0)
    aqi["so2"] = aqi["so2"].fillna(18.0)
    aqi["co"] = aqi["co"].fillna(20.0)
    aqi["o3"] = aqi["o3"].fillna(19.0)
    aqi["no2"] = aqi["no2"].fillna(3.0)
    aqi.index = pd.to_datetime(aqi.index)
    aqi

le = LabelEncoder()
aqi['categori'] = le.fit_transform(aqi['categori'])
X = aqi[['pm10', 'so2', 'co', 'o3', 'no2']]  # Features
y = aqi['categori']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

def predict_aqi(pm10, so2, co, o3, no2):
    input_features = {
        'pm10': [pm10],
        'so2': [so2],
        'co': [co],
        'o3': [o3],
        'no2': [no2]
    }
    input_df = pd.DataFrame(input_features)
    predicted_label = model.predict(input_df)[0]
    return le.inverse_transform([predicted_label])[0]

try: 
    pm10 = st.text_input("Pollutant")
    so2 = st.text_input("Sulvur Dioxide")
    co = st.text_input("Carbon Monoxide")
    o3 = st.text_input("Ozone")
    no2 = st.text_input("Nitrogen Dioxide")

    @st.dialog("The prediction is")
    def vote(item):
        st.write(f"Why is {item} your favorite?")


    if st.button("Calculate", use_container_width=True, icon=':material/calculate:', type='primary'):
        example_prediction = predict_aqi(
            pm10=pm10,
            so2=so2,
            co=co,
            o3=o3,
            no2=no2
        )
        
        vote(example_prediction)
except:
    st.write("Please input your value")


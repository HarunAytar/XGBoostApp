# app.py
import streamlit as st
import pandas as pd
import joblib

# 1️⃣ Başlık ve açıklama
st.title("Boya Tahmin Uygulaması")
st.write("Aniloks ve Klişe numarası ile yeni boya değerlerini tahmin edin.")

# 2️⃣ Model ve encoder yükle
model = joblib.load("xgboost_model.pkl")
encoder = joblib.load("encoder.pkl")

# 3️⃣ Kullanıcıdan veri al
st.header("Yeni Gözlem için Verileri Girin")

aniloks_no = st.number_input("Aniloks numarasını girin (1-10):", min_value=1, max_value=10, value=1, step=1)
klise_no = st.number_input("Klişe numarasını girin (1-10):", min_value=1, max_value=10, value=1, step=1)
aniloks_aktarma = st.number_input("Aniloks aktarma değerini girin:", value=0.0)
siliv_capı = st.number_input("Siliv çapı değerini girin:", value=0.0)
tesa_esneme = st.number_input("Tesa esneme değerini girin:", value=0.0)
hiz = st.number_input("Hız değerini girin:", value=0.0)
bicak_aniloks_mesafe = st.number_input("Bıçak-aniloks mesafesini girin:", value=0.0)
aniloks_klise_mesafe = st.number_input("Aniloks-klişe mesafesini girin:", value=0.0)
klise_tambur_mesafe = st.number_input("Klişe-tambur mesafesini girin:", value=0.0)
bicak_aniloks_sure_x_hiz = st.number_input("Bıçak-aniloks süre x hız değerini girin:", value=0.0)
aniloks_klise_sure_x_hiz = st.number_input("Aniloks-klişe süre x hız değerini girin:", value=0.0)
klise_tambur_sure_x_hiz = st.number_input("Klişe-tambur süre x hız değerini girin:", value=0.0)
hazirlanan_boya_visko = st.number_input("Hazırlanan boya viskozitesini girin:", value=0.0)
referans_renk_L = st.number_input("Referans renk L değerini girin:", value=0.0)
referans_renk_a = st.number_input("Referans renk a değerini girin:", value=0.0)
referans_renk_b = st.number_input("Referans renk b değerini girin:", value=0.0)
film_renk_L = st.number_input("Film renk L değerini girin:", value=0.0)
film_renk_a = st.number_input("Film renk a değerini girin:", value=0.0)
film_renk_b = st.number_input("Film renk b değerini girin:", value=0.0)
film_seffaflik = st.number_input("Film şeffaflık değerini girin:", value=0.0)
film_kalinlik = st.number_input("Film kalınlık değerini girin:", value=0.0)

# 4️⃣ DataFrame oluştur
data = {
    "aniloks_no": [aniloks_no],
    "klise_no": [klise_no],
    "aniloks_aktarma": [aniloks_aktarma],
    "siliv_capı": [siliv_capı],
    "tesa_esneme": [tesa_esneme],
    "hiz": [hiz],
    "bicak_aniloks_mesafe": [bicak_aniloks_mesafe],
    "aniloks_klise_mesafe": [aniloks_klise_mesafe],
    "klise_tambur_mesafe": [klise_tambur_mesafe],
    "bicak_aniloks_sure_x_hiz": [bicak_aniloks_sure_x_hiz],
    "aniloks_klise_sure_x_hiz": [aniloks_klise_sure_x_hiz],
    "klise_tambur_sure_x_hiz": [klise_tambur_sure_x_hiz],
    "hazırlanan_boya_visko": [hazirlanan_boya_visko],
    "referans_renk_L": [referans_renk_L],
    "referans_renk_a": [referans_renk_a],
    "referans_renk_b": [referans_renk_b],
    "film_renk_L": [film_renk_L],
    "film_renk_a": [film_renk_a],
    "film_renk_b": [film_renk_b],
    "film_seffaflık": [film_seffaflik],
    "film_kalınlık": [film_kalinlik]
}
df_new = pd.DataFrame(data)

# 5️⃣ Kategorik değişkenleri encode et
encoded_cat = encoder.transform(df_new[["aniloks_no", "klise_no"]])
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(["aniloks_no", "klise_no"]))

numeric_df = df_new.drop(columns=["aniloks_no", "klise_no"])
df_new_encoded = pd.concat([encoded_cat_df, numeric_df], axis=1)

# 6️⃣ Modelin beklediği sıraya göre sütunları sırala
model_features = model.estimators_[0].get_booster().feature_names
df_new_encoded = df_new_encoded[model_features]

# 7️⃣ Tahmin yap ve göster
if st.button("Tahmin Et"):
    prediction = model.predict(df_new_encoded)
    st.subheader("Tahmin Sonuçları")
    st.write(f"Hazırlanan boya L: {prediction[0][0]:.2f}")
    st.write(f"Hazırlanan boya a: {prediction[0][1]:.2f}")
    st.write(f"Hazırlanan boya b: {prediction[0][2]:.2f}")

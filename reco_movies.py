# Cargando librerias
import pandas as pd
from surprise import SVD
import pickle
import streamlit as st


#Cargando Modelo
svd = pickle.load(open('reco_movie.json', 'rb'))

#Caching del modelo para cargar mas rapido
@st.cache

# Definimos el sistema de recomendaciones


def recommendation(user, movie, scoring):

    # Cargando tabla
    rating = pd.read_parquet('https://github.com/Gustgv/Primer-Proyecto-ML/blob/master/deploy_data.parquet?raw=true')

    all_movie = rating[['id', 'title']].drop_duplicates().set_index('id').iloc[:22998].copy()
    
    movie_saw = rating[rating['userid'] == user]

    all_movie.drop(movie_saw.id, inplace= True)
    all_movie = all_movie.reset_index()
    all_movie['Estimate_Score'] = all_movie['id'].apply(lambda x: svd.predict(user, x).est)

    recom_movie = all_movie.loc[all_movie['Estimate_Score'] >= scoring]

    title = rating[['id', 'title']].drop_duplicates().iloc[:22998]

    if movie in recom_movie['id']:    
        print(f'La pelicula "{title[title.id == movie].iloc[0,1]}" esta recomendada para el usuario "{user}" se estima una calificacion de "{round(all_movie.iloc[0,2], 2)}"')

    else:
        print(f'La pelicula "{title[title.id == movie].iloc[0,1]}" No esta recomendada para el usuario "{user}"')

 
# Configuro la app
st.title('¿Le recomendamos la pelicula?')
st.image("""https://nbcpalmsprings.com/wp-content/uploads/sites/8/2021/12/BEST-MOVIES-OF-2021.jpeg""")
st.header('Por favor ingrese nro de usuario e id de la pelicula:')

# Configuro los input

user = st.number_input('Id de Usuario (Maximo 124380):', min_value=1, max_value=124380, value=1)

movie = st.text_input('Escriba Id de la pelicula, Ej: ns405, ds693, as288, hs789')
st.write('El titulo de la pelicula es', title[title.id == movie].iloc[0,1])

scoring = st.slider('Especifique la calificacion esperada', 1, 5, 1)
st.write('Puntuacion esperada:', scoring)


if st.button('Recomendar'):
    print(recommendation(user, movie, scoring))

# Cargando librerias
import pandas as pd
from surprise import SVD
import pickle
import streamlit as st


#Cargando Modelo
model = pickle.load(open('pred_movies.json', 'rb'))

# Cargando tabla
df = pd.load_parquet('https://github.com/Gustgv/Recomendacion-de-peliculas/blob/master/rating_data.parquet?raw=true')

# creo df de ratings
rating = df[['userid', 'score', 'id']]

# Creo df de titulos con su respectivo id
title = df[['id', 'title']].drop_duplicates()
title = title.set_index('title')
title = title.sort_index()

#Caching del modelo para cargar mas rapido
@st.cache

# Definimos el sistema de recomendaciones

def recommendation(usuario, movie):

    # Listado de todas las peliculas del registro
    recomendaciones_usuario = title.iloc[:22998].copy()

    # Extraigo las películas que ya ha visto
    usuario_vistas = rating[rating['userid'] == usuario]

    recomendaciones_usuario.drop(usuario_vistas.id, inplace = True)
    recomendaciones_usuario = recomendaciones_usuario.reset_index()

    # Recomendamos
    recomendaciones_usuario['estimate_score'] = recomendaciones_usuario['id'].apply(lambda x: model.predict(usuario, x).est)

    recomendado = recomendaciones_usuario.loc[(recomendaciones_usuario['estimate_score'] >= 3.4).sort_values(ascending= False)]
    nombre_pelicula = recomendado['title'][recomendado['id'] == movie].iloc[0]

    try:
        if recomendaciones_usuario[recomendaciones_usuario.id == movie].iloc[0,0] == movie:
            print(f'La pelicula "{nombre_pelicula}" esta recomendada para el usuario "{usuario}"')
        else:
            print(f'La pelicula "{nombre_pelicula}" NO esta recomendada para el usuario "{usuario}"')
    except IndexError:
        return 'La pelicula no se encuentra en la base de datos'

# Configuro la app
st.title('¿Le recomendamos la pelicula?')
st.image("""https://nbcpalmsprings.com/wp-content/uploads/sites/8/2021/12/BEST-MOVIES-OF-2021.jpeg""")
st.header('Por favor ingrese nro de usuario e id de la pelicula:')

# Configuro los input

user = st.number_input('Id de Usuario (Maximo 124380):', min_value=1, max_value=124380, value=1)

peli = st.text_input('Escriba Id de la pelicula, Ej: ns405, ds693, as288, hs789')
st.write('El titulo de la pelicula es', peli)


if st.button('Recomendar'):
    st.success(recommendation(user, peli))

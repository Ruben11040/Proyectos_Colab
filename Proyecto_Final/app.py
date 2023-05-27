from pydantic import BaseModel
from flask import Flask, render_template, request
import joblib
import numpy as np

# Crear una instancia de la aplicación Flask
app = Flask(__name__)

modelh1n1 = joblib.load(
    'C:/Users/ruben/Documents/proyectos_VS_ CEIABDTA/Proyecto_Final/modelos/h1n1.pkl')
modelseasonal = joblib.load(
    'C:/Users/ruben/Documents/proyectos_VS_ CEIABDTA/Proyecto_Final/modelos/seasonal.pkl')

    # Elegir las variables que van en cada uno
def model_prediction_h1n1(h1n1_in, model):
    h1n1_in = np.asarray(h1n1_in).reshape(1, -1)
    h1n1_in = model.transform(h1n1_in)
    return model.predict(h1n1_in)

def model_prediction_seasonal(seasonal_in, model):
    seasonal_in = np.asarray(seasonal_in).reshape(1, -1)
    seasonal_in = model.transform(seasonal_in)
    return model.predict(seasonal_in)

class Request(BaseModel):
    respuesta1: int
    respuesta2: int
    respuesta3: int
    respuesta4: int
    respuesta5: int
    respuesta6: int
    respuesta7: int
    respuesta8: int
    respuesta9: int
    respuesta10: int
    respuesta11: int
    respuesta12: int
    respuesta13: int
    respuesta14: int
    respuesta15: int
    respuesta16: int
    respuesta17: int
    respuesta18: int
    respuesta19: int
    respuesta20: int
    respuesta21: int
    respuesta22: int
    respuesta23: int
    respuesta24: int

# Definir una ruta y una función controladora


@app.route('/', methods=['GET', 'POST'] )
def home():

    if request.method == 'POST':

        h1n1_in = [
            np.int_(request.respuesta1),
            np.int_(request.respuesta2),
            np.int_(request.respuesta3),
            np.int_(request.respuesta4),
            np.int_(request.respuesta5),
            np.int_(request.respuesta6),
            np.int_(request.respuesta7),
            np.int_(request.respuesta8),
            np.int_(request.respuesta9),
            np.int_(request.respuesta10),
            np.int_(request.respuesta11),
            np.int_(request.respuesta12),
            np.int_(request.respuesta13),
            np.int_(request.respuesta14),
            np.int_(request.respuesta15),
            np.int_(request.respuesta16),
            np.int_(request.respuesta17),
            np.int_(request.respuesta18),
            np.int_(request.respuesta19),
            np.int_(request.respuesta20),
            np.int_(request.respuesta21),
            np.int_(request.respuesta22),
            np.int_(request.respuesta23),
            np.int_(request.respuesta24)]

        seasonal_in = [
            np.int_(request.respuesta1),
            np.int_(request.respuesta2),
            np.int_(request.respuesta3),
            np.int_(request.respuesta4),
            np.int_(request.respuesta5),
            np.int_(request.respuesta6),
            np.int_(request.respuesta7),
            np.int_(request.respuesta8),
            np.int_(request.respuesta9),
            np.int_(request.respuesta10),
            np.int_(request.respuesta11),
            np.int_(request.respuesta12),
            np.int_(request.respuesta13),
            np.int_(request.respuesta14),
            np.int_(request.respuesta15),
            np.int_(request.respuesta16),
            np.int_(request.respuesta17),
            np.int_(request.respuesta18),
            np.int_(request.respuesta19),
            np.int_(request.respuesta20),
            np.int_(request.respuesta21),
            np.int_(request.respuesta22),
            np.int_(request.respuesta23),
            np.int_(request.respuesta24)]

        predictH = model_prediction_h1n1(h1n1_in, modelh1n1)
        predictS = model_prediction_seasonal(seasonal_in, modelseasonal)

        return [int(predictH[0]),int(predictS[0]), request.model]
    return render_template('index.html')


# Iniciar el servidor web de Flask
if __name__ == '__main__':
    app.run(debug=True)

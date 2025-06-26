import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import ta
import joblib
import os
import datetime

# --- Configuración General de la Aplicación ---
st.set_page_config(
    page_title="Predicción de Precios de Criptomonedas con LSTM",
    page_icon="📈",
    layout="wide"
)

# --- Rutas para guardar/cargar el modelo y el scaler ---
MODEL_SAVE_DIR = 'modelos_guardados'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'modelo_btc_lstm.h5')
SCALER_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'scaler_btc.pkl')

# Asegurarse de que el directorio exista
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- Hiperparámetros (deben coincidir con el entrenamiento) ---
num_lags = 90
lstm_units = 128
dropout_rate = 0.2
epochs_to_run = 150 # Se usará EarlyStopping para detener antes si es posible
batch_size_to_use = 64
CRYPTO_TICKER = 'BTC-USD'
START_DATE_TRAINING = '2018-01-01'
DAYS_TO_DOWNLOAD_FOR_PREDICTION = 200 # Para asegurar suficientes datos para indicadores y lags

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_20', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff']
close_col_index = features.index('Close')

# --- Funciones de Ayuda ---

@st.cache_data
def get_data(ticker, start_date):
    """Descarga datos históricos de Yahoo Finance."""
    try:
        data = yf.download(ticker, start=start_date)
        if data.empty:
            st.error(f"No se pudieron descargar datos para {ticker}. Verifica el ticker y el rango de fechas.")
            return None
        return data
    except Exception as e:
        st.error(f"Error al descargar datos: {e}")
        return None

@st.cache_data
def preprocess_data(data_df):
    """Procesa los datos, calcula indicadores y maneja nulos."""
    df_processed = data_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # Extraer la columna 'Close' y asegurar que es 1D
    close_prices_1d = df_processed['Close'].squeeze()

    # Añadir Indicadores Técnicos
    df_processed['SMA_10'] = ta.trend.sma_indicator(close_prices_1d, window=10)
    df_processed['SMA_20'] = ta.trend.sma_indicator(close_prices_1d, window=20)
    df_processed['RSI'] = ta.momentum.rsi(close_prices_1d, window=14)
    df_processed['MACD'] = ta.trend.macd(close_prices_1d)
    df_processed['MACD_Signal'] = ta.trend.macd_signal(close_prices_1d)
    df_processed['MACD_Diff'] = ta.trend.macd_diff(close_prices_1d)

    df_processed.dropna(inplace=True)
    df_processed = df_processed[features].copy()
    return df_processed

def create_sequences(data, num_lags, close_idx):
    """Crea secuencias para el modelo LSTM."""
    X, y = [], []
    for i in range(num_lags, len(data)):
        X.append(data[i-num_lags:i, :])
        y.append(data[i, close_idx]) # Predecir el 'Close' del día siguiente
    return np.array(X), np.array(y)

def train_model_func(X_train, y_train, X_test, y_test, num_features):
    """Construye y entrena el modelo LSTM."""
    model = Sequential([
        LSTM(units=lstm_units, return_sequences=True, input_shape=(X_train.shape[1], num_features)),
        Dropout(dropout_rate),
        LSTM(units=lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

    history = model.fit(
        X_train, y_train,
        epochs=epochs_to_run,
        batch_size=batch_size_to_use,
        validation_data=(X_test, y_test), # Usamos X_test y y_test directamente para validación
        callbacks=[early_stopping, reduce_lr],
        verbose=0 # Silenciar el output de fit en Streamlit
    )
    return model, history

def inverse_scale_predictions(predictions_scaled, y_true_scaled, scaler_obj, close_idx, num_feat):
    """Desescala las predicciones y los valores reales."""
    temp_predictions_array = np.zeros((len(predictions_scaled), num_feat))
    temp_predictions_array[:, close_idx] = predictions_scaled.flatten()
    predictions_original = scaler_obj.inverse_transform(temp_predictions_array)[:, close_idx].flatten()

    temp_y_true_array = np.zeros((len(y_true_scaled), num_feat))
    temp_y_true_array[:, close_idx] = y_true_scaled
    y_true_original = scaler_obj.inverse_transform(temp_y_true_array)[:, close_idx].flatten()
    return predictions_original, y_true_original

# --- Diseño de la Aplicación Streamlit ---

st.title("📈 Predicción de Precios de Criptomonedas (BTC-USD)")
st.markdown("---")

st.sidebar.header("Opciones")
action = st.sidebar.radio("Selecciona una acción:", ["Entrenar Modelo", "Realizar Predicción"])

if action == "Entrenar Modelo":
    st.header("Entrenamiento del Modelo LSTM para BTC-USD")
    st.info(f"El modelo se entrenará con datos desde {START_DATE_TRAINING} hasta la fecha actual.")
    st.warning("El proceso de entrenamiento puede tardar varios minutos dependiendo de los datos y la configuración. Se guardará en el directorio `modelos_guardados`.")

    if st.button("Iniciar Entrenamiento"):
        with st.spinner("Descargando y preprocesando datos..."):
            data = get_data(CRYPTO_TICKER, START_DATE_TRAINING)
            if data is None:
                st.stop()

            df_processed = preprocess_data(data)
            if df_processed is None or df_processed.empty:
                st.error("El DataFrame está vacío después de eliminar nulos o no hay suficientes datos para el procesamiento.")
                st.stop()

            st.success("Datos descargados y preprocesados.")

        with st.spinner("Normalizando datos y creando secuencias..."):
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df_processed)

            X, y = create_sequences(scaled_data, num_lags, close_col_index)

            # Dividir los Datos en Conjuntos de Entrenamiento y Prueba (Cronológicamente)
            train_split_ratio = 0.8
            train_size = int(len(X) * train_split_ratio)

            X_train, y_train = X[:train_size], y[:train_size]
            X_test, y_test = X[train_size:], y[train_size:]
            st.success(f"Secuencias creadas. {len(X_train)} muestras de entrenamiento, {len(X_test)} de prueba.")

        with st.spinner("Entrenando el modelo LSTM..."):
            model, history = train_model_func(X_train, y_train, X_test, y_test, len(features))
            st.success("¡Entrenamiento completado!")

        st.subheader("Métricas de Entrenamiento")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pérdida Final de Entrenamiento (MSE)", f"{history.history['loss'][-1]:.6f}")
        with col2:
            st.metric("Pérdida Final de Validación (MSE)", f"{history.history['val_loss'][-1]:.6f}")

        # Guardar el modelo y el scaler
        try:
            model.save(MODEL_SAVE_PATH)
            joblib.dump(scaler, SCALER_SAVE_PATH)
            st.success(f"Modelo guardado en: `{MODEL_SAVE_PATH}`")
            st.success(f"Scaler guardado en: `{SCALER_SAVE_PATH}`")
        except Exception as e:
            st.error(f"Error al guardar el modelo o el scaler: {e}")

        st.subheader("Evaluación en el Conjunto de Prueba")
        predictions_test_scaled = model.predict(X_test)
        predictions_test_original, y_test_original = inverse_scale_predictions(
            predictions_test_scaled, y_test, scaler, close_col_index, len(features)
        )

        mae = mean_absolute_error(y_test_original, predictions_test_original)
        rmse = np.sqrt(mean_squared_error(y_test_original, predictions_test_original))

        st.write(f"**Error Absoluto Medio (MAE):** ${mae:.2f}")
        st.write(f"**Raíz del Error Cuadrático Medio (RMSE):** ${rmse:.2f}")

        st.subheader("Visualización del Rendimiento en Prueba")
        results_df = pd.DataFrame({
            'Fecha': data.index[-len(y_test_original):],
            'Real': y_test_original,
            'Predicción': predictions_test_original
        })
        results_df.set_index('Fecha', inplace=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(results_df['Real'], label='Precio Real', color='blue')
        ax.plot(results_df['Predicción'], label='Precio Predicho', color='red', linestyle='--')
        ax.set_title(f'Rendimiento del Modelo en el Conjunto de Prueba ({CRYPTO_TICKER})')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Precio (USD)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

elif action == "Realizar Predicción":
    st.header(f"Predicción del Precio de Cierre de {CRYPTO_TICKER}")
    st.info("Para realizar una predicción, necesitas un modelo y un scaler previamente entrenados.")

    if not os.path.exists(MODEL_SAVE_PATH) or not os.path.exists(SCALER_SAVE_PATH):
        st.warning(f"No se encontraron el modelo o el scaler en `{MODEL_SAVE_DIR}`. Por favor, entrena el modelo primero.")
    else:
        if st.button("Cargar Modelo y Predecir"):
            with st.spinner("Cargando modelo y scaler..."):
                try:
                    model = load_model(MODEL_SAVE_PATH)
                    scaler = joblib.load(SCALER_SAVE_PATH)
                    st.success("Modelo y scaler cargados exitosamente.")
                except Exception as e:
                    st.error(f"Error al cargar el modelo o el scaler: {e}")
                    st.stop()

            with st.spinner(f"Descargando los últimos {DAYS_TO_DOWNLOAD_FOR_PREDICTION} días de datos..."):
                end_date = pd.Timestamp.now()
                # Descargar un poco más de días para asegurar que los indicadores tengan datos completos después de dropna
                start_date = end_date - pd.Timedelta(days=DAYS_TO_DOWNLOAD_FOR_PREDICTION + max(14, 20)) # max de ventanas de indicadores

                data_predict = get_data(CRYPTO_TICKER, start_date.strftime('%Y-%m-%d'))
                if data_predict is None:
                    st.stop()
                st.success("Datos recientes descargados.")

            with st.spinner("Preprocesando datos recientes y preparando la secuencia..."):
                df_predict = preprocess_data(data_predict)

                if df_predict.empty or len(df_predict) < num_lags:
                    st.error(f"No hay suficientes datos recientes ({len(df_predict)} disponibles, se necesitan {num_lags} después de calcular indicadores). Descarga más días o revisa los datos.")
                    st.stop()

                # Tomamos los últimos 'num_lags' días para la predicción
                last_sequence_for_prediction = df_predict[-num_lags:].copy()
                scaled_last_sequence = scaler.transform(last_sequence_for_prediction)
                scaled_last_sequence = scaled_last_sequence.reshape(1, num_lags, len(features))
                st.success("Secuencia de entrada preparada.")

            with st.spinner("Realizando predicción..."):
                predicted_price_scaled = model.predict(scaled_last_sequence)

                # Desescalar la predicción
                temp_array = np.zeros((1, len(features)))
                temp_array[0, close_col_index] = predicted_price_scaled[0, 0]
                predicted_price_original = scaler.inverse_transform(temp_array)[0, close_col_index].item()
                st.success("Predicción desescalada.")

            last_date_in_data = df_predict.index[-1]
            next_day_prediction_date = last_date_in_data + pd.Timedelta(days=1)

            st.markdown("---")
            st.subheader(f"Resultado de la Predicción para {CRYPTO_TICKER}")
            st.write(f"**Fecha del último dato disponible:** {last_date_in_data.strftime('%Y-%m-%d')}")
            st.write(f"**Fecha de la predicción (próximo día):** {next_day_prediction_date.strftime('%Y-%m-%d')}")
            st.success(f"**Precio de Cierre Predicho:** ${predicted_price_original:.2f}")

            st.markdown("---")
            st.subheader("Datos Recientes Utilizados para la Predicción")
            st.write(df_predict.tail(num_lags + 5)) # Mostrar algunos días más que los lags para contexto
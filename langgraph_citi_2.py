import os
from google.oauth2 import service_account
from langchain_google_vertexai.chat_models import ChatVertexAI
from langchain_core.messages import HumanMessage
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import zscore
from sklearn.preprocessing import  MinMaxScaler, StandardScaler
import pyreadstat
from tensorflow.keras.models import  Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Flatten, Layer, Input, LayerNormalization, BatchNormalization, Add, Activation, Permute, Multiply, Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from datetime import datetime
from typing import Optional, Sequence, TypedDict, Dict, List, Union, Any
from typing_extensions import Annotated
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from keras.models import load_model
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import time
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.language_models import BaseChatModel, LLM
from langchain_core.outputs import Generation, ChatResult, ChatGeneration
from langchain_core.runnables import Runnable
from langchain_core.tools import Tool, tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import (
    AgentState,
    add_messages,
    IsLastStep,
    RemainingSteps,
)

from langchain_google_vertexai import ChatVertexAI

from vertexai import init as vertexai_init




def Agentic_AI_F():
  os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/content/GEMINI_JSON_ACTIVE.json"
  credentials = service_account.Credentials.from_service_account_file(
      "/content/GEMINI_JSON_ACTIVE.json"
  )

  def llm_obj():
      llm = ChatVertexAI(
        model="gemini-2.5-pro",
        project="steady-bonsai-467007-g0",
        location="us-central1",
        max_output_tokens=2048,
        temperature=0,
        credentials=credentials
    )
      return llm

  from typing import Optional, Dict, Any
  from pydantic import BaseModel

  class Agentic_AI(BaseModel):
      Corecast: Optional[Dict[str, Any]] = None
      FestivCast: Optional[Dict[str, Any]] = None
      SesoCast: Optional[Dict[str, Any]] = None

  ###########################################################################################################################
  #####################################...............CORECAST..............#################################################
  ###########################################################################################################################
  def Corecast(state:Agentic_AI ):
    @tool
    def preprocess(input: str) -> str:
        """
        Preprocess the call volume dataset for modeling.

        Returns:
            pd.DataFrame: Preprocessed and feature-engineered DataFrame ready for LSTM modeling,
                          with relevant columns encoded and 'REPORT_DT' set as the index.
        """
        global holidays_df,model_data,model_data_encoded
        holidays_df = pd.read_csv(r"/content/Holiday.csv")                   ###PATH_CHANGE
        model_data = pd.read_excel(r"/content/Call_Volume_Data_2020_to_2025.xlsx")  ###PATH_CHANGE
        holidays_df['Date'] = pd.to_datetime(holidays_df['Date'],dayfirst=True)
        model_data = pd.merge(model_data, holidays_df[['Date', 'Holiday']],
                              left_on='REPORT_DT', right_on='Date', how='left')
        model_data.drop(columns=['Date'], inplace=True)
        model_data['Holiday'] = model_data['Holiday'].fillna('No Holiday')
        model_data.sort_values(by='REPORT_DT').reset_index()
        model_data.set_index('REPORT_DT', inplace=True)
        # model_data = model_data.drop(columns=['index'], axis = 1)
        # Ensure that columns are numeric
        model_data['DAY_OF_WEEK'] = model_data.index.dayofweek
        model_data['DAY_OF_MONTH'] = pd.to_numeric(model_data['DAY_OF_MONTH'], errors='coerce')
        model_data['MONTH'] = pd.to_numeric(model_data['MONTH'], errors='coerce')
        model_data['QUARTER'] = pd.to_numeric(model_data['QUARTER'], errors='coerce')
        # Encode DAY_OF_WEEK (1 to 7)
        model_data['DAY_OF_WEEK_SIN'] = np.sin(2 * np.pi * model_data['DAY_OF_WEEK'] / 7)
        model_data['DAY_OF_WEEK_COS'] = np.cos(2 * np.pi * model_data['DAY_OF_WEEK'] / 7)
        # Encode DAY_OF_MONTH (1 to 31)
        model_data['DAY_OF_MONTH_SIN'] = np.sin(2 * np.pi * model_data['DAY_OF_MONTH'] / 31)
        model_data['DAY_OF_MONTH_COS'] = np.cos(2 * np.pi * model_data['DAY_OF_MONTH'] / 31)
        # Encode MONTH (1 to 12)
        model_data['MONTH_SIN'] = np.sin(2 * np.pi * model_data['MONTH'] / 12)
        model_data['MONTH_COS'] = np.cos(2 * np.pi * model_data['MONTH'] / 12)
        # Encode QUARTER (1 to 4)
        model_data['QUARTER_SIN'] = np.sin(2 * np.pi * model_data['QUARTER'] / 4)
        model_data['QUARTER_COS'] = np.cos(2 * np.pi * model_data['QUARTER'] / 4)
        # Encode extra
        model_data['is_sunday'] = (model_data.index.dayofweek == 6).astype(int)
        model_data['is_monday'] = (model_data.index.dayofweek == 0).astype(int)
        model_data['is_weekend'] = (model_data.index.dayofweek >= 5).astype(int)
        encoding = model_data.groupby('Holiday')['TOTAL_OFFERED_CALL_VOLUME'].mean().to_dict()
        model_data['HOLIDAY_ENCODED'] = model_data['Holiday'].map(encoding)
        default_value = model_data['TOTAL_OFFERED_CALL_VOLUME'].mean()
        model_data['HOLIDAY_ENCODED'] = model_data['HOLIDAY_ENCODED'].fillna(default_value)
        model_data_encoded = model_data.drop(columns=['DAY_OF_WEEK','DAY_OF_MONTH',"YEAR","QUARTER","Holiday","MONTH"])
        return model_data_encoded

    @tool
    def LSTM_Prediction(input: str) -> str:
        """
        Predicts future call volumes using a pre-trained LSTM model.

        This function takes the preprocessed and encoded data, creates input sequences,
        loads a trained LSTM model, and returns a 180-day forecast.

        Returns:
            pd.DataFrame: A DataFrame containing 180 days of future dates and corresponding forecasted
                          'TOTAL_OFFERED_CALL_VOLUME' values.
        """
        global forecast_df,model_data_encoded
        # Define input features and target
        feature_cols = ['DAY_OF_WEEK_SIN', 'DAY_OF_WEEK_COS',
                        'DAY_OF_MONTH_SIN', 'DAY_OF_MONTH_COS', 'MONTH_SIN', 'MONTH_COS',
                        'QUARTER_SIN', 'QUARTER_COS', 'HOLIDAY_ENCODED']
        target_col = 'TOTAL_OFFERED_CALL_VOLUME'
        model_data_encoded_scaled = model_data_encoded.copy()
        scaler = MinMaxScaler()
        model_data_encoded_scaled.loc[:, ['TOTAL_OFFERED_CALL_VOLUME', 'HOLIDAY_ENCODED']] = scaler.fit_transform(
            model_data_encoded_scaled[['TOTAL_OFFERED_CALL_VOLUME', 'HOLIDAY_ENCODED']])

        def create_sequences(data, seq_length, horizon):
            X= []
            for i in range(len(data) - seq_length - horizon + 1):
                X.append(data[i:i + seq_length])
            return np.array(X)

        X_train= create_sequences(model_data_encoded_scaled, 360, 180)
        model = tf.keras.models.load_model(r"/content/Agentic_AI_LSTM_v1.h5") ###PATH_CHANGE
        predictions = model.predict(X_train)
        predictions_rescaled = scaler.data_min_[0] + predictions * (scaler.data_max_[0] - scaler.data_min_[0])
        predictions_rescaled = predictions_rescaled[0]
        start_date = model_data_encoded.index[-1] + pd.Timedelta(days=1)
        date_range = pd.date_range(start=start_date, periods=180, freq='D')
        forecast_df = pd.DataFrame({'Date':date_range, 'Forecast':predictions_rescaled})
        return forecast_df

    @tool
    def Last_week_Mape(input: str) -> str:
        """
        Calculates the Mean Absolute Percentage Error (MAPE) between predicted and actual call volumes
        for the last 7 days (after forecasting 180 days using data excluding the final 14 days).

        Returns:
            pd.DataFrame: DataFrame containing date, forecast, actual call volume, and calculated MAPE
                          for the overlapping 7-day period.
        """
        global holidays_df,model_data,forecast_last_week_mape
        print("HERE")
        holidays_df = pd.read_csv(r"/content/Holiday.csv") ###PATH_CHANGE
        model_data = pd.read_excel(r"/content/Call_Volume_Data_2020_to_2025.xlsx")  ###PATH_CHANGE
        model_data = model_data.iloc[:-7]
        holidays_df['Date'] = pd.to_datetime(holidays_df['Date'],dayfirst=True)
        model_data = pd.merge(model_data, holidays_df[['Date', 'Holiday']],
                              left_on='REPORT_DT', right_on='Date', how='left')
        model_data.drop(columns=['Date'], inplace=True)
        model_data['Holiday'] = model_data['Holiday'].fillna('No Holiday')
        model_data.sort_values(by='REPORT_DT').reset_index()
        model_data.set_index('REPORT_DT', inplace=True)
        # model_data = model_data.drop(columns=['index'], axis = 1)
        # Ensure that columns are numeric
        model_data['DAY_OF_WEEK'] = model_data.index.dayofweek
        model_data['DAY_OF_MONTH'] = pd.to_numeric(model_data['DAY_OF_MONTH'], errors='coerce')
        model_data['MONTH'] = pd.to_numeric(model_data['MONTH'], errors='coerce')
        model_data['QUARTER'] = pd.to_numeric(model_data['QUARTER'], errors='coerce')
        # Encode DAY_OF_WEEK (1 to 7)
        model_data['DAY_OF_WEEK_SIN'] = np.sin(2 * np.pi * model_data['DAY_OF_WEEK'] / 7)
        model_data['DAY_OF_WEEK_COS'] = np.cos(2 * np.pi * model_data['DAY_OF_WEEK'] / 7)
        # Encode DAY_OF_MONTH (1 to 31)
        model_data['DAY_OF_MONTH_SIN'] = np.sin(2 * np.pi * model_data['DAY_OF_MONTH'] / 31)
        model_data['DAY_OF_MONTH_COS'] = np.cos(2 * np.pi * model_data['DAY_OF_MONTH'] / 31)
        # Encode MONTH (1 to 12)
        model_data['MONTH_SIN'] = np.sin(2 * np.pi * model_data['MONTH'] / 12)
        model_data['MONTH_COS'] = np.cos(2 * np.pi * model_data['MONTH'] / 12)
        # Encode QUARTER (1 to 4)
        model_data['QUARTER_SIN'] = np.sin(2 * np.pi * model_data['QUARTER'] / 4)
        model_data['QUARTER_COS'] = np.cos(2 * np.pi * model_data['QUARTER'] / 4)
        # Encode extra
        model_data['is_sunday'] = (model_data.index.dayofweek == 6).astype(int)
        model_data['is_monday'] = (model_data.index.dayofweek == 0).astype(int)
        model_data['is_weekend'] = (model_data.index.dayofweek >= 5).astype(int)
        encoding = model_data.groupby('Holiday')['TOTAL_OFFERED_CALL_VOLUME'].mean().to_dict()
        model_data['HOLIDAY_ENCODED'] = model_data['Holiday'].map(encoding)
        default_value = model_data['TOTAL_OFFERED_CALL_VOLUME'].mean()
        model_data['HOLIDAY_ENCODED'] = model_data['HOLIDAY_ENCODED'].fillna(default_value)
        model_data_encoded = model_data.drop(columns=['DAY_OF_WEEK','DAY_OF_MONTH',"YEAR","QUARTER","Holiday","MONTH"])
            # Define input features and target
        feature_cols = ['DAY_OF_WEEK_SIN', 'DAY_OF_WEEK_COS',
                        'DAY_OF_MONTH_SIN', 'DAY_OF_MONTH_COS', 'MONTH_SIN', 'MONTH_COS',
                        'QUARTER_SIN', 'QUARTER_COS', 'HOLIDAY_ENCODED']
        target_col = 'TOTAL_OFFERED_CALL_VOLUME'
        model_data_encoded_scaled = model_data_encoded.copy()
        scaler = MinMaxScaler()
        model_data_encoded_scaled.loc[:, ['TOTAL_OFFERED_CALL_VOLUME', 'HOLIDAY_ENCODED']] = scaler.fit_transform(
            model_data_encoded_scaled[['TOTAL_OFFERED_CALL_VOLUME', 'HOLIDAY_ENCODED']])

        def create_sequences(data, seq_length, horizon):
            X= []
            for i in range(len(data) - seq_length - horizon + 1):
                X.append(data[i:i + seq_length])
            return np.array(X)
        X_train= create_sequences(model_data_encoded_scaled, 360, 180)
        model = tf.keras.models.load_model(r"/content/Agentic_AI_LSTM_v1.h5") ###PATH_CHANGE
        predictions = model.predict(X_train)
        predictions_rescaled = scaler.data_min_[0] + predictions * (scaler.data_max_[0] - scaler.data_min_[0])
        predictions_rescaled = predictions_rescaled[0]
        start_date = model_data_encoded.index[-1] + pd.Timedelta(days=1)
        date_range = pd.date_range(start=start_date, periods=180, freq='D')
        forecast_df = pd.DataFrame({'Date':date_range, 'Forecast':predictions_rescaled})
        model_data = pd.read_excel(r"/content/Call_Volume_Data_2020_to_2025.xlsx")  ###PATH_CHANGE
        model_data = model_data.reset_index()
        model_data.rename(columns={'REPORT_DT':'Date'},inplace=True)
        forecast_last_week = forecast_df.merge(model_data[['Date','TOTAL_OFFERED_CALL_VOLUME']],on=['Date'],how='inner')
        forecast_last_week['MAPE'] = (abs(forecast_last_week['TOTAL_OFFERED_CALL_VOLUME'] - forecast_last_week['Forecast'])
                                      /forecast_last_week['TOTAL_OFFERED_CALL_VOLUME'])*100
        forecast_last_week_mape = forecast_last_week['MAPE'].mean()

        return forecast_last_week_mape

    tools = [preprocess, LSTM_Prediction,Last_week_Mape]
    llm = llm_obj()
    bound_llm = llm.bind_tools(tools)

    agent_node = create_react_agent(bound_llm, tools=tools, state_schema=AgentState)
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")



    runnable = graph.compile()
    prompt = "Preprocesses the dataframe with  'model_data' "
    output = runnable.invoke({"messages": [{"role": "user", "content": prompt}]})
    
    time.sleep(5)

    prompt2 = "Predicts future call volumes using a pre-trained LSTM model with 'model_data' "
    output2 = runnable.invoke({"messages": [{"role": "user", "content": prompt2}]})

    time.sleep(5)

    prompt3 = "Calculate the Mean Absolute Percentage Error (MAPE) between predicted and actual call volumes for the last 7 days with 'model_data' "
    output3 = runnable.invoke({"messages": [{"role": "user", "content": prompt3}]})

    st.title("🤖 AGENT CoreCast")
    # placeholder = st.empty()
    print("\nFinal output from agent:\n")
    # with placeholder.container():
    full_output = ""

    # Loop over all output blocks
    for output_block in [output, output2, output3]:
        for msg in output_block["messages"]:
            name = getattr(msg, "name", None)
            if name is None:
                full_output += f"📌 Agent says: {msg.content}\n"
            else:
                full_output += f"🔧 Using **{name}** tool to perform the task\n"
    # st.write(full_output) 
    llm = llm_obj()

    prompt = f"""
    Restructure the following output with two sections:

    Thought:
    Describe everything the agent did, step-by-step, as if it's explaining what it was trying to accomplish and how.

    Reason:
    List and explain all the tools the agent used and why each was used.

    Output to convert:
    {full_output}
    """

    # For storing the full streamed output
    collected_chunks = []

    # Define a generator that yields chunks and stores them
    def stream_and_store():
        for chunk in llm.stream(prompt):
            collected_chunks.append(chunk)
            yield chunk  # Stream to Streamlit in real-time

    # Stream the response live in Streamlit
    st.write_stream(stream_and_store)


    time.sleep(15)  
    # placeholder.empty()
    st.markdown(f"##### Forecasted DataFrame:")
    st.dataframe(forecast_df)
    return None

  ###########################################################################################################################
  #####################################...............FestivCast..............###############################################
  ###########################################################################################################################
  def FestivCast(state:Agentic_AI):
    @tool
    def identify_holidays(input: str) -> str:

      """
      Identify holiday dates in the forecast dataset and compare forecasted call volumes with the average actuals
      for the same weekday over the past 3 weeks to estimate the forecasted holiday impact.

      Returns:
          pd.DataFrame: A DataFrame showing:
              - Holiday date
              - Holiday name
              - Day of the week
              - 3-week average actuals for the same weekday
              - Forecasted volume
              - Percentage difference indicating the forecasted impact
      """
      global forecast_df, model_data, holidays_df,forecast_holiday_impact
      df_actual = model_data.copy()
      df_holidays = holidays_df.copy()
      df_forecast = forecast_df.copy()
      df_actual.rename(columns={'REPORT_DT':'Date','TOTAL_OFFERED_CALL_VOLUME':'Call Volume'},inplace=True)
      df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
      df_actual['Date'] = pd.to_datetime(df_actual['Date'])
      df_holidays['Date'] = pd.to_datetime(df_holidays['Date'],dayfirst=True)
      df_forecast['day_of_week'] = pd.to_datetime(df_forecast['Date']).dt.day_name()
      df_actual['day_of_week'] = pd.to_datetime(df_actual['Date']).dt.day_name()
      holiday_forecast = df_forecast.merge(df_holidays, left_on = "Date", right_on = "Date", how = "inner")
      holiday_forecast = holiday_forecast[["Date","Holiday","day_of_week","Forecast"]]
      holiday_forecast.rename(columns={"day_of_week":"Day of Week"}, inplace = True)
      dow_averages = []
      for date, dow in zip(holiday_forecast['Date'], holiday_forecast['Day of Week']):
          start_date = date - pd.Timedelta(weeks=3)
          filtered = df_actual[(df_actual['Date'] < date) & (df_actual['Date'] >= start_date)]
          avg = filtered[filtered['day_of_week'] == dow]['Call Volume'].mean()
          dow_averages.append(avg)
      holiday_forecast['dow_avg_3w'] = dow_averages
      holiday_forecast['Impact_Captured'] = round(
          (holiday_forecast['Forecast'] - holiday_forecast['dow_avg_3w']) /
          holiday_forecast['dow_avg_3w'] * 100, 2
      )
      forecast_holiday_impact = holiday_forecast.copy()
      return forecast_holiday_impact

    @tool
    def holiday_impact(input: str) -> str:

        """
        Evaluate the *actual* impact of holidays on call volume by comparing actuals to a 3-week weekday average.

        Returns:
            pd.DataFrame: A summarized DataFrame grouped by holiday, showing:
                - Holiday name
                - Mean actual call volume on the holiday
                - Mean percentage difference compared to normal same-day-of-week volumes (Actual Impact)
        """
        global model_data, holidays_df,historic_holiday_impact
        df_actuals = model_data.copy()
        df_holidays = holidays_df.copy()
        df_actuals.rename(columns={'REPORT_DT':'Date','TOTAL_OFFERED_CALL_VOLUME':'Call Volume'},inplace=True)
        df_forecast = forecast_df.copy()
        df_actuals['Date'] = pd.to_datetime(df_actuals['Date'])
        df_holidays['Date'] = pd.to_datetime(df_holidays['Date'],dayfirst=True)
        merged_df = pd.merge(df_actuals, df_holidays[['Date', "Holiday"]], left_on='Date', right_on='Date', how='left')
        # Fill NaN values in the 'Holiday' column with 'Normal Day'
        merged_df['Holiday'] = merged_df['Holiday'].fillna('Normal Day')
        dow_averages = []
        for date, dow in zip(merged_df['Date'], merged_df['DAY_OF_WEEK']):
            start_date = date - pd.Timedelta(weeks=3)
            filtered = merged_df[(merged_df['Date'] < date) & (merged_df['Date'] >= start_date)]
            avg = filtered[filtered['DAY_OF_WEEK'] == dow]['Call Volume'].mean()
            dow_averages.append(avg)
        merged_df['dow_avg_3w'] = dow_averages
        merged_df['Actual_Impact'] = round((merged_df['Call Volume']-merged_df['dow_avg_3w'])/merged_df['dow_avg_3w']*100,2)
        historic_holiday_impact = merged_df.groupby("Holiday",as_index = False).agg({"Actual_Impact":"mean","Call Volume":"mean"}).reset_index(drop=True)
        return historic_holiday_impact

    @tool
    def final_adjustment(input: str) -> str:

        """
        Performs a final adjustment on the holiday-forecasted call volumes by incorporating
        the difference between actual historical impact and the impact already captured.
        """
        global adjust_df,forecast_holiday_impact,historic_holiday_impact
        adjust_df = forecast_holiday_impact.merge(historic_holiday_impact, on="Holiday", how = "left")
        adjust_df['Impact_Not_Captured'] = adjust_df['Actual_Impact'] - adjust_df['Impact_Captured']
        adjust_df['FestiVcast_Adjusted_Forecast'] = np.where(adjust_df['Impact_Not_Captured']>0, adjust_df['Forecast']*(1+(adjust_df['Impact_Not_Captured'])/100), adjust_df['Forecast'])
        st.markdown(f"##### FestivCast Adjusted DataFrame:")
        st.dataframe(adjust_df)
        return adjust_df


    tools = [final_adjustment, holiday_impact,identify_holidays]
    llm = llm_obj()
    bound_llm = llm.bind_tools(tools)

    agent_node = create_react_agent(bound_llm, tools=tools, state_schema=AgentState )
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")

    runnable = graph.compile()
    prompt = "identify the holidays present in 'forecast_df' and thenEvaluate the actual impact of holidays on call volume and finally  Performs a final adjustment on the holiday-forecasted call volumes"
    output = runnable.invoke({"messages": [{"role": "user", "content": prompt}]})


    st.title("🤖 AGENT FestivCast")
    full_output = ""

    # Loop over all output blocks
    for output_block in [output]:
        for msg in output_block["messages"]:
            name = getattr(msg, "name", None)
            if name is None:
                full_output += f"📌 Agent says: {msg.content}\n"
            else:
                full_output += f"🔧 Using **{name}** tool to perform the task\n"
    # st.write(full_output) 
    llm = llm_obj()

    prompt = f"""
    Restructure the following output with two sections:

    Thought:
    Describe everything the agent did, step-by-step, as if it's explaining what it was trying to accomplish and how.

    Reason:
    List and explain all the tools the agent used and why each was used.

    Output to convert:
    {full_output}
    """

    # For storing the full streamed output
    collected_chunks = []

    # Define a generator that yields chunks and stores them
    def stream_and_store():
        for chunk in llm.stream(prompt):
            collected_chunks.append(chunk)
            yield chunk  # Stream to Streamlit in real-time

    # Stream the response live in Streamlit
    st.write_stream(stream_and_store)


    time.sleep(15)  
    # placeholder.empty()
    st.markdown(f"##### Forecasted DataFrame:")
    st.dataframe(adjust_df) 
    # placeholder.empty()

    return None

  ###########################################################################################################################
  #####################################...............SesoCast..............###############################################
  ###########################################################################################################################
  def Sesocast(state:Agentic_AI):
    @tool
    def get_historical_periods_dynamic_actual(input: str) -> str:
        """
        Extracts the same date range in previous years from a DataFrame, dynamically determining available years.
        Returns:
            dict: A dictionary where keys are years and values are DataFrames
                  containing the corresponding period. Returns an empty dictionary if there are issues.
        """
        global forecast_df, model_data,final_last_28_days
        df = model_data.iloc[-28:].copy()
        df_history =  model_data.copy()
        date_column='Date'
        start_date=None
        end_date=None

        df_history.rename(columns={'REPORT DT':'Date', 'TOTAL_OFFERED_CALL_VOLUME':'Call Volume'},inplace=True)
        df[date_column] = pd.to_datetime(df[date_column])
        # Determine start and end dates if not provided
        if start_date is None or end_date is None:
            end_date = df[date_column].max()
            start_date = df[date_column].min()
        else:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
        # Calculate the date range
        date_range = end_date - start_date
        if date_range.days < 0:
            print("Error: End date is earlier than start date.")
        historical_periods = {}
        current_year = end_date.year # Use the end date year as the "current" year
        available_years = sorted(df_history[date_column].dt.year.unique(), reverse=True)
        # Iterate through available years *excluding* the current year
        for year in available_years:
            if year == current_year:
                continue #skip current year
            # Calculate the start and end dates for the current year
            year_start_date = pd.to_datetime(f'{year}-{start_date.month}-{start_date.day}')
            year_end_date = pd.to_datetime(f'{year}-{end_date.month}-{end_date.day}')
            # Filter the Dataframe for the current year period
            year_df = df_history[(df_history[date_column] >= year_start_date) & (df_history[date_column] <= year_end_date)].copy()
            if not year_df.empty:
                historical_periods[year] = year_df
            else:
                print(f'No data found for {year}.')
        historical_periods_yearly = pd.DataFrame()
        last_28days_dict = historical_periods.copy()
        counter = 0
        final_last_28_days = pd.DataFrame()
        for i in last_28days_dict.keys():
            if counter<3:
                globals()[f"last_28days_{i}"] = last_28days_dict[i]
            else:
                break
            final_last_28_days = pd.concat([final_last_28_days,globals()[f"last_28days_{i}"]], ignore_index = True)
            final_last_28_days['Year'] = final_last_28_days['Date'].dt.year
        return final_last_28_days


    @tool
    def get_historical_periods_dynamic_forecast(input: str) -> str:

        """
        Forecast is Extracted for the same date range from forecast dataframe, dynamically determining available years.
        Returns:
            dict: A dictionary where keys are years and values are DataFrames
                  containing the corresponding period. Returns an empty dictionary if there are issues.
        """
        global forecast_df, model_data,final_next_28_days
        df = forecast_df.copy()
        df_history =  model_data.copy()
        date_column='Date'
        start_date=None
        end_date=None

        df_history.rename(columns={'REPORT_DT': 'Date', 'TOTAL_OFFERED_CALL_VOLUME':'Call Volume'},inplace=True)
        df[date_column] = pd.to_datetime(df[date_column])
        # Determine start and end dates if not provided
        if start_date is None or end_date is None:
            end_date = df[date_column].max()
            start_date = df[date_column].min()
        else:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
        # Calculate the date range
        date_range = end_date - start_date
        if date_range.days < 0:
            print("Error: End date is earlier than start date.")
        historical_periods = {}
        current_year = end_date.year # Use the end_date year as the "current" year
        available_years = sorted(df_history[date_column].dt.year.unique(), reverse=True)
        # Iterate through available years excluding the "current" year
        for year in available_years:
            if year == current_year:
                continue #skip current year
            year_start_date = pd.to_datetime(f'{year}-{start_date.month}-{start_date.day}')
            year_end_date = pd.to_datetime(f'{year}-{end_date.month}-{end_date.day}')
            year_df = df_history[(df_history[date_column] >= year_start_date) & (df_history[date_column] <= year_end_date)].copy()
            if not year_df.empty:
                historical_periods[year] = year_df
        else:
            historical_periods[year] = pd.DataFrame()
        next_28days_dict = historical_periods.copy()
        counter = 0
        # Initialize an empty DataFrame
        final_next_28_days = pd.DataFrame()
        for i in next_28days_dict.keys():
            if counter<3:
                globals()[f"next_28days_{i}"] = next_28days_dict[i]
                counter+=1
            else:
                break
            final_next_28_days = pd.concat([final_next_28_days,globals()[f"next_28days_{i}"]], ignore_index = True)
            final_next_28_days['Year'] = final_next_28_days['Date'].dt.year
        return final_next_28_days

    @tool
    def Final_adjustment_Seso(input: str) -> str:
      """
        Applies a last-mile seasonal adjustment to the forecasted call volumes based on
        day-of-week (DOW) trends observed in the last 28 days vs the upcoming 28 days.
      """
      global forecast_df, model_data,final_next_28_days,forecast_merged,final_last_28_days
      final_next_28_dow_avg = final_next_28_days.groupby(["Year","DAY_OF_WEEK"], as_index = False).agg({"Call Volume":"mean"})
      final_next_28_dow_avg = final_next_28_dow_avg.rename(columns = {'Call Volume':'actual_offered_next'})
      final_last_28_dow_avg = final_last_28_days.groupby(["Year","DAY_OF_WEEK"], as_index = False).agg({"Call Volume":"mean"})
      final_last_28_dow_avg = final_last_28_dow_avg.rename(columns = {'Call Volume':'actual_offered_last'})
      final_seasonal_comparison_df = final_next_28_dow_avg.merge(final_last_28_dow_avg, on = ["Year","DAY_OF_WEEK"], how = "inner")
      final_seasonal_comparison_df["Seasonal_Factor"] = (final_seasonal_comparison_df["actual_offered_next"]/ final_seasonal_comparison_df["actual_offered_last"])
      final_seasonality_dim = final_seasonal_comparison_df.groupby(["DAY_OF_WEEK"], as_index = False).agg({"Seasonal_Factor":"mean"})

      forecast_df['DAY_OF_WEEK'] = pd.to_datetime(forecast_df['Date']).dt.day_name()

      forecast_merged = forecast_df.merge(final_seasonality_dim,on=['DAY_OF_WEEK'],how='inner')
      forecast_merged['LM_Seasonal_Adjustment'] = forecast_merged['Forecast'] * forecast_merged['Seasonal_Factor']
      forecast_merged = forecast_merged[['Date','DAY_OF_WEEK','Forecast','Seasonal_Factor','LM_Seasonal_Adjustment']]
      return forecast_merged


    tools = [Final_adjustment_Seso, get_historical_periods_dynamic_forecast,get_historical_periods_dynamic_actual]
    llm = llm_obj()
    bound_llm = llm.bind_tools(tools)

    agent_node = create_react_agent(bound_llm, tools=tools, state_schema=AgentState)
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")

    runnable = graph.compile()
    prompt = "extract the same date range in previous years from 'model_data' and also  Extract for the same date range from forecast dataframe 'forecast_df'"
    output = runnable.invoke({"messages": [{"role": "user", "content": prompt}]})


    prompt1 = "finally Apply a last-mile seasonal adjustment to the forecasted call volumes 'model_data'"
    output1 = runnable.invoke({"messages": [{"role": "user", "content": prompt1}]})

    st.title("🤖 AGENT SesoCast")
    # placeholder = st.empty()
    print("\nFinal output from agent:\n")
    full_output = ""

    # Loop over all output blocks
    for output_block in [output,output1]:
        for msg in output_block["messages"]:
            name = getattr(msg, "name", None)
            if name is None:
                full_output += f"📌 Agent says: {msg.content}\n"
            else:
                full_output += f"🔧 Using **{name}** tool to perform the task\n"
    # st.write(full_output) 
    llm = llm_obj()

    prompt = f"""
    Restructure the following output with two sections:

    Thought:
    Describe everything the agent did, step-by-step, as if it's explaining what it was trying to accomplish and how.

    Reason:
    List and explain all the tools the agent used and why each was used.

    Output to convert:
    {full_output}
    """

    # For storing the full streamed output
    collected_chunks = []

    # Define a generator that yields chunks and stores them
    def stream_and_store():
        for chunk in llm.stream(prompt):
            collected_chunks.append(chunk)
            yield chunk  # Stream to Streamlit in real-time

    # Stream the response live in Streamlit
    st.write_stream(stream_and_store)


    time.sleep(15)  
    # placeholder.empty()
    st.markdown(f"##### Forecasted DataFrame:")
    st.dataframe(forecast_merged) 
    return None

  ################################################################################################################################################
  graph = StateGraph(Agentic_AI)
  graph.add_node("Corecast", Corecast)
  graph.add_node("Festivcast", FestivCast)
  graph.add_node("SesoCast", Sesocast)
  graph.set_entry_point("Corecast")
  graph.add_edge("Corecast","Festivcast")
  graph.add_edge("Festivcast","SesoCast")

  app = graph.compile()

  runnable = app.invoke({})
  return None

import streamlit as st


########
# Start from current week's Monday
today = datetime.today()
start_of_week = today - timedelta(days=today.weekday())  # Monday

# Generate data for 5 weeks
weeks = [start_of_week - timedelta(weeks=i) for i in range(4, -1, -1)]  # Oldest to newest

df_1 = pd.DataFrame({
    "Week Start Date": [week.date() for week in weeks],
    "Volume": np.random.randint(1000, 3001, size=5),
    "AHT": np.random.randint(150, 451, size=5),
    "NORM (%)": np.round(np.random.uniform(90, 110, size=5), 2)
})



# Generate daily dates for the past 6 months
end_date = datetime.today()
start_date = end_date - timedelta(days=180)  # ~6 months
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Create DataFrame
df = pd.DataFrame({
    "Date": dates,
    "CT": "CostcoBrand1",
    "Volume": np.random.randint(1000, 2001, size=len(dates)),
    "Forecast": np.random.randint(1000, 2001, size=len(dates))
})

# Melt the DataFrame for Plotly
df_melted = df.melt(id_vars=["Date", "CT"], value_vars=["Volume", "Forecast"],
                    var_name="Type", value_name="Value")
########



st.set_page_config(page_title="WFM - Nexus")
st.title("Nexus - Next Gen Workforce Planning Solution Powered By Agentic AI📈🔍")
MIN_MAX_SKU = ["CostcoBrand1","CostcoBrand2","CostcoBrand3"]
selected_option_1 = st.selectbox("Select type of CT to forecast:", MIN_MAX_SKU)
st.write(f"You selected: {selected_option_1}")

##########
avg_volume = 1000
Norm = "102%"
MAPE = 12
AHT = 250

# Title
st.write("")
st.markdown(f"##### 📊 Previous Forecast Summary of {selected_option_1}")

with st.container():
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="📈 Avg Volume", value=avg_volume)

    with col2:
        st.metric(label="⚠️ NORM %", value=Norm)

    with col3:
        st.metric(label="📉 MAPE ", value=MAPE)
    with col4:
        st.metric(label="⚠️ AHT ", value=AHT)
st.markdown("##### Week over week Summary")
st.dataframe(df_1)

# Plotly Chart
fig = px.line(
    df_melted,
    x="Date",
    y="Value",
    color="Type",
    title="📈 Daily Volume vs Forecast - CostcoBrand1",
    labels={"Value": "Call Volume", "Date": "Date"},
    markers=False
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Volume / Forecast",
    template="plotly_white",
    xaxis=dict(rangeslider=dict(visible=True), type="date")
)

# Show Plotly chart in Streamlit
st.plotly_chart(fig, use_container_width=True)
#########

if st.button(f"Forecast for {selected_option_1}"):
  Agentic_AI_F()

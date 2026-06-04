import streamlit as st
import pandas as pd
import plotly.express as px

from ml_utils import train_classifier

st.set_page_config(
    page_title="ML CSV Analyzer",
    layout="wide"
)

st.title("Machine Learning CSV Analyzer")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.header("Dataset Preview")

    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    col1.metric(
        "Rows",
        df.shape[0]
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    st.header("Missing Values")

    st.dataframe(
        df.isnull().sum()
    )

    st.header("Statistics")

    st.dataframe(
        df.describe(include="all")
    )

    st.header("Target Selection")

    target = st.selectbox(
        "Choose Target Column",
        df.columns
    )

    if st.button("Train Model"):

        model, accuracy = train_classifier(
            df,
            target
        )

        st.success(
            f"Model Accuracy: {accuracy:.2%}"
        )

        st.balloons()

        numeric_cols = df.select_dtypes(
            include="number"
        ).columns

        if len(numeric_cols) > 0:

            selected_col = st.selectbox(
                "Select Column",
                numeric_cols
            )

            fig = px.histogram(
                df,
                x=selected_col
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

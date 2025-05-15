import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from fpdf import FPDF
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import chi2_contingency

# Set page configuration
st.set_page_config(layout="wide", page_title="Election Analysis Dashboard")
pio.templates.default = "plotly_white"

# Load data function
@st.cache_data
def load_data():
    results = pd.read_csv("results.csv")
    advertisers = pd.read_csv("advertisers.csv")
    locations = pd.read_csv("locations.csv")

    # Strip spaces and standardize column names
    results["State"] = results["State"].str.strip().str.lower()
    locations["Location name"] = locations["Location name"].str.strip().str.lower()

    # Merge the data
    merged_data = results.merge(locations, left_on="State", right_on="Location name", how="left")

    advertisers["Amount spent (INR)"] = pd.to_numeric(advertisers["Amount spent (INR)"], errors="coerce")
    advertisers.dropna(subset=["Amount spent (INR)"], inplace=True)

    return results, advertisers, locations, merged_data

# Load data
results, advertisers, locations, merged_data = load_data()

# Sidebar Party Selection with an "All" option
party_options = ["All"] + list(advertisers["Page name"].unique())  # Add "All" option
selected_party = st.sidebar.selectbox("Select a Political Party", party_options)

# Sidebar Search Bar for State
search_state = st.sidebar.text_input("Search for a State (e.g., Maharashtra, Uttar Pradesh)", "")

# Filter the merged data based on the selected state
if search_state:
    filtered_merged_data = merged_data[merged_data["State"].str.contains(search_state.strip().lower(), case=False, na=False)]
    if filtered_merged_data.empty:
        st.warning(f"No data found for '{search_state}'. Please try again with a different state.")
else:
    filtered_merged_data = merged_data  # Show all data if no search term is provided

# Sidebar Election Overview
if selected_party == "All":
    filtered_advertisers = advertisers  # Show data for all parties
    total_spent = filtered_advertisers["Amount spent (INR)"].sum()
    total_parties = filtered_advertisers["Page name"].nunique()
    top_3_parties = filtered_advertisers.groupby("Page name")["Amount spent (INR)"].sum().nlargest(3).reset_index()
else:
    filtered_advertisers = advertisers[advertisers["Page name"] == selected_party]
    total_spent = filtered_advertisers["Amount spent (INR)"].sum()
    total_parties = filtered_advertisers["Page name"].nunique()
    top_3_parties = advertisers.groupby("Page name")["Amount spent (INR)"].sum().nlargest(3).reset_index()

# Sidebar settings
st.sidebar.title("Election Analysis Dashboard")
st.sidebar.markdown("📊 Explore election data through various insights.")

st.header(f"🗳️ Election Overview for {selected_party}")
col1, col2, col3 = st.columns(3)
col1.metric("Total Amount Spent (INR)", f"INR {total_spent:,.0f}")  # Ensure full value display
col2.metric("Total Number of Parties", total_parties)
col3.markdown("**Top 3 Spending Parties**")
if selected_party == "All":
    # Show top 3 overall when "All" is selected
    for index, row in top_3_parties.iterrows():
        col3.write(f"{index+1}. {row['Page name']}: INR {row['Amount spent (INR)']:,.0f}")
else:
    # Show top 3 for the selected party's ad spend data
    col3.write(f"1. {selected_party}: INR {total_spent:,.0f}")

st.divider()

# PDF generation function
def generate_pdf(title, content, file_name="report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=title, ln=True, align="C")
    pdf.ln(10)
    for line in content:
        pdf.multi_cell(0, 10, txt=line)
    pdf.output(file_name)
    return file_name

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["Ad Spend by State", "Voter Turnout", "Party Ad Spend", "Phase-wise Analysis", "Clustering Ad Spend and Voter Turnout", "Regression Analysis", "Chi-Square Test", "SVM Analysis", "Random Forest Analysis"])

# Tab 1: Ad Spend by State
with tab1:
    st.header("Total Ad Spend by State")
    st.write("This tab shows the total advertisement spend (in INR) for each state. It gives an overview of how political parties are allocating their ad spend across different regions.")
    
    state_ad_spend = filtered_merged_data.groupby("State")["Amount spent (INR)"].sum().reset_index()
    fig = px.bar(state_ad_spend, x="State", y="Amount spent (INR)", title="Total Ad Spend by State")
    fig.update_layout(xaxis_tickangle=-90, width=900, height=600)
    st.plotly_chart(fig)

    # PDF Report
    content = [
        "Total Advertisement Spend by State:",
        state_ad_spend.to_string(index=False)
    ]
    pdf_filename = generate_pdf("Ad Spend by State Report", content, file_name="ad_spend_by_state_report.pdf")
    st.download_button(
        label="Download Ad Spend Report",
        data=open(pdf_filename, "rb").read(),
        file_name=pdf_filename,
        mime="application/pdf"
    )

# Tab 2: Average Voter Turnout by State
with tab2:
    st.header("Average Voter Turnout by State")
    st.write("This tab displays the average voter turnout percentage across different states. It helps in understanding voter engagement in various regions.")
    
    state_voter_turnout = filtered_merged_data.groupby("State")["Polled (%)"].mean().reset_index()
    fig = px.bar(state_voter_turnout, x="State", y="Polled (%)", title="Average Voter Turnout by State")
    fig.update_layout(xaxis_tickangle=-90, width=900, height=600)
    st.plotly_chart(fig)

    # PDF Report
    content = [
        "Average Voter Turnout by State:",
        state_voter_turnout.to_string(index=False)
    ]
    pdf_filename = generate_pdf("Voter Turnout by State Report", content, file_name="voter_turnout_by_state_report.pdf")
    st.download_button(
        label="Download Voter Turnout Report",
        data=open(pdf_filename, "rb").read(),
        file_name=pdf_filename,
        mime="application/pdf"
    )

# Tab 3: Party Ad Spend
with tab3:
    st.header("Top 5 Political Parties by Ad Spend")
    st.write("This tab shows the top 5 political parties based on their advertisement spend. It provides insights into the political parties that are investing the most in ads.")
    
    party_ad_spend = filtered_advertisers.groupby("Page name")["Amount spent (INR)"].sum().nlargest(5).reset_index()
    fig = px.pie(party_ad_spend, values="Amount spent (INR)", names="Page name", title="Top 5 Parties by Ad Spend")
    st.plotly_chart(fig)

    # PDF Report
    content = [
        "Top 5 Political Parties by Ad Spend:",
        party_ad_spend.to_string(index=False)
    ]
    pdf_filename = generate_pdf("Top 5 Political Parties Ad Spend Report", content, file_name="top_5_parties_ad_spend_report.pdf")
    st.download_button(
        label="Download Party Ad Spend Report",
        data=open(pdf_filename, "rb").read(),
        file_name=pdf_filename,
        mime="application/pdf"
    )

# Tab 4: Ad Spend and Voter Turnout by Election Phase
with tab4:
    st.header("Ad Spend and Voter Turnout by Election Phase")
    st.write("This tab compares the advertisement spend and voter turnout percentage for each election phase. It highlights any correlations or trends between spending and voter participation.")
    
    phase_analysis = filtered_merged_data.groupby("Phase").agg({"Amount spent (INR)": "sum", "Polled (%)": "mean"}).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=phase_analysis["Phase"], y=phase_analysis["Amount spent (INR)"], name="Ad Spend (INR)", yaxis="y1"))
    fig.add_trace(go.Scatter(x=phase_analysis["Phase"], y=phase_analysis["Polled (%)"], name="Voter Turnout (%)", yaxis="y2", mode='lines+markers'))
    fig.update_layout(title="Ad Spend and Voter Turnout by Election Phase", width=900, height=600, yaxis2=dict(overlaying='y', side='right'))
    st.plotly_chart(fig)

    # PDF Report
    content = [
        "Ad Spend and Voter Turnout by Election Phase:",
        phase_analysis.to_string(index=False)
    ]
    pdf_filename = generate_pdf("Ad Spend and Voter Turnout by Phase Report", content, file_name="ad_spend_voter_turnout_by_phase_report.pdf")
    st.download_button(
        label="Download Phase Report",
        data=open(pdf_filename, "rb").read(),
        file_name=pdf_filename,
        mime="application/pdf"
    )

# Tab 5: Clustering Ad Spend and Voter Turnout
with tab5:
    st.subheader("Clustering Ad Spend and Voter Turnout")
    st.write("This tab uses K-Means clustering to segment states based on advertisement spend and voter turnout. It helps in identifying patterns or trends across various clusters of states.")
    
    # Check if there's data after filtering
    data = filtered_merged_data[["Amount spent (INR)", "Polled (%)"]].dropna()
    
    if data.empty:
        st.warning("No data available for clustering. Please adjust your filters.")
    else:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        data["Cluster"] = kmeans.fit_predict(data)
        
        fig = px.scatter(data, x="Amount spent (INR)", y="Polled (%)", color="Cluster", title="Clustering Analysis")
        st.plotly_chart(fig)

    # PDF Report
    content = [
        "Clustering of Ad Spend and Voter Turnout:",
        data.to_string(index=False)
    ]
    pdf_filename = generate_pdf("Clustering Report", content, file_name="clustering_report.pdf")
    st.download_button(
        label="Download Clustering Report",
        data=open(pdf_filename, "rb").read(),
        file_name=pdf_filename,
        mime="application/pdf"
    )

# Tab 6: Regression Analysis
with tab6:
    st.subheader("Regression Analysis")
    st.write("This tab provides a regression analysis to predict voter turnout based on advertisement spending. It helps to understand the relationship between ad spend and voter participation.")
    
    data = filtered_merged_data[["Amount spent (INR)", "Polled (%)"]].dropna()
    
    if data.empty:
        st.warning("No data available for regression analysis. Please adjust your filters.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(data[["Amount spent (INR)"]], data["Polled (%)"], test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        st.write(f"R² Score: {r2:.4f}")
        st.write(f"Mean Absolute Error: {mae:.4f}")

        fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"}, title="Actual vs Predicted")
        fig.add_shape(type="line", x0=y_test.min(), x1=y_test.max(), y0=y_test.min(), y1=y_test.max(), line=dict(color="red"))
        st.plotly_chart(fig)

    # PDF Report
    content = [
        f"Regression Analysis (R² Score: {r2:.4f}, MAE: {mae:.4f}):",
        f"Actual values vs Predicted values:\n{pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).to_string(index=False)}"
    ]
    pdf_filename = generate_pdf("Regression Analysis Report", content, file_name="regression_analysis_report.pdf")
    st.download_button(
        label="Download Regression Report",
        data=open(pdf_filename, "rb").read(),
        file_name=pdf_filename,
        mime="application/pdf"
    )

# Tab 7: Chi-Square Test for Independence
with tab7:
    st.subheader("Chi-Square Test")
    st.write("This tab performs a Chi-Square test to analyze the relationship between state and election phase. It helps determine if there is a significant association between these two variables.")
    
    # Create the contingency table
    contingency_table = pd.crosstab(filtered_merged_data["State"], filtered_merged_data["Phase"])
    
    # Check if the contingency table has any data
    if contingency_table.empty:
        st.warning("No data available for the Chi-Square Test. Please adjust your filters.")
    else:
        # Perform Chi-Square test
        chi2, p, _, _ = chi2_contingency(contingency_table)

        # Display the Chi-Square statistic and p-value
        st.write(f"Chi-Square Statistic: {chi2:.4f}")
        st.write(f"P-Value: {p:.4f}")
        
        # Interpretation based on the p-value
        if p < 0.05:
            st.write("The result is statistically significant. There is an association between the state and election phase.")
        else:
            st.write("The result is not statistically significant. There is no association between the state and election phase.")

        # Display the contingency table as a heatmap
        st.write("Contingency Table Heatmap:")
        fig = go.Figure(data=go.Heatmap(
            z=contingency_table.values,
            x=contingency_table.columns,
            y=contingency_table.index,
            colorscale='Viridis',  # Customize the color scale as needed
            colorbar=dict(title="Frequency")
        ))
        fig.update_layout(title="Contingency Table Heatmap", xaxis_title="Election Phase", yaxis_title="State")
        st.plotly_chart(fig)

        # PDF Report
        content = [
            f"Chi-Square Test Results: Chi2: {chi2:.2f}, p-value: {p:.4f}",
            f"Contingency Table Heatmap:\n{contingency_table.to_string()}"
        ]
        pdf_filename = generate_pdf("Chi-Square Test Report", content, file_name="chi_square_test_report.pdf")
        st.download_button(
            label="Download Chi-Square Test Report",
            data=open(pdf_filename, "rb").read(),
            file_name=pdf_filename,
            mime="application/pdf"
        )
# Tab 8: Support Vector Machine (SVM) Analysis
with tab8:
    st.subheader("Support Vector Machine (SVM) Analysis")
    st.write("This tab uses SVM to classify the relationship between ad spend and voter turnout. We aim to predict voter turnout based on advertisement spending.")

    # Prepare the data for SVM model
    data = filtered_merged_data[["Amount spent (INR)", "Polled (%)"]].dropna()
    
    if data.empty:
        st.warning("No data available for SVM analysis. Please adjust your filters.")
    else:
        # Define the feature (X) and target (y)
        X = data[["Amount spent (INR)"]]
        y = data["Polled (%)"]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create the SVM model
        from sklearn.svm import SVR
        svm_model = SVR(kernel="rbf")  # Using Radial Basis Function kernel
        svm_model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = svm_model.predict(X_test)

        # Calculate R² score and Mean Absolute Error (MAE)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Display results
        st.write(f"R² Score: {r2:.4f}")
        st.write(f"Mean Absolute Error: {mae:.4f}")

        # Plot the SVM model predictions
        fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"}, title="SVM: Actual vs Predicted")
        fig.add_shape(type="line", x0=y_test.min(), x1=y_test.max(), y0=y_test.min(), y1=y_test.max(), line=dict(color="red"))
        st.plotly_chart(fig)

        # PDF Report for SVM analysis
        content = [
            f"SVM Analysis (R² Score: {r2:.4f}, MAE: {mae:.4f}):",
            f"Actual values vs Predicted values:\n{pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).to_string(index=False)}"
        ]
        pdf_filename = generate_pdf("SVM Analysis Report", content, file_name="svm_analysis_report.pdf")
        st.download_button(
            label="Download SVM Analysis Report",
            data=open(pdf_filename, "rb").read(),
            file_name=pdf_filename,
            mime="application/pdf"
        )
# Tab 9: Random Forest Analysis
with tab9:
    st.subheader("Random Forest Analysis")
    st.write("This tab uses a Random Forest Regressor to predict voter turnout based on advertisement spending.")

    data = filtered_merged_data[["Amount spent (INR)", "Polled (%)"]].dropna()
    
    if data.empty:
        st.warning("No data available for Random Forest analysis. Please adjust your filters.")
    else:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data[["Amount spent (INR)"]], data["Polled (%)"], test_size=0.2, random_state=42)

        # Create the Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = rf_model.predict(X_test)

        # Calculate R² score and Mean Absolute Error (MAE)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Display results
        st.write(f"R² Score: {r2:.4f}")
        st.write(f"Mean Absolute Error: {mae:.4f}")

        # Plot the Random Forest model predictions
        fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"}, title="Random Forest: Actual vs Predicted")
        fig.add_shape(type="line", x0=y_test.min(), x1=y_test.max(), y0=y_test.min(), y1=y_test.max(), line=dict(color="red"))
        st.plotly_chart(fig)

        # PDF Report for Random Forest analysis
        content = [
            f"Random Forest Analysis (R² Score: {r2:.4f}, MAE: {mae:.4f}):",
            f"Actual values vs Predicted values:\n{pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).to_string(index=False)}"
        ]
        pdf_filename = generate_pdf("Random Forest Analysis Report", content, file_name="random_forest_analysis_report.pdf")
        st.download_button(
            label="Download Random Forest Report",
            data=open(pdf_filename, "rb").read(),
            file_name=pdf_filename,
            mime="application/pdf"
        )









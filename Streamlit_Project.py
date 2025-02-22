import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
st.title("Project CO2 emissions by vehicles")
st.sidebar.title("Table of contents")
pages = ["Exploration", "DataVizualization", "Modelling"]
page = st.sidebar.radio("Go to", pages)

if page == pages[0]: 
    st.write("### Data exploration")
    st.title('Data Cleaning')

    # Read the overall data set 
    df = pd.read_csv('filtered_data_2022_BE_NL_.csv')
    new_column_names = [
    "ID", "Country", "Vehicle family identification number", "Pool", "Manufacturer name (EU standard)",
    "Manufacturer name (OEM declaration)", "Manufacturer name (MS registry denomination)", "Type approval number",
    "Type", "Variant", "Version", "Make", "Commercial name", "Category of the vehicle type approved",
    "Category of the vehicle registered", "Total new registrations", "Mass in running order (kg)", "WLTP test mass",
    "Specific CO2 Emissions in g/km (NEDC)", "Specific CO2 Emissions in g/km (WLTP)", "Wheel base in mm",
    "Axle width steering axle in mm", "Axle width other axle in mm", "Fuel type", "Fuel mode", "Engine capacity in cm3",
    "Engine power in KW", "Electric energy consumption in Wh/km", "Innovative technology",
    "Emissions reduction through IT in g/km", "Emissions reduction through IT in g/km (WLTP)",
    "Deviation factor", "Verification factor", "Type of data", "Registration year", "Date of registration",
    "Fuel consumption", "Character corresponding to the provisions used for the type-approval", "Roadload (Matrix) family’s",
    "Electric range (km)"]
    df.columns = new_column_names
    st.write('### Raw Data', df.head())
    st.write(f'Dataset contains {10734656} rows and {40} columns.')
    #Delete the overall dataset to focus only on data from France 
    del df
    #Focus only on the data from France 
    df = pd.read_csv('filtered_data_2022_FR.csv')
    df.columns = new_column_names
    # Data type inspection and conversion
    st.write('### Data Type Inspection')
    st.write(df.dtypes)
    columns_to_convert = st.multiselect('Select columns to convert to numeric', df.columns.tolist())
    if columns_to_convert:
        df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
        st.write('Data types after conversion:', df.dtypes)

    # Data distribution across different countries 
    # Read dataset
    df_Specific = pd.read_csv('filtered_data_2022_Country.csv')

    # Set the Seaborn style to 'white' for a clean background without gridlines
    sns.set_style("white")

    # Calculate the absolute counts and percentages
    country_counts = df_Specific['Country'].value_counts()
    country_percentages = df_Specific['Country'].value_counts(normalize=True) * 100

    # Create the figure and axis objects with higher DPI for better quality
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)

    # Plot the absolute counts on the left y-axis
    country_counts.plot(kind='bar', ax=ax1, color='steelblue', alpha=0.7, position=1, width=0.4)

    # Set the labels and title for the left y-axis with custom fonts
    ax1.set_ylabel('Absolute Counts', color='steelblue', fontsize=14)
    ax1.set_xlabel('Country', fontsize=14)
    ax1.set_title('Country Distribution: Absolute Counts and Percentages', fontsize=16, fontweight='bold')

    # Create a second y-axis for the percentages
    ax2 = ax1.twinx()

    # Plot the percentages on the right y-axis
    country_percentages.plot(kind='bar', ax=ax2, color='green', alpha=0.5, position=0, width=0.4)

    # Set the labels for the right y-axis with custom fonts
    ax2.set_ylabel('Percentage (%)', color='green', fontsize=14)

    # Rotate x labels for better readability and set custom fonts
    plt.xticks(rotation=45, ha='right', fontsize=12)

    # Add a legend
    ax1.legend(['Absolute Counts'], loc='upper left')
    ax2.legend(['Percentage (%)'], loc='upper right')

    # Tighten layout for better spacing
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)    
    #Processing of the fuel type 





    # Handle missing values
    st.write('### Missing Value Handling')
    missing_values = df.isnull().sum()
    st.write('Missing values per column:', missing_values[missing_values > 0])

    # Option to fill or drop missing values
    if st.checkbox('Fill Missing Values'):
        fill_value = st.text_input('Enter value to fill missing data:', '0')
        df = df.fillna(fill_value)
        st.write('Missing values filled.')
    elif st.checkbox('Drop Missing Values'):
        df = df.dropna()
        st.write('Rows with missing values dropped.')

    st.write('### Cleaned Data', df.head())
    st.write(f'Cleaned dataset contains {df.shape[0]} rows and {df.shape[1]} columns.')

    st.write('---')

if page == pages[1]: 
    st.write("### Data Visualization")
    st.title('Data Visualization')

    # Read dataset directly from URL
    df = pd.read_csv(csv_url)
    st.write('### Data Preview', df.head())

    # Correlation matrix
    if st.checkbox('Show Correlation Matrix'):
        st.write('### Correlation Matrix')
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)

    # Distribution plots
    selected_columns = st.multiselect('Select columns for distribution plots', df.columns.tolist())
    if selected_columns:
        for column in selected_columns:
            st.write(f'### Distribution of {column}')
            fig, ax = plt.subplots()
            sns.histplot(df[column], kde=True, ax=ax)
            st.pyplot(fig)

    # Pairplot for selected features
    pairplot_columns = st.multiselect('Select columns for pairplot', df.columns.tolist(), default=df.columns[:3].tolist())
    if pairplot_columns:
        st.write('### Pairplot')
        sns.pairplot(df[pairplot_columns])
        st.pyplot()

    st.write('---')
    st.write('Developed by Koffi Koumi')

if page == pages[2]: 
    
    st.write("### Modelling")
    st.title('Data Modeling')

    import streamlit as st
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import mean_squared_error, f1_score, r2_score
    from imblearn.under_sampling import RandomUnderSampler
    import statsmodels.api as sm
    import shap
    from sklearn.ensemble import RandomForestClassifier
    import time
    from sklearn.ensemble import VotingClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.tree import DecisionTreeClassifier

    # Read dataset directly from the specified local file path
    data_file_path = 'numerical_data_2022_v2.csv'
    df = pd.read_csv(data_file_path)
    # High-Level Check on the Data
    st.write("### High-Level Check on the Data")
    st.write("#### Data Preview")
    st.write(df.head())
    st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

    # Correlation Matrix for numerical features
    st.write("#### Correlation Matrix")
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=ax)
    ax.set_title("Correlation Matrix of Numerical Features")
    st.pyplot(fig)

    # Encoding or removing the 'Fuel type' column
    st.write("#### Handle the 'Fuel type' Column")
    if 'Fuel type' in df.columns:
        fuel_type_action = st.radio(
            "What would you like to do with the 'Fuel type' column?",
            ('Encode using pd.get_dummies', 'Remove from dataset')
        )

        if fuel_type_action == 'Encode using pd.get_dummies':
            df_encoded = pd.get_dummies(df, columns=['Fuel type'], dtype=int)
            st.write("Data after encoding 'Fuel type' using pd.get_dummies:")
        else:
            df_encoded = df.drop(columns=['Fuel type'])
            st.write("Data after removing the 'Fuel type' column:")

        st.write(df_encoded.head())
    else:
        df_encoded = df  # In case the column doesn't exist, proceed without modification

    # Select task type: regression or classification
    task_type = st.selectbox('Select Task Type', ['Regression', 'Classification'])

    # Processing for regression tasks
    if task_type == 'Regression':
        st.write('### Regression Task Preprocessing')

        # Feature and target selection
        features = st.multiselect('Select Features for Regression', df_encoded.columns.tolist())
        target = st.selectbox('Select Target Variable for Regression', df_encoded.columns.tolist())

        if features and target:
            X = df_encoded[features]
            y = df_encoded[target]

            # Option to scale features
            if st.checkbox('Scale Features'):
                scaler = StandardScaler()
                st.write('Features have been scaled using StandardScaler.')

            # Split data into training and testing sets
            test_size = st.slider('Test Size', 0.1, 0.5, 0.2, key='reg_test_size')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            X_train=scaler.fit_transform(X_train)
            X_test=scaler.transform(X_test)
            st.write(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

            # Model selection for regression
            model_type = st.selectbox('Choose Regression Model', ['Recom: Linear Regression', 'Lasso Regression', 'Ridge Regression', 'ElasticNet'])

            # Train the selected model
            if model_type == 'Recom: Linear Regression':
                model = LinearRegression()
            elif model_type == 'Lasso Regression':
                model = LassoCV()
            elif model_type == 'Ridge Regression':
                model = RidgeCV()
            elif model_type == 'ElasticNet':
                model = ElasticNetCV()

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Display model evaluation metrics
            st.write('### Model Evaluation')
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)  # Calculate R-squared
            st.write(f'**Mean Squared Error (MSE)**: {mse:.4f}')
            st.write(f'**R-squared (R²)**: {r2:.4f}')
            # Display regression coefficients
            st.write('### Regression Coefficients')
            coefficients = pd.DataFrame(model.coef_, index=features, columns=['Coefficient'])
            st.write(coefficients)

            # Residual Plot
            st.write('### Residual Plot')
            residuals = y_test - predictions
            fig, ax = plt.subplots()
            sns.scatterplot(x=predictions, y=residuals, ax=ax)
            ax.axhline(0, linestyle='--', color='red', linewidth=2)
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title('Residuals vs. Predicted Values')
            st.pyplot(fig)

            # Probability Plot
            st.write('### Probability Plot')
            fig, ax = plt.subplots()
            sm.qqplot(residuals, line='45', ax=ax)
            ax.set_xlim(-6, 6)
            ax.set_title('Probability Plot of Residuals')
            st.pyplot(fig)

    # Processing for classification tasks
    elif task_type == 'Classification':
        st.write('### Classification Task Preprocessing')

        # Feature and target selection
        features = st.multiselect('Select Features for Classification', df_encoded.columns.tolist())
        target = st.selectbox('Select Target Variable for Classification', df_encoded.columns.tolist())

        if features and target:
            X = df_encoded[features]
            y = df_encoded[target]

            # Allow the user to define custom bin ranges and labels for segmentation
            st.write('#### Define Custom Bins for Target Segmentation')
            bins_input = st.text_input('Enter bin ranges (comma-separated):', '0, 100, 130, 160, 190, 220, 250, inf')
            labels_input = st.text_input('Enter labels for the bins (comma-separated):', '0, 1, 2, 3, 4, 5, 6')

            # Convert input strings to lists
            bins = [float(b.strip()) for b in bins_input.split(',')]
            labels = [int(l.strip()) for l in labels_input.split(',')]

            # Segment the target using the user-defined bins and labels
            st.write(f'Using the following bins: {bins} with labels: {labels}')
            y_binned = pd.cut(y, bins=bins, labels=labels)
            st.write(f'Target variable segmented into {len(labels)} classes:')
            st.write(y_binned.value_counts())

            # Update y with the segmented target
            y = y_binned

            # Display class distribution before undersampling
            st.write('### Target Class Distribution Before Undersampling')
            fig, ax = plt.subplots()
            sns.countplot(x=y)
            ax.set_title('Class Distribution Before Undersampling')
            st.pyplot(fig)

            # Option to perform undersampling
            if st.checkbox('Perform Undersampling'):
                 # Display current class distribution
                st.write('### Current Class Distribution')
                st.write(y.value_counts().to_dict())

                undersample_ratio = st.slider('Select undersampling ratio (0.1 to 1.0)', 0.1, 1.0, 0.5, key='class_undersample_ratio')
                
                # Calculate the desired number of samples for each class
                min_class_count = int(min(y.value_counts()) * undersample_ratio)
                sampling_strategy = {cls: min_class_count for cls in y.unique()}
                
                st.write(f'Sampling strategy for undersampling: {sampling_strategy}')

                rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
                X_res, y_res = rus.fit_resample(X, y)

                # Display class distribution after undersampling
                st.write('### Class Distribution After Undersampling')
                st.write(dict(pd.Series(y_res).value_counts()))

                # Plot the class distribution after undersampling
                fig, ax = plt.subplots()
                sns.countplot(x=y_res)
                ax.set_title('Class Distribution After Undersampling')
                st.pyplot(fig)

                X, y = X_res, y_res

            # Split data into training and testing sets
            test_size = st.slider('Test Size', 0.1, 0.5, 0.2, key='class_test_size')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            st.write(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

            # Model selection for classification
            model_type = st.selectbox('Choose Classification Model', ['Reco: Random Forest', 'Decision Tree'])

            # Train the selected model
            if model_type == 'Reco: Random Forest':
                model = RandomForestClassifier(n_jobs=-1,random_state=321)
            elif model_type == 'Decision Tree':
                model = DecisionTreeClassifier(criterion='entropy',max_depth=7,random_state=123)

            model.fit(X_train, y_train)
           
            # Display model evaluation metrics
            start_train = time.time()
            y_pred = model.predict(X_test)
            end_train = time.time()
            time_taken = end_train - start_train

            # Calculate metrics
            cm = confusion_matrix(y_test, y_pred)
            cr = classification_report(y_test, y_pred, output_dict=True)
            cr_text = classification_report(y_test, y_pred)  # For displaying in text format

            # Display confusion matrix as a DataFrame for better readability
            cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)

            # Display metrics in Streamlit
            st.write(f'**Time taken for prediction**: {time_taken:.4f} seconds')
            st.write('**Confusion Matrix**:')
            st.dataframe(cm_df)  # Display the confusion matrix as a DataFrame
            st.write('**Classification Report**:')
            st.text(cr_text)  # Display the text version of the classification report
            # SHAP Analysis for classification models
            if st.checkbox('Show SHAP Analysis for Classification', value=False):
                st.write('#### SHAP Analysis for Classification Model')
                explainer = shap.Explainer(model, X_test)
                shap_values = explainer(X_test)
                for class_idx, class_name in enumerate(model.classes_):
                    st.write(f"SHAP values for class: {class_name}")
                    shap.summary_plot(shap_values[..., class_idx], X_test, plot_type="bar", show=False)
                    st.pyplot(plt.gcf())
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
st.title("Project CO2 emissions by vehicles")
st.sidebar.title("Table of contents")
pages = ["Executive summary","Methodology","Data exploration", "Modelling","Additional Exploration" ]
page = st.sidebar.radio("Go to", pages)

if page == pages[0]: 
    st.write("## Executive summary")
    st.write("""
        This project addresses the critical challenge of reducing CO₂ emissions from vehicles, a major contributor to global greenhouse gas emissions, particularly in the transportation sector. By analyzing a large dataset of vehicle registrations and emissions tests from the European Union, this study aimed to understand the key factors influencing CO₂ emissions and develop machine learning models to predict emissions based on vehicle characteristics.

        **Objectives:**
        - Analyze the relationship between vehicle characteristics (mass, engine capacity, fuel type) and CO₂ emissions.
        - Develop predictive models to accurately estimate emissions based on these features.
        - Identify the most significant factors that impact emissions, providing insights for vehicle manufacturers to design more sustainable cars.

        Several machine learning algorithms were tested for regression and classification tasks. **Linear regression** was selected for predicting absolute CO₂ values. When fuel consumption was included as a predictor, the model achieved an **R² of 0.99** and an **RMSE of 5.6 g/km**. Without fuel consumption, the model still performed well, achieving an **R² of 0.87** and an **RMSE of 14.38 g/km**, indicating that fuel consumption plays a significant role but the model remains robust without it.

        For classification tasks, **Random Forest** was chosen as the best model. When fuel consumption was included, the model achieved **an accuracy of 98.3%** and an **F1-score of 0.98**. Without fuel consumption, the model still performed well, with **an accuracy of 94.2%** and an **F1-score of 0.94**, demonstrating robustness even when key variables were excluded. The prediction time remained **0.1 seconds**, making it suitable for real-time applications. These models provide actionable insights for vehicle manufacturers and regulators, helping to design greener vehicles and meet sustainability goals.
        """)

if page == pages[1]: 
    st.write("## Methodology")
    st.write("""
            ### Data Collection and Preprocessing
            - **Choosing the dataset**: The project utilized a dataset from the European Environment Agency (EEA) containing CO₂ emission records for over 10 million vehicles registered in the EU and associated countries.
            - **Data Cleaning**: Missing values were handled by removing columns with more than 50% missing data and imputing where necessary. Redundant features with high correlation, such as vehicle mass and fuel consumption, were removed to avoid multicollinearity.
            - **Outlier Detection**: Box plots and Z-scores were used to detect outliers, which were retained if they represented legitimate vehicle data, such as high-performance cars.
            - **Normalization**: Features like engine size and power were normalized using Min-Max scaling to ensure uniformity for machine learning algorithms.
         
            ### Feature Engineering
            Key features selected for the analysis included vehicle mass, engine capacity, engine power, fuel type, and CO₂ emissions (target variable). Innovative technologies like LED lighting were also encoded for analysis. The analysis was based on the work of Zubair et al., and Al-Nefaie et al.

            ### Machine Learning Models
            - **Regression**: Linear regression was chosen due to its balance between simplicity and performance. The model achieved an **R² of 0.99** and an **RMSE of 5.6 g/km** when fuel consumption was included, and an **R² of 0.87** and **RMSE of 14.38 g/km** without it.
            - **Classification**: Random Forest was selected as the best model for classifying vehicles into seven CO₂ emission categories. When fuel consumption was included, it achieved **an accuracy of 98.3%** and an **F1-score of 0.98**. Without fuel consumption, it still performed well with **an accuracy of 94.2%** and **F1-score of 0.94**, demonstrating robustness even when key variables were excluded.

            These models provide actionable insights into the relationship between vehicle characteristics and CO₂ emissions, offering potential solutions to reduce the environmental impact of the automotive industry.
            """)

if page == pages[2]: 
    # Use st.cache_data to cache the data loading
    @st.cache_data
    def load_data(file_path):
        df = pd.read_csv(file_path)
        return df

    # Use st.cache_data to cache the data loading of multiple files
    @st.cache_data
    def load_and_prepare_data(file_path, column_names):
        df = pd.read_csv(file_path)
        df.columns = column_names
        return df

    # Define column names
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
        "Electric range (km)"
    ]

    # Load datasets
    st.write("## Data Overview")
    df = load_and_prepare_data('filtered_data_2022_BE_NL_.csv', new_column_names)
    st.write('### Raw Data', df.head())
    st.write(f'Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.')

    # Delete the dataset to free memory and focus only on French data
    del df

    # Load the French dataset
    df_france = load_and_prepare_data('filtered_data_2022_FR.csv', new_column_names)

    # Data type inspection and conversion
    st.write('### Data Type Inspection')
    st.write(df_france.dtypes)

    # Load specific dataset for country distribution
    df_specific = load_data('filtered_data_2022_Country.csv')

    # Set the Seaborn style to 'white' for a clean background without gridlines
    sns.set_style("white")

    # Calculate the absolute counts and percentages
    country_counts = df_specific['Country'].value_counts()
    country_percentages = df_specific['Country'].value_counts(normalize=True) * 100

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

    # Handle missing values
    st.write('### Missing Value Handling')
    missing_values = df_france.isnull().sum()
    st.write('Missing values per column:', missing_values[missing_values > 0])

    # Option to fill or drop missing values
    if st.checkbox('Fill Missing Values'):
        fill_value = st.text_input('Enter value to fill missing data:', '0')
        df_france = df_france.fillna(fill_value)
        st.write('Missing values filled.')
    elif st.checkbox('Drop Missing Values'):
        df_france = df_france.dropna()
        st.write('Rows with missing values dropped.')

    # Load cleaned data for visualization
    df_cleaned = load_data('Data_For_Visualization_2022.csv')
    st.write('### Cleaned Data', df_cleaned.head())
    st.write(f'Cleaned dataset contains {df_cleaned.shape[0]} rows and {df_cleaned.shape[1]} columns.')

    # Correlation matrix
    if st.checkbox('Show Correlation Matrix'):
        st.write('### Correlation Matrix')
        numerical_df = df_cleaned.select_dtypes(include='number')
        corr = numerical_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)

    # Pairplot for selected features (after correlation matrix)
    selected_columns = ['Specific CO2 Emissions in g/km (WLTP)', 'Engine capacity in cm3', 'Engine power in KW', 'Electric energy consumption in Wh/km','Fuel consumption']

    # Ensure that the selected columns are numerical
    numeric_columns = df_cleaned[selected_columns].select_dtypes(include=['float64', 'int64'])

    # Check if there are any numeric columns to avoid ValueError
    if not numeric_columns.empty:
        st.write('### Pairplot of selected numerical columns')
        
        # Create the pairplot for the selected columns
        fig = sns.pairplot(numeric_columns).fig
        
        # Pass the figure to st.pyplot
        st.pyplot(fig)
    else:
        st.error('The selected columns do not contain numeric data for the pairplot.')

    st.write('---')


if page == pages[3]:

    st.write("## Modelling")
    import streamlit as st
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    from imblearn.under_sampling import RandomUnderSampler
    import statsmodels.api as sm
    import time
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.tree import DecisionTreeClassifier
    import shap
    from PIL import Image 
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Lasso
    import pickle
    
    
    @st.cache_data
    def load_data(file_path):
        return pd.read_csv(file_path)
    # Load the dataset
    data_file_path = 'numerical_data_2022_v2.csv'
    df_or = load_data(data_file_path)
    
    # High-Level Check on the Data
    st.write("#### Data Preview")
    st.write(df_or.head())
    st.write(f"Dataset contains {df_or.shape[0]} rows and {df_or.shape[1]} columns.")
   
    st.write('### Correlation Matrix')
    numerical_df = df_or.select_dtypes(include='number')
    corr = numerical_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)
        # Function to load pre-trained model from a file
    def load_model(file_path):
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model

    # Allow the user to choose between a model with or without fuel consumption
    fuel_option = st.selectbox(
        "Choose the type of model:",
        ["With Fuel Consumption", "Without Fuel Consumption"]
    )

    # Allow the user to choose task type (Regression or Classification)
    task_type = st.selectbox(
        "Choose the type of task:",
        ["Regression", "Classification"]
    )

    # Load your dataset (df) here and ensure it's preprocessed as per your requirements
        # Cache the data loading function
    @st.cache_data
    def load_data(file_path):
        return pd.read_csv(file_path)

    # Load the dataset
    data_file_path = 'numerical_data_2022_v2.csv'
    df_or = load_data(data_file_path)
    
    # High-Level Check on the Data
    st.write("#### Data Preview")
    st.write(df_or.head())
    st.write(f"Dataset contains {df_or.shape[0]} rows and {df_or.shape[1]} columns.")

    
    
    # Depending on the user's selection, drop or keep the 'Fuel consumption' column
    if fuel_option == "Without Fuel Consumption" and 'Fuel consumption' in df_or.columns:
        df = df_or.drop('Fuel consumption', axis=1)
    else:
        df = df_or.copy()

    df = df.drop('Mass in running order (kg)', axis=1)  # Drop unnecessary column
    del df_or
    
     # Handling 'Fuel type' column
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
        df_encoded = df  # If the column doesn't exist, proceed without modification


    # Regression Model Selection
    if task_type == "Regression":
        # Allow the user to choose between different regression models
        model_type = st.selectbox(
            "Choose the regression model:",
            ["Linear Regression", "Lasso Regression", "Ridge Regression"]
        )

        # Feature and target selection based on the model (adjust accordingly)
        features = st.multiselect('Select Features for Regression',df_encoded.columns.tolist())
        target = st.selectbox('Select Target Variable for Regression', df_encoded.columns.tolist())

        if features and target:
            X = df_encoded[features]
            y = df_encoded[target]
           

            # Split the data into training and testing sets
            test_size = st.slider('Test Size', 0.1, 0.5, 0.2)
            scaler = None
            if st.checkbox('Scale Features'):
                scaler = StandardScaler()
                st.write('Features have been scaled using StandardScaler.')
            # Split data into training and testing sets
            test_size = st.slider('Test Size', 0.1, 0.5, 0.2, key='reg_test_size')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            @st.cache_data
            def load_model(model_type, fuel_option):
                model_file = ''
                
                if model_type == "LR":
                    model_file = 'LR_model_with_fuel.pkl' if fuel_option == "With Fuel Consumption" else 'LR_model_without_fuel.pkl'
                elif model_type == "Lasso":
                    model_file = 'Lasso_model_with_fuel.pkl' if fuel_option == "With Fuel Consumption" else 'Lasso_model_without_fuel.pkl'
                
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                
                return model

            if model_type == "Linear Regression":
                if fuel_option == "With Fuel Consumption":
                    selected_model= load_model("LR", "With Fuel Consumption")
                    model_Coeff=load_model("Lasso", "With Fuel Consumption") 
                else:
                    selected_model= load_model("LR", "Without Fuel Consumption") 
                    model_Coeff=load_model("Lasso", "Without Fuel Consumption") 
            elif model_type == "Lasso Regression":
                if fuel_option == "With Fuel Consumption":
                     selected_model= load_model("Lasso", "With Fuel Consumption") 
                     model_Coeff=selected_model
                else:
                    selected_model= load_model("Lasso", "Without Fuel Consumption") 
                    model_Coeff=selected_model

          
            model = selected_model
        

            # Make predictions using the loaded model
            predictions = model.predict(X_test)

            # Display model evaluation metrics
            st.write('### Model Evaluation')
            mse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            st.write(f'**Mean Squared Error (MSE)**: {mse:.4f}')
            st.write(f'**R-squared (R²)**: {r2:.4f}')

            # Display regression coefficients
            st.write('### Regression Coefficients')
            coefficients = pd.DataFrame(model_Coeff.coef_, index=features, columns=['Coefficient'])
            st.write(coefficients)

            # Create a visual plot for regression coefficients
            st.write('### Visual Representation of Regression Coefficients')
            fig, ax = plt.subplots(figsize=(10, 6))
            coefficients['Coefficient'].plot(kind='barh', ax=ax, color='skyblue')
            ax.set_title(f'Feature Coefficients in the {model_type}')
            ax.set_xlabel('Coefficient Value')
            ax.set_ylabel('Features')
            for i, v in enumerate(coefficients['Coefficient']):
                ax.text(v, i, f'{v:.2f}', va='center')
            st.pyplot(fig)

            # Residual Plot
            @st.cache_resource
            def create_residual_plot(predictions, residuals):
                fig, ax = plt.subplots()
                sns.scatterplot(x=predictions, y=residuals, ax=ax)
                ax.axhline(0, linestyle='--', color='red', linewidth=2)
                ax.set_xlabel('Predicted Values')
                ax.set_ylabel('Residuals')
                ax.set_title('Residuals vs. Predicted Values')
                return fig

            residuals = y_test - predictions
            st.write('### Residual Plot')
            residual_fig = create_residual_plot(predictions, residuals)
            st.pyplot(residual_fig)

            # Probability Plot
            @st.cache_resource
            def create_probability_plot(residuals):
                fig, ax = plt.subplots()
                sm.qqplot(residuals, line='45', ax=ax)
                ax.set_xlim(-6, 6)
                ax.set_title('Probability Plot of Residuals')
                return fig

            st.write('### Probability Plot')
            probability_fig = create_probability_plot(residuals)
            st.pyplot(probability_fig)

            
    # Classification Model Selection
    elif task_type == "Classification":
        # Allow the user to choose between different classification models
        model_type = st.selectbox(
            "Choose the classification model:",
            ["Random Forest", "Decision Tree"]
        )

        # Feature and target selection based on the model (adjust accordingly)
        features = st.multiselect('Select Features for Classification', df_encoded.columns.tolist())
        target = st.selectbox('Select Target Variable for Classification', df_encoded.columns.tolist())

        if features and target:
            X = df_encoded[features]
            y = df_encoded[target]
            # Custom binning for target segmentation
            st.write('#### Bins for Target Segmentation')
            bins_input = st.text_input('Enter bin ranges (comma-separated):', '0, 100, 130, 160, 190, 220, 250, inf')
            labels_input = st.text_input('Enter labels for the bins (comma-separated):', '0, 1, 2, 3, 4, 5, 6')

            @st.cache_data
            def load_image(image_path):
                img = Image.open(image_path)
                return img
            # Path to the image
            image_path = 'Car_labels.jpg'
            # Load and display the image
            image = load_image(image_path)
            st.image(image, caption='Car Fuel Consumption and CO2 Emission Labels', use_column_width=True)


            bins = [float(b.strip()) for b in bins_input.split(',')]
            labels = [int(l.strip()) for l in labels_input.split(',')]

            st.write(f'Using the following bins: {bins} with labels: {labels}')
            y_binned = pd.cut(y, bins=bins, labels=labels)

            y = y_binned

            # Display class distribution
            st.write('### Target Class Distribution Before Undersampling')
            fig, ax = plt.subplots()
            sns.countplot(x=y)
            ax.set_title('Class Distribution Before Undersampling')
            st.pyplot(fig)

            # Option to perform undersampling
            if st.checkbox('Perform Undersampling'):
                undersample_ratio = st.slider('Select undersampling ratio (0.1 to 1.0)', 0.1, 1.0, 0.5, key='class_undersample_ratio')
                min_class_count = int(min(y.value_counts()) * undersample_ratio)
                sampling_strategy = {cls: min_class_count for cls in y.unique()}

                @st.cache_data
                def undersample(X, y, sampling_strategy):
                    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
                    return rus.fit_resample(X, y)

                X_res, y_res = undersample(X, y, sampling_strategy)

                st.write('### Class Distribution After Undersampling')
                fig, ax = plt.subplots()
                sns.countplot(x=y_res)
                ax.set_title('Class Distribution After Undersampling')
                st.pyplot(fig)

                X, y = X_res, y_res

            # Split data into training and testing sets
            test_size = st.slider('Test Size', 0.1, 0.5, 0.2, key='class_test_size')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            st.write(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
            scaler = None
            if st.checkbox('Scale Features'):
                scaler = StandardScaler()
                st.write('Features have been scaled using StandardScaler.')
                X_train=scaler.fit_transform(X_train)
                X_test=scaler.transform(X_test)

            # Cache the model training function for classification


            # Loading the model based on the user's choice
            @st.cache_data
            def load_model(model_type, fuel_option):
                model_file = ''
                
                if model_type == "Random Forest":
                    model_file = 'random_forest_model_with_fuel.pkl' if fuel_option == "With Fuel Consumption" else 'random_forest_model_without_fuel.pkl'
                elif model_type == "Decision Tree":
                    model_file = 'decision_tree_model_with_fuel.pkl' if fuel_option == "With Fuel Consumption" else 'decision_tree_model_without_fuel.pkl'
                
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                
                return model


            if model_type == "Random Forest":
                if fuel_option == "With Fuel Consumption":
                    selected_model= load_model("Random Forest", "With Fuel Consumption")
                else:
                    selected_model= load_model("Random Forest", "Without Fuel Consumption") 
            elif model_type == "Decision Tree":
                if fuel_option == "With Fuel Consumption":
                     selected_model= load_model("Decision Tree", "With Fuel Consumption") 
                else:
                    selected_model= load_model("Decision Tree", "Without Fuel Consumption") 


            model = selected_model
            #load_model(model_file)

            # Make predictions using the loaded model
            # Make predictions
            start_train = time.time()
            y_pred = model.predict(X_test)
            end_train = time.time()

            # Display model evaluation metrics
            st.write(f'**Time taken for prediction**: {end_train - start_train:.4f} seconds')
            cm = confusion_matrix(y_test, y_pred)
            cr = classification_report(y_test, y_pred, output_dict=True)
            cr_text = classification_report(y_test, y_pred)

            # Confusion Matrix and Classification Report
            st.write('**Confusion Matrix**:')
            cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
            st.dataframe(cm_df)
            st.write('**Classification Report**:')
            st.text(cr_text)

            # SHAP Analysis for classification models
            # Cache the PCA transformation to improve performance
            feature_names=X.columns
            @st.cache_data
            def perform_pca(X_train, n_components=2):
                # Define PCA pipeline
                pca = PCA(n_components=n_components)
                scaler = StandardScaler()
                pipeline_pca = Pipeline(steps=[('normalization', scaler), ('pca', pca)])
                
                # Transform the data
                X_new = pipeline_pca.fit_transform(X_train)
                coeff = pca.components_.transpose()
                
                return X_new, coeff, pipeline_pca

            # Perform PCA with 2 components
            X_new, coeff, pipeline_pca = perform_pca(X_train)

            # Calculate scaling for the plot
            xs = X_new[:, 0]
            ys = X_new[:, 1]
            scalex = 1.0 / (xs.max() - xs.min())
            scaley = 1.0 / (ys.max() - ys.min())

            # Step 2: Prepare DataFrame for plotting
            principalDf = pd.DataFrame({'PC1': xs * scalex, 'PC2': ys * scaley})
            y_train_pred = model.predict(X_train)  # Assuming clf_RF is a trained RandomForestClassifier
            finalDF = pd.concat([principalDf, pd.Series(y_train_pred, name='CO2 emission class')], axis=1)

            # Step 3: Create the PCA plot
            def create_pca_plot(finalDF, coeff, feature_names, n_features):
                plt.figure(figsize=(13, 10))
                plt.title('PCA of CO2 Emission Data')
                
                # Scatter plot for PCA components
                sns.scatterplot(x='PC1', y='PC2', hue='CO2 emission class', data=finalDF, alpha=0.5)
                
                # Plot the vectors (arrows) for principal components
                for i in range(n_features):
                    plt.arrow(0, 0, coeff[i, 0] * 1.5, coeff[i, 1] * 1.5, color='red', alpha=0.5, head_width=0.01)
                    plt.text(coeff[i, 0] * 1.5, coeff[i, 1] * 1.5, feature_names[i], color='red')

                plt.xlim(-0.6, 0.8)
                plt.ylim(-0.8, 0.8)

                # Display the plot in Streamlit
                st.pyplot(plt)

            # Step 4: Visualize the PCA plot in Streamlit
            if st.button('Generate PCA Plot'):
                st.write('### PCA of CO2 Emission Data')
                create_pca_plot(finalDF, coeff, feature_names, n_features=X_train.shape[1])


                # Step 1: Ensure X_test is a DataFrame with proper feature names
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=feature_names)

        # Step 2: SHAP Analysis for multi-class classification
       # SHAP analysis
        st.write("### SHAP Analysis for Multiclass Classification")

        # If X_test is not already a DataFrame, convert it to one with actual feature names
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=feature_names)

        # Calculate SHAP values for multiclass classification
        explainer = shap.TreeExplainer(model)  # Replace 'clf_RF' with your actual model
        shap_values = explainer.shap_values(X_test[0:100])

        # Display the shapes of the SHAP values and X_test for debugging
        st.write("Shape of shap_values:", np.array(shap_values).shape)
        st.write("Shape of X_test:", X_test[0:100].shape)

        # Reshape shap_values to match the expected format
        n_samples, n_features, n_classes = np.array(shap_values).shape
        reshaped_shap_values = [shap_values[:, :, i] for i in range(n_classes)]

        # Generate SHAP summary plots for each class
        for i in range(n_classes):
            st.subheader(f"SHAP Feature Importance for Class {i}")
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(reshaped_shap_values[i], X_test[0:100], plot_type="bar", feature_names=feature_names, show=False)
            plt.title(f"SHAP Feature Importance for Class {i}")
            plt.tight_layout()
            st.pyplot(fig)

        # Overall feature importance across all classes
        st.subheader("Overall SHAP Feature Importance")
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(reshaped_shap_values, X_test[0:100], plot_type="bar", feature_names=feature_names, show=False)
        plt.title("Overall SHAP Feature Importance")
        plt.tight_layout()
        st.pyplot(fig)

if page == pages[4]:
    st.title("Additional Exploration" )
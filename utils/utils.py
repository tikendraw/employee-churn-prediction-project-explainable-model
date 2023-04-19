import sklearn
import warnings
import streamlit as st
import shap
import numpy as np
import pandas as pd

def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    # Remove the internal helper function
    #check_is_fitted(column_transformer)

    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                return (
                    column
                    if (
                        (not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)
                    )
                    else column_transformer._df_columns[column]
                )
            indices = np.arange(column_transformer._n_features)
            return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            return [] if column is None else [f"{name}__{f}" for f in column]
        return [f"{name}__{f}" for f in trans.get_feature_names()]

    ### Start of processing
    feature_names = []

    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))


    for name, trans, column, _ in l_transformers: 
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [f"{name}__{f}" for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))

    return feature_names








def explain(data, model, column_transformer):
    shap.initjs()

    #write a shap explainer for model
    st.header('Explaination')
    st.write(
        """
        Bars in Red shows the positive correlation of inputs to the Churning Outcome while 
        bars in Blue show the negative correlation of inputs.
        
        The shap values are the expected value plus the shap value for each input"""
    )
    explainer = shap.TreeExplainer(model)

    feature_names = get_feature_names(column_transformer)
    df = pd.DataFrame(data, columns=feature_names)

    shap_values = explainer.shap_values(df)
    # shap.summary_plot(shap_values, datadf)
    # shap.force_plot(explainer.expected_value, shap_values, df)

    shapl = explainer(df)

    def tab1_content():
        shap.plots.waterfall(shapl[0], max_display=20)

        pos_mask = (shapl.values>0).flatten()
        neg_mask = (shapl.values<0).flatten()

        pos_inp = np.array(df.columns)[pos_mask]
        neg_inp = np.array(df.columns)[neg_mask]



        st.pyplot()

        c20,c21 = st.columns(2)
        with c20:
            st.write('Positive Inputs toward Churn: ')
            for i in pos_inp:
                st.markdown(f"* <p  style=\"color:#f73664;font-size:18px;border-radius:2%;\">{i }</p>", unsafe_allow_html=True)

        with c21:
            st.write('Negative Inputs toward Churn: ')
            for i in neg_inp:
                st.markdown(f"* <p  style=\"color:#369af7;font-size:18px;border-radius:2%;\">{i }</p>", unsafe_allow_html=True)


    def tab2_content():
        shap.summary_plot(shap_values, df, plot_type="bar")
        st.pyplot()
    def how_to_read():
        st.write("The SHAP Waterfall Chart is a visual representation of the contribution of each feature to a model's prediction. Here's how to read it:")
        st.markdown('1. The y-axis represents the features in the model, with the topmost feature being the most important one and the bottom feature being the least important one.')
        st.markdown("2. The x-axis represents the feature value contribution to the model prediction. The further to the right a bar is, the higher its contribution to the model prediction.")
        st.markdown("3. The base value is represented by a dashed line and shows the model's average output. Each feature is then added to this base value in sequence, with the contribution of each feature shown by a colored bar.")
        st.markdown("4. If the bar extends to the right of the base value, it means that the feature is pushing the prediction higher than the average. If the bar extends to the left of the base value, it means that the feature is pushing the prediction lower than the average.")
        st.markdown("5. The sum of all the contributions (colored bars) plus the base value equals the final prediction of the model.")
        st.markdown("6. The chart can help you understand which features are driving a prediction and in what direction, which can be useful for interpreting the model's behavior and identifying potential issues.")

    
    tab1, tab2, tab3 = st.tabs(["WaterFlow Chart", "Summary Chart", "How to Read"])

    with tab1:
        st.header("WaterFlow Chart")
        tab1_content()

    with tab2:
        st.header("Summary Chart")
        tab2_content()

    with tab3:
        st.header("How To Read the WaterFlow Chart")
        how_to_read()




if __name__=='__main__':
    print('main')
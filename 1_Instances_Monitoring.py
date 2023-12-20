import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np

import streamlit as st

import utils.titan_ico as ti

#######################################
# This is how you should run this code:
# streamlit run main.py
# Not using python main.py !!!
#######################################

# Create Supabase client
supabase = ti.initialize_supabase()

def main():
    ti.initialize_streamlit('Instances Monitoring')
    instance_ids = ti.query_instance_ids(supabase)

    # Create two columns and placeholder
    col1, col2 = st.columns(2)
    col1_slot = col1.empty()
    col2_slot = col2.empty()
    chart_slot = st.empty()

    if 'df_all_instances' not in st.session_state:
        # Initialize df_all_instances before the loop
        df_all_instances = pd.DataFrame(columns=['instanceid', 'ChatGPT_Verdict', 'Indicators'])

        for instance_id in instance_ids:
            instance_df = ti.create_instance_dataframe(instance_id, supabase)
            # Concatenate the instance-specific data to df_all_instances
            df_all_instances = pd.concat([df_all_instances, instance_df], ignore_index=True)

            ti.display_leds(instance_df['instanceid'][0], instance_df['Indicators'][0], instance_df['ChatGPT_Verdict'][0].comment)

        st.session_state['df_all_instances'] = df_all_instances
    else:
        df_all_instances = st.session_state['df_all_instances']
        for _, instance_df in df_all_instances.iterrows():
            ti.display_leds(instance_df['instanceid'], instance_df['Indicators'], instance_df['ChatGPT_Verdict'].comment)

    if 'costs_response' not in st.session_state:
        costs_response = ti.fetch_costs_data(supabase)
        st.session_state['costs_response'] = costs_response
    else:
        costs_response = st.session_state['costs_response']

    total_cost = sum(costs_response['blended_costs'])
    unused_n = df_all_instances['Indicators'].str.contains('red', case=False, na=False).sum()
    predicted_savings = total_cost/len(instance_ids)*unused_n*0.3

    with col1_slot:
        st.metric(label='Total Costs Last Month', value='{:.2f} USD'.format(total_cost))
    with col2_slot:
        st.metric(label='Predicted Cost Savings per Month', value='{:.2f} USD'.format(predicted_savings))
    chart_slot.bar_chart(costs_response, x='datetimes', y='blended_costs')


if __name__ == "__main__":
    main()
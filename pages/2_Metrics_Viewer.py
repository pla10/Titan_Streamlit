import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd

from supabase import create_client
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

import utils.titan_ico as ti


# Create Supabase client
supabase = ti.initialize_supabase()

def main():
    ti.initialize_streamlit('Metrics Viewer')

    if 'instance_ids' not in st.session_state:
        switch_page("Instances Monitoring")
    else:
        # Query instance IDs from the table
        instance_ids = st.session_state['instance_ids']

    selected_instance = ti.display_data_selection(instance_ids)

    metrics_response = ti.fetch_metrics_data(supabase, selected_instance)

    ti.display_metrics_data(metrics_response)


if __name__ == "__main__":
    main()
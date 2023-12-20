import os
import random
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from statistics import mean, variance, mode
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from models.verdict import Verdict
from supabase import create_client
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

def initialize_supabase():
    """
    Initialize the Supabase client.
    """
    # Load the .env file
    load_dotenv()
    # Supabase configuration
    SUPABASE_URL = os.environ.get('SUPABASE_URL')
    SUPABASE_KEY = os.environ.get('SUPABASE_KEY')
    # Create Supabase client
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def initialize_streamlit(page_name):
    if 'page_title' not in st.session_state:
        st.session_state['page_title'] = 'Titan Infra Cloud Optimization'
    if 'page_icon' not in st.session_state:
        st.session_state['page_icon'] = '🚀'

    st.set_page_config(
        page_title=st.session_state['page_title'],
        page_icon=st.session_state['page_icon']
    )

    # Streamlit app
    st.title(st.session_state['page_icon'] + ' ' + st.session_state['page_title'])

    st.sidebar.success(page_name)

def display_data_selection(instance_ids):
    """
    Display the instance ID selection UI.
    """
    if 'instance_id_selected' in st.session_state:
        default_ix = instance_ids.index(st.session_state['instance_id_selected'])
        selected_instance = st.selectbox('Select the AWS EC2 Instance ID', instance_ids, index=default_ix)
    else:
        selected_instance = st.selectbox('Select the AWS EC2 Instance ID', instance_ids)
    return selected_instance

def display_metrics_data(metrics_data):
    """
    Display metrics data and plots in Streamlit.
    """
    if metrics_data:
        # Initialize empty arrays
        global timestamps
        global cpu_utilizations
        global network_ins
        global network_outs
        global ebs_read_ops
        global ebs_write_ops

        # Get the lengths of each component
        lengths = [len(metrics_data[0][key]) for key in ["timestamps", "cpu_utilizations", "network_ins", "network_outs", "ebs_read_ops", "ebs_write_ops"]]

        # Find the maximum length
        max_length = max(lengths)

        # Pad shorter arrays with zeros and fill void arrays with -500
        for key in ["timestamps", "cpu_utilizations", "network_ins", "network_outs", "ebs_read_ops", "ebs_write_ops"]:
            data = np.array(metrics_data[0][key])
            if len(data) < max_length:
                print(f'{key}: padded {max_length - len(data)} values')
                data = np.pad(data, (0, max_length - len(data)), 'constant', constant_values=0)
            if len(data) == 0:
                print(f'{key}: void filled')
                data = np.full(max_length, -500)

            # Assign the modified data to the corresponding variable
            globals()[key] = data

        # Create Pandas DataFrame
        df = pd.DataFrame({
            "timestamps": timestamps,
            "cpu_utilizations": cpu_utilizations,
            "network_ins": network_ins,
            "network_outs": network_outs,
            "ebs_read_ops": ebs_read_ops,
            "ebs_write_ops": ebs_write_ops
        })

        del timestamps
        del cpu_utilizations
        del network_ins
        del network_outs
        del ebs_read_ops
        del ebs_write_ops

        display_timeseries_plots(df)
    else:
        st.warning("No data available for the selected instance.")

def display_timeseries_plots(df):
    """
    Display timeseries plots for each metric in Streamlit.
    """
    # Plot timeseries for each metric
    st.title("CPU Utilization")
    st.line_chart(df.set_index("timestamps")["cpu_utilizations"])

    # Create two columns
    col1, col2 = st.columns(2)
    # Plot timeseries for each metric in separate columns
    with col1:
        st.title("Network In")
        st.line_chart(df.set_index("timestamps")["network_ins"])
        st.title("EBS Read Ops")
        st.line_chart(df.set_index("timestamps")["ebs_read_ops"])

    with col2:
        st.title("Network Out")
        st.line_chart(df.set_index("timestamps")["network_outs"])
        st.title("EBS Write Ops")
        st.line_chart(df.set_index("timestamps")["ebs_write_ops"])
    # Display the DataFrame
    st.write("Raw Data:", df)

def fetch_metrics_data(supabase, selected_instance):
    """
    Fetch metrics data for the selected instance from the metrics table.
    """
    metrics_query = supabase.from_('tata_metrics').select('*').eq('instanceid', selected_instance)
    metrics_response = metrics_query.execute()
    metrics_response = metrics_response.model_dump()
    return metrics_response['data']

def fetch_costs_data(supabase):
    """
    Fetch metrics data for the selected instance from the metrics table.
    """
    metrics_query = supabase.from_('tata_costs').select('*')
    metrics_response = metrics_query.execute()
    metrics_response = metrics_response.model_dump()
    # Create Pandas DataFrame
    df = pd.DataFrame({
        "datetimes": np.array(metrics_response['data'][0]['datetimes']),
        "blended_costs": np.array(metrics_response['data'][0]['blended_costs']),
    })
    return df

def query_instance_ids(supabase):
    if 'instance_ids' not in st.session_state:
        query = supabase.from_('tata_aws_ec2').select('instanceid')
        response = query.execute()
        response = response.model_dump()
        instance_ids = [row['instanceid'] for row in response['data']]
        ####################################################################################
        instance_ids = random.sample(instance_ids, k=3)
        ####################################################################################
        st.session_state['instance_ids'] = instance_ids
    else:
        instance_ids = st.session_state['instance_ids']

    return instance_ids

def create_instance_dataframe(instance_id, supabase):
    # Fetch data for the selected instance
    metrics_query = supabase.from_('tata_metrics').select('*').eq('instanceid', instance_id)
    metrics_response = metrics_query.execute()
    metrics_response = metrics_response.model_dump()

    # Create a DataFrame for instance
    df = pd.DataFrame(metrics_response["data"])
    # Create a new column for ChatGPT verdicts
    df['ChatGPT_Verdict'] = df.apply(
        lambda row: get_verdict(row['cpu_utilizations']),
        axis=1
    )

    # Create a new column for LED-like indicators
    df['Indicators'] = df['ChatGPT_Verdict'].apply(
        lambda verdict: '<p style="color: green; font-size: 24px">&#11044; Used</p>' if verdict.verdict == 'used'
        else '<p style="color: red; font-size: 24px">&#11044; Unused</p>'
    )

    return df[['instanceid', 'ChatGPT_Verdict', 'Indicators']]

def get_verdict(cpu_utilization_timeseries):

    class verdict:
        verdict = random.choice(['used','unused'])
        comment = 'no comments from chatGPT'

    return verdict

# Function to interact with ChatGPT using the LangChain prompt
def get_verdict_GPT(cpu_utilization_timeseries):
    avg = mean(cpu_utilization_timeseries)
    maxx = max(cpu_utilization_timeseries)
    vari = variance(cpu_utilization_timeseries)
    mode_val = mode(cpu_utilization_timeseries)
    cpu_utilization_timeseries = cpu_utilization_timeseries[:3:len(cpu_utilization_timeseries)]
    # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=Verdict)
    prompt = PromptTemplate(
        template='Give me a verdict between “used” and “unused”.'
        'I’m using an AWS EC2 instance. '
        'Based on the CPU utilization give me a verdict on whether it is being used or not. '
        'Evaluate the average cpu utilization, peak cpu utilization, trends, recent utilization before giving your verdict. '
        'In addition to the statistical metrics, consider also the raw timeseries to check if in the last period the instance has been used less, hence is not used anymore. '
        'Do NOT rate every instance as being "used". You have to be balanced in the verdicts and as fair as possible. '
        'Do NOT rate every instance as being "used". You have to be balanced in the verdicts and as fair as possible. '
        'CPU utilization below 20% is considered low. '
        'You are a professional informatics engineer, specialized in AWS management. '
        'Consider that low values of cpu utilization, even if not zero, could mean that the EC2 instance is not being used. '
        'Respond only with the verdict please. '
        'Do NOT write anything else in the response, except for the verdict keyword. '
        'Response example: "used" '
        'Response example: "unused" '
        '\n{format_instructions}\n'
        'The CPU utilization statistical data in the last 30 days are: average: {average}, mode value: {mode} variance: {variance}, maximum: {maximum}'
        'The CPU utilization in the last 30 days, with 30-minute time intervals, was {timeseries}.',
        input_variables=["timeseries","average","maximum","variance","mode"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    # Make a request to ChatGPT using LangChain prompt
    model = ChatOpenAI(temperature=0.3, model="gpt-4")    # gpt-3.5-turbo-16k
    
    # And a query intended to prompt a language model to populate the data structure.
    prompt_and_model = prompt | model
    output = prompt_and_model.invoke({"timeseries": cpu_utilization_timeseries, "average":avg, "maximum":maxx, "variance":vari, "mode":mode_val})
    verdict = parser.invoke(output)


    print('------------')
    print('------------')
    print(verdict)
    print('------------')
    print('------------')

    return verdict

def display_leds(instanceid,indicator,comment):
    # Create two columns
    col1, col2 = st.columns(2)
    with col1:
        # Make instance id clickable and redirect to tab2 with the selected instance
        if st.button(instanceid):
            st.session_state['instance_id_selected'] = instanceid
            switch_page("Metrics Viewer")

    with col2:
        # Create LED-like indicators for ChatGPT verdicts
        st.markdown(indicator, unsafe_allow_html=True)

    st.write(comment)
    # # Draw separetor line
    # st.markdown("""---""")

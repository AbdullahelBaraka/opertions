import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import io

# Set page configuration
st.set_page_config(
    page_title="Surgical Operations Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and header
st.title("📊 Surgical Operations Dashboard")
st.markdown("---")

# Initialize session state for data
if 'surgical_data' not in st.session_state:
    st.session_state.surgical_data = None

# File upload section
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload surgical data file",
    type=['xlsx', 'xls', 'csv'],
    help="Upload Excel or CSV file containing surgical operations data"
)

def load_data(file):
    """Load and process the uploaded data file"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Basic data cleaning and standardization
        df.columns = df.columns.str.strip()
        
        # Convert date columns if they exist
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'Date' in col]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def build_surgeon_department_database(df):
    """Build comprehensive surgeon-department mapping from all operation columns"""
    surgeon_dept_map = {}
    
    # Process all operations (1, 2, 3) to collect surgeon-department pairs
    for idx, row in df.iterrows():
        for i in range(1, 4):  # Operations 1, 2, 3
            if i == 1:
                dept = row.get('Department')
                consultant = row.get('Consultant Name')
                main_surgeon = row.get('Main Surgeon')
                assistant = row.get('Assistant Surgeon')
            elif i == 2:
                dept = row.get('Department.1')
                consultant = row.get('Consultant Name.1')
                main_surgeon = row.get('Main Surgeon.1')
                assistant = row.get('Assistant Surgeon.1')
            else:
                dept = row.get('Department.2')
                consultant = row.get('Consultant Name.2')
                main_surgeon = row.get('Main Surgeon.2')
                assistant = row.get('Assistant Surgeon.2')
            
            # Only process if department is valid
            if pd.notna(dept) and str(dept).strip() and str(dept).strip() != 'Unknown':
                dept_clean = str(dept).strip()
                
                # Map consultant to department
                if pd.notna(consultant) and str(consultant).strip() and str(consultant).strip() != 'Unknown':
                    surgeon_dept_map[str(consultant).strip()] = dept_clean
                
                # Map main surgeon to department
                if pd.notna(main_surgeon) and str(main_surgeon).strip() and str(main_surgeon).strip() != 'Unknown':
                    surgeon_dept_map[str(main_surgeon).strip()] = dept_clean
                
                # Map assistant to department
                if pd.notna(assistant) and str(assistant).strip() and str(assistant).strip() != 'Unknown':
                    surgeon_dept_map[str(assistant).strip()] = dept_clean
    
    return surgeon_dept_map

def extract_operations_data(df):
    """Extract and structure operations data from the uploaded file"""
    # First build the comprehensive surgeon-department database
    global_surgeon_dept_map = build_surgeon_department_database(df)
    
    operations_list = []
    
    for idx, row in df.iterrows():
        # Handle the actual structure from the data file
        # Extract up to 3 operations per patient
        for i in range(1, 4):  # Operations 1, 2, 3
            if i == 1:
                # Operation 1
                op_name = row.get('Operation 1 Name')
                price = row.get('price')
                dept = row.get('Department')
                consultant = row.get('Consultant Name')
                main_surgeon = row.get('Main Surgeon')
                assistant = row.get('Assistant Surgeon')
                pre_auth = row.get('Pre-Authorization')
                code = row.get('Code')
            elif i == 2:
                # Operation 2
                op_name = row.get('Operation 2 Name')
                price = row.get('price.1')
                dept = row.get('Department.1')
                consultant = row.get('Consultant Name.1')
                main_surgeon = row.get('Main Surgeon.1')
                assistant = row.get('Assistant Surgeon.1')
                pre_auth = row.get('Pre-Authorization.1')
                code = row.get('Code.1')
            else:
                # Operation 3
                op_name = row.get('Operation 3 Name')
                price = row.get('price.2')
                dept = row.get('Department.2')
                consultant = row.get('Consultant Name.2')
                main_surgeon = row.get('Main Surgeon.2')
                assistant = row.get('Assistant Surgeon.2')
                pre_auth = row.get('Pre-Authorization.2')
                code = row.get('Code.2')
            
            # Only add operation if it exists
            if pd.notna(op_name) and str(op_name).strip():
                # Use global mapping if local department is missing
                final_dept = str(dept) if pd.notna(dept) and str(dept).strip() else 'Unknown'
                if final_dept == 'Unknown' and pd.notna(main_surgeon) and str(main_surgeon).strip() in global_surgeon_dept_map:
                    final_dept = global_surgeon_dept_map[str(main_surgeon).strip()]
                
                operations_list.append({
                    'patient_id': row.get('Patient ID', idx),
                    'patient_name': row.get('Patient Name', 'Unknown'),
                    'gender': row.get('Gender', 'Unknown'),
                    'operation_name': str(op_name).strip(),
                    'department': final_dept,
                    'main_surgeon': str(main_surgeon) if pd.notna(main_surgeon) else 'Unknown',
                    'consultant': str(consultant) if pd.notna(consultant) else 'Unknown',
                    'assistant': str(assistant) if pd.notna(assistant) else 'Unknown',
                    'anesthesiologist': str(row.get('Anesthesiologist Name', 'Unknown')),
                    'diagnosis': str(row.get('Diagnosis', 'Unknown')),
                    'date': pd.to_datetime(row.get('Admission Date'), errors='coerce'),
                    'price': pd.to_numeric(price, errors='coerce') if pd.notna(price) else 0.0,
                    'pre_auth': str(pre_auth) if pd.notna(pre_auth) else 'Unknown',
                    'code': str(code) if pd.notna(code) else 'Unknown',
                    'source': str(row.get('Source', 'Unknown')),
                    'operation_number': i
                })
    
    return pd.DataFrame(operations_list)

def calculate_adjusted_operation_price(row):
    """Calculate adjusted price based on operation sequence"""
    if row['operation_number'] == 1:
        return row['price']  # Full price for first operation
    else:
        return row['price'] * 0.5  # Half price for subsequent operations

def calculate_surgeon_revenue_split(row):
    """Calculate revenue split for each surgeon based on roles and count"""
    # Get all surgeons involved in this operation
    surgeons = []
    if pd.notna(row['consultant']) and str(row['consultant']).strip() != 'Unknown':
        surgeons.append(('consultant', str(row['consultant']).strip()))
    if pd.notna(row['main_surgeon']) and str(row['main_surgeon']).strip() != 'Unknown' and str(row['main_surgeon']).strip() != str(row['consultant']).strip():
        surgeons.append(('main_surgeon', str(row['main_surgeon']).strip()))
    if pd.notna(row['assistant']) and str(row['assistant']).strip() != 'Unknown':
        surgeons.append(('assistant', str(row['assistant']).strip()))
    
    # Calculate adjusted operation price
    adjusted_price = calculate_adjusted_operation_price(row)
    
    # Determine revenue split based on number of surgeons and roles
    surgeon_revenues = []
    
    if len(surgeons) == 1:
        # Only one surgeon gets 100%
        surgeon_revenues.append((surgeons[0][1], adjusted_price))
    elif len(surgeons) == 2:
        # Check if one is consultant
        consultant_present = any(role == 'consultant' for role, name in surgeons)
        if consultant_present:
            for role, name in surgeons:
                if role == 'consultant':
                    surgeon_revenues.append((name, adjusted_price * 0.6))  # 60%
                else:
                    surgeon_revenues.append((name, adjusted_price * 0.4))  # 40%
        else:
            # Equal split if no consultant
            for role, name in surgeons:
                surgeon_revenues.append((name, adjusted_price * 0.5))  # 50% each
    elif len(surgeons) == 3:
        # Check if consultant is present
        consultant_present = any(role == 'consultant' for role, name in surgeons)
        if consultant_present:
            for role, name in surgeons:
                if role == 'consultant':
                    surgeon_revenues.append((name, adjusted_price * 0.4))  # 40%
                else:
                    surgeon_revenues.append((name, adjusted_price * 0.3))  # 30% each
        else:
            # Equal split if no consultant
            for role, name in surgeons:
                surgeon_revenues.append((name, adjusted_price * 0.333))  # ~33% each
    
    return surgeon_revenues

def get_comprehensive_surgeon_dept_mapping(df):
    """Get comprehensive surgeon-department mapping from the processed dataframe"""
    # Since the data is already processed, we can use a simpler approach
    # But we'll still clean and use the most frequent department for each surgeon
    df_clean = df.copy()
    df_clean['department'] = df_clean['department'].fillna('Unknown')
    df_clean['department'] = df_clean['department'].replace('', 'Unknown')
    
    # Filter out 'Unknown' departments for better mapping
    df_valid_dept = df_clean[df_clean['department'] != 'Unknown']
    
    if not df_valid_dept.empty:
        # Get most frequent department for each surgeon
        surgeon_dept_map = df_valid_dept.groupby('main_surgeon')['department'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
        ).to_dict()
        
        # Also map consultants and assistants
        consultant_dept_map = df_valid_dept.groupby('consultant')['department'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
        ).to_dict()
        
        assistant_dept_map = df_valid_dept.groupby('assistant')['department'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
        ).to_dict()
        
        # Combine all mappings
        all_surgeon_dept_map = {**surgeon_dept_map, **consultant_dept_map, **assistant_dept_map}
        return all_surgeon_dept_map
    
    return {}

def calculate_comprehensive_revenue_metrics(df):
    """Calculate comprehensive revenue metrics with new logic"""
    if df.empty:
        return {}, {}
    
    # Calculate total adjusted revenue (operation sequence-based)
    df = df.copy()
    df['adjusted_price'] = df.apply(calculate_adjusted_operation_price, axis=1)
    total_adjusted_revenue = df['adjusted_price'].sum()
    
    # Calculate surgeon revenues with splits
    surgeon_revenue_dict = {}
    
    for _, row in df.iterrows():
        surgeon_splits = calculate_surgeon_revenue_split(row)
        for surgeon_name, revenue in surgeon_splits:
            if surgeon_name in surgeon_revenue_dict:
                surgeon_revenue_dict[surgeon_name] += revenue
            else:
                surgeon_revenue_dict[surgeon_name] = revenue
    
    # Convert to Series for compatibility
    surgeon_revenue_series = pd.Series(surgeon_revenue_dict)
    
    revenue_metrics = {
        'total_adjusted_revenue': total_adjusted_revenue,
        'surgeon_revenues': surgeon_revenue_series,
        'avg_adjusted_price_per_operation': total_adjusted_revenue / len(df) if len(df) > 0 else 0
    }
    
    return revenue_metrics, df

def calculate_kpis(df):
    """Calculate key performance indicators"""
    if df.empty:
        return {}
    
    # Calculate comprehensive revenue metrics
    revenue_metrics, df_with_adjusted = calculate_comprehensive_revenue_metrics(df)
    
    total_surgeries = len(df)
    unique_patients = df['patient_id'].nunique()
    total_revenue = df['price'].sum()
    avg_ops_per_patient = total_surgeries / unique_patients if unique_patients > 0 else 0
    
    # Multi-operation cases
    patient_op_counts = df.groupby('patient_id').size()
    multi_op_patients = (patient_op_counts > 1).sum()
    multi_op_percentage = (multi_op_patients / unique_patients * 100) if unique_patients > 0 else 0
    
    avg_revenue_per_op = total_revenue / total_surgeries if total_surgeries > 0 else 0
    
    return {
        'total_surgeries': total_surgeries,
        'unique_patients': unique_patients,
        'total_revenue': revenue_metrics['total_adjusted_revenue'],  # Use adjusted revenue
        'avg_ops_per_patient': avg_ops_per_patient,
        'multi_op_percentage': multi_op_percentage,
        'avg_revenue_per_op': revenue_metrics['avg_adjusted_price_per_operation'],  # Use adjusted average
        'original_total_revenue': total_revenue,  # Keep original for reference
        'surgeon_revenues': revenue_metrics['surgeon_revenues']
    }

def create_temporal_trends(df, comparison_period='Monthly'):
    """Create temporal trend visualizations with comparison periods"""
    if df.empty:
        return None, None, None
    
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Calculate adjusted prices for all operations
    df = df.copy()
    df['adjusted_price'] = df.apply(calculate_adjusted_operation_price, axis=1)
    
    # Group by selected comparison period
    if comparison_period == 'Daily':
        df['period'] = df['date'].dt.day
        df['period_str'] = df['period'].astype(str)
        temporal_data = df.groupby(['period', 'period_str']).agg({
            'patient_id': 'count',  # operations count
            'adjusted_price': 'sum'  # Use adjusted price
        }).reset_index()
        title_suffix = 'by Day'
        period_label = 'Day'
    elif comparison_period == 'Quarterly':
        df['period'] = df['date'].dt.to_period('Q')
        temporal_data = df.groupby('period').agg({
            'patient_id': 'count',  # operations count
            'adjusted_price': 'sum'  # Use adjusted price
        }).reset_index()
        temporal_data['period_str'] = temporal_data['period'].astype(str)
        title_suffix = 'by Quarter'
        period_label = 'Quarter'
    elif comparison_period == 'Half-Annually':
        df['period'] = df['date'].dt.to_period('6M')
        temporal_data = df.groupby('period').agg({
            'patient_id': 'count',  # operations count
            'adjusted_price': 'sum'  # Use adjusted price
        }).reset_index()
        temporal_data['period_str'] = temporal_data['period'].astype(str)
        title_suffix = 'by Half-Year'
        period_label = 'Half-Year'
    elif comparison_period == 'Annually':
        df['period'] = df['date'].dt.year
        df['period_str'] = df['period'].astype(str)
        temporal_data = df.groupby(['period', 'period_str']).agg({
            'patient_id': 'count',  # operations count
            'adjusted_price': 'sum'  # Use adjusted price
        }).reset_index()
        title_suffix = 'by Year'
        period_label = 'Year'
    else:  # Monthly
        df['year_month'] = df['date'].dt.to_period('M')
        temporal_data = df.groupby('year_month').agg({
            'patient_id': 'count',  # operations count
            'adjusted_price': 'sum'  # Use adjusted price
        }).reset_index()
        # Convert to month names
        temporal_data['period_str'] = temporal_data['year_month'].dt.strftime('%B %Y')
        title_suffix = 'by Month'
        period_label = 'Month'
    
    # Operations trend line chart
    fig_ops = px.line(
        temporal_data, 
        x='period_str', 
        y='patient_id',
        title=f'📈 Total Operations {title_suffix}',
        labels={'patient_id': 'Number of Operations', 'period_str': period_label}
    )
    fig_ops.update_layout(xaxis_tickangle=-45)
    
    # Revenue trend line chart (changed from bar to line)
    fig_revenue = px.line(
        temporal_data,
        x='period_str',
        y='adjusted_price',
        title=f'💰 Revenue {title_suffix}',
        labels={'adjusted_price': 'Revenue ($)', 'period_str': period_label}
    )
    fig_revenue.update_layout(xaxis_tickangle=-45)
    
    return fig_ops, fig_revenue, temporal_data

def create_department_analytics(df):
    """Create department analytics"""
    if df.empty:
        return None
    
    # Department distribution pie chart
    dept_counts = df['department'].value_counts()
    fig_dept_pie = px.pie(
        values=dept_counts.values,
        names=dept_counts.index,
        title='🧩 Operations by Department'
    )
    
    return fig_dept_pie

def create_surgeon_analytics(df):
    """Create surgeon performance analytics"""
    if df.empty:
        return None, None
    
    # Top surgeons by operations
    surgeon_counts = df['main_surgeon'].value_counts().head(10)
    fig_surgeons = px.bar(
        x=surgeon_counts.values,
        y=surgeon_counts.index,
        orientation='h',
        title='🧑‍⚕️ Top Main Surgeons by Operations',
        labels={'x': 'Number of Operations', 'y': 'Surgeon'}
    )
    
    # Stacked bar chart for role distribution by department
    role_data = []
    for dept in df['department'].unique():
        dept_data = df[df['department'] == dept]
        consultants = dept_data['consultant'].value_counts().sum()
        main_surgeons = dept_data['main_surgeon'].value_counts().sum()
        assistants = dept_data['assistant'].value_counts().sum()
        
        role_data.append({
            'Department': dept,
            'Consultants': consultants,
            'Main Surgeons': main_surgeons,
            'Assistants': assistants
        })
    
    role_df = pd.DataFrame(role_data)
    fig_roles = px.bar(
        role_df,
        x='Department',
        y=['Consultants', 'Main Surgeons', 'Assistants'],
        title='🧮 Surgery Team Roles by Department',
        labels={'value': 'Number of Operations', 'variable': 'Role'}
    )
    fig_roles.update_layout(xaxis_tickangle=-45)
    
    return fig_surgeons, fig_roles

def create_weekday_operations_chart(df):
    """Create weekday operations bar chart"""
    if df.empty:
        return None, None
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Group by day of week and count operations
    df['weekday'] = df['date'].dt.day_name()
    weekday_ops = df['weekday'].value_counts()
    
    # Order by weekday
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_ops = weekday_ops.reindex([day for day in weekday_order if day in weekday_ops.index])
    
    # Create bar chart
    fig_weekday = px.bar(
        x=weekday_ops.index,
        y=weekday_ops.values,
        title='🌞 Operations by Day of Week',
        labels={'x': 'Day of Week', 'y': 'Number of Operations'}
    )
    
    # Create data table
    weekday_data = weekday_ops.reset_index()
    weekday_data.columns = ['Day of Week', 'Operations']
    
    return fig_weekday, weekday_data

def create_data_quality_panel(df):
    """Create data quality completeness panel"""
    if df.empty:
        return {}
    
    total_records = len(df)
    quality_metrics = {}
    
    # Check completeness for key fields
    key_fields = ['operation_name', 'department', 'main_surgeon', 'diagnosis', 'price']
    
    for field in key_fields:
        if field in df.columns:
            missing_count = df[field].isna().sum() + (df[field] == '').sum() + (df[field] == 'Unknown').sum()
            completeness = ((total_records - missing_count) / total_records * 100) if total_records > 0 else 0
            quality_metrics[field] = completeness
    
    return quality_metrics

# Load data if file is uploaded
if uploaded_file is not None:
    if st.session_state.surgical_data is None or st.sidebar.button("Reload Data"):
        with st.spinner("Loading and processing data..."):
            raw_data = load_data(uploaded_file)
            if raw_data is not None:
                st.session_state.surgical_data = extract_operations_data(raw_data)
                st.success(f"✅ Data loaded successfully! {len(st.session_state.surgical_data)} operations found.")

# Main dashboard content
if st.session_state.surgical_data is not None and not st.session_state.surgical_data.empty:
    df = st.session_state.surgical_data
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Date range filter with separate start and end dates
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        if pd.notna(min_date) and pd.notna(max_date):
            st.sidebar.subheader("📅 Date Range")
            start_date = st.sidebar.date_input(
                "Start Date",
                value=min_date.date(),
                min_value=min_date.date(),
                max_value=max_date.date()
            )
            end_date = st.sidebar.date_input(
                "End Date", 
                value=max_date.date(),
                min_value=min_date.date(),
                max_value=max_date.date()
            )
            
            # Apply date filter
            if start_date and end_date:
                df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
    
    # Department filter
    departments = ['All'] + sorted([str(x) for x in df['department'].unique() if pd.notna(x)])
    selected_dept = st.sidebar.selectbox("🏥 Department", departments)
    if selected_dept != 'All':
        df = df[df['department'] == selected_dept]
    
    # Surgeon filter
    surgeons = ['All'] + sorted([str(x) for x in df['main_surgeon'].unique() if pd.notna(x)])
    selected_surgeon = st.sidebar.selectbox("🧑‍⚕️ Main Surgeon", surgeons)
    if selected_surgeon != 'All':
        df = df[df['main_surgeon'] == selected_surgeon]
    
    # Operation name filter
    operations = ['All'] + sorted([str(x) for x in df['operation_name'].unique() if pd.notna(x)])
    selected_operation = st.sidebar.selectbox("🔍 Operation", operations)
    if selected_operation != 'All':
        df = df[df['operation_name'] == selected_operation]
    
    # Pre-authorization filter
    pre_auth_options = ['All'] + sorted([str(x) for x in df['pre_auth'].unique() if pd.notna(x)])
    selected_pre_auth = st.sidebar.selectbox("✔️ Pre Authorization", pre_auth_options)
    if selected_pre_auth != 'All':
        df = df[df['pre_auth'] == selected_pre_auth]
    
    # Comparison period selector
    st.sidebar.header("📊 Comparison Settings")
    comparison_period = st.sidebar.selectbox(
        "📈 Comparison Period",
        ['Daily', 'Monthly', 'Quarterly', 'Half-Annually', 'Annually'],
        index=1  # Default to Monthly
    )
    
    # 1. Overview KPIs
    st.header("📊 Overview KPIs")
    st.markdown("**Key performance indicators providing a comprehensive overview of surgical operations.**")
    
    kpis = calculate_kpis(df)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Surgeries", f"{kpis.get('total_surgeries', 0):,}")
        st.metric("Multi-Operation Cases", f"{kpis.get('multi_op_percentage', 0):.1f}%")
    
    with col2:
        st.metric("Unique Patients", f"{kpis.get('unique_patients', 0):,}")
        st.metric("Avg Operations/Patient", f"{kpis.get('avg_ops_per_patient', 0):.1f}")
    
    with col3:
        st.metric("Total Revenue", f"${kpis.get('total_revenue', 0):,.2f}")
        st.metric("Avg Revenue/Operation", f"${kpis.get('avg_revenue_per_op', 0):,.2f}")
    
    st.markdown("---")
    
    # 2. Operations Trend Analysis
    st.header("📈 Operations Trend Analysis")
    st.markdown("**Temporal analysis showing how operation volume and revenue change over time periods.**")
    
    fig_ops_trend, fig_revenue_trend, temporal_data = create_temporal_trends(df, comparison_period)
    
    if fig_ops_trend and fig_revenue_trend:
        # Operations Trend
        st.subheader("📊 Operations Volume Trend")
        st.markdown(f"**Description:** Number of surgical operations performed {comparison_period.lower()} showing patterns and trends over the selected date range.")
        st.plotly_chart(fig_ops_trend, use_container_width=True)
        
        # Operations data table with change percentage
        ops_data = temporal_data[['period_str', 'patient_id']].copy()
        ops_data.columns = ['Period', 'Operations']
        
        # Calculate change percentage
        ops_data['Change (%)'] = ops_data['Operations'].pct_change() * 100
        ops_data['Change (%)'] = ops_data['Change (%)'].round(2)
        ops_data['Change (%)'] = ops_data['Change (%)'].fillna(0).astype(str)
        ops_data.loc[0, 'Change (%)'] = '-'
        
        st.dataframe(ops_data, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Revenue Trend
        st.subheader("💰 Revenue Trend")
        st.markdown(f"**Description:** Total revenue generated {comparison_period.lower()} using comprehensive calculation: first operation full price, subsequent operations half price, distributed according to surgeon roles.")
        st.plotly_chart(fig_revenue_trend, use_container_width=True)
        
        # Revenue data table with change percentage
        revenue_data = temporal_data[['period_str', 'adjusted_price']].copy()
        revenue_data.columns = ['Period', 'Revenue ($)']
        revenue_data['Revenue ($)'] = revenue_data['Revenue ($)'].round(2)
        
        # Calculate change percentage
        revenue_data['Change (%)'] = revenue_data['Revenue ($)'].pct_change() * 100
        revenue_data['Change (%)'] = revenue_data['Change (%)'].round(2)
        revenue_data['Change (%)'] = revenue_data['Change (%)'].fillna(0).astype(str)
        revenue_data.loc[0, 'Change (%)'] = '-'
        
        st.dataframe(revenue_data, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Weekday Operations Analysis
    st.subheader("🌞 Operations by Day of Week")
    st.markdown("**Description:** Distribution of operations across weekdays showing which days are busiest for surgical procedures.")
    
    fig_weekday, weekday_data = create_weekday_operations_chart(df)
    if fig_weekday:
        st.plotly_chart(fig_weekday, use_container_width=True)
        
        # Add percentage column to weekday data
        total_ops = weekday_data['Operations'].sum()
        weekday_data['Percentage (%)'] = (weekday_data['Operations'] / total_ops * 100).round(2)
        st.dataframe(weekday_data, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # 3. Department Analytics
    st.header("🏥 Department Analytics")
    st.markdown("**Analysis of surgical operations distributed across different hospital departments.**")
    
    fig_dept_pie = create_department_analytics(df)
    
    if fig_dept_pie:
        # Department Distribution
        st.subheader("📊 Department Distribution")
        st.markdown("**Description:** Distribution of operations across departments showing which departments handle the most procedures.")
        st.plotly_chart(fig_dept_pie, use_container_width=True)
        
        # Department data table with percentage
        dept_counts = df['department'].value_counts().reset_index()
        dept_counts.columns = ['Department', 'Operations']
        
        # Add percentage column
        total_ops = dept_counts['Operations'].sum()
        dept_counts['Percentage (%)'] = (dept_counts['Operations'] / total_ops * 100).round(2)
        
        st.dataframe(dept_counts, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Department trends by comparison period
        if 'date' in df.columns:
            st.subheader("📈 Department Trends")
            st.markdown(f"**Description:** {comparison_period} breakdown of operations by department within the selected date range.")
            
            # Create period grouping based on comparison_period
            df = df.copy()
            if comparison_period == 'Daily':
                df['period'] = df['date'].dt.day.astype(str)
                period_label = 'Day'
            elif comparison_period == 'Quarterly':
                df['period'] = df['date'].dt.to_period('Q').astype(str)
                period_label = 'Quarter'
            elif comparison_period == 'Half-Annually':
                df['period'] = df['date'].dt.to_period('6M').astype(str)
                period_label = 'Half-Year'
            elif comparison_period == 'Annually':
                df['period'] = df['date'].dt.year.astype(str)
                period_label = 'Year'
            else:  # Monthly
                df['period'] = df['date'].dt.to_period('M').dt.strftime('%B %Y')
                period_label = 'Month'
            
            dept_trends = df.groupby(['period', 'department']).size().reset_index(name='operations')
            
            if not dept_trends.empty:
                fig_dept_trends = px.bar(
                    dept_trends,
                    x='period',
                    y='operations',
                    color='department',
                    title=f'📊 Operations by Department {comparison_period}',
                    labels={'operations': 'Number of Operations', 'period': period_label}
                )
                fig_dept_trends.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_dept_trends, use_container_width=True)
                
                # Add percentage column to department trends
                period_totals = dept_trends.groupby('period')['operations'].sum()
                dept_trends['Percentage (%)'] = dept_trends.apply(
                    lambda row: (row['operations'] / period_totals[row['period']] * 100).round(2), axis=1
                )
                st.dataframe(dept_trends, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # 4. Surgeon & Team Performance
    st.header("👨‍⚕️ Surgeon & Team Performance")
    st.markdown("**Analysis of surgeon productivity and team composition across departments.**")
    
    fig_surgeons, fig_roles = create_surgeon_analytics(df)
    
    if fig_surgeons:
        # Top Surgeons Analysis
        st.subheader("🏆 Top Performing Surgeons")
        st.markdown("**Description:** Top 10 main surgeons ranked by total number of operations performed, showing productivity levels.")
        st.plotly_chart(fig_surgeons, use_container_width=True)
        
        # Top surgeons data table with department percentage
        surgeon_counts = df['main_surgeon'].value_counts().head(10).reset_index()
        surgeon_counts.columns = ['Surgeon', 'Operations']
        
        # Add department and percentage columns using comprehensive mapping
        surgeon_dept_map = get_comprehensive_surgeon_dept_mapping(df)
        surgeon_counts['Department'] = surgeon_counts['Surgeon'].map(surgeon_dept_map).fillna('Unknown')
        
        # Calculate percentage of operations within each surgeon's department
        dept_totals = df.groupby('department')['main_surgeon'].count()
        surgeon_counts['Dept Percentage (%)'] = surgeon_counts.apply(
            lambda row: (row['Operations'] / dept_totals[row['Department']] * 100).round(2)
            if row['Department'] in dept_totals else 0, axis=1
        )
        
        st.dataframe(surgeon_counts, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Top Surgeons by Revenue (using new comprehensive calculation)
        st.subheader("💰 Top Main Surgeons by Revenue")
        st.markdown("**Description:** Top 10 main surgeons ranked by total revenue generated using comprehensive calculation (operation sequence pricing + surgeon role splits).")
        
        # Use the comprehensive revenue calculation
        revenue_metrics, _ = calculate_comprehensive_revenue_metrics(df)
        surgeon_revenue = revenue_metrics['surgeon_revenues'].sort_values(ascending=False).head(10)
        
        fig_surgeon_revenue = px.bar(
            x=surgeon_revenue.values,
            y=surgeon_revenue.index,
            orientation='h',
            title='💰 Top Main Surgeons by Revenue (Comprehensive)',
            labels={'x': 'Total Revenue ($)', 'y': 'Surgeon'}
        )
        st.plotly_chart(fig_surgeon_revenue, use_container_width=True)
        
        # Surgeon revenue data table with department percentage
        surgeon_revenue_data = surgeon_revenue.reset_index()
        surgeon_revenue_data.columns = ['Surgeon', 'Total Revenue ($)']
        surgeon_revenue_data['Total Revenue ($)'] = surgeon_revenue_data['Total Revenue ($)'].round(2)
        
        # Add department and percentage columns using comprehensive mapping
        surgeon_revenue_data['Department'] = surgeon_revenue_data['Surgeon'].map(surgeon_dept_map).fillna('Unknown')
        
        # Calculate percentage of revenue within each surgeon's department using adjusted totals
        dept_adjusted_revenue = {}
        for dept in df['department'].unique():
            if pd.notna(dept):
                dept_df = df[df['department'] == dept]
                dept_metrics, _ = calculate_comprehensive_revenue_metrics(dept_df)
                dept_adjusted_revenue[dept] = dept_metrics['total_adjusted_revenue']
        
        surgeon_revenue_data['Dept Percentage (%)'] = surgeon_revenue_data.apply(
            lambda row: (row['Total Revenue ($)'] / dept_adjusted_revenue[row['Department']] * 100).round(2)
            if row['Department'] in dept_adjusted_revenue and row['Department'] != 'Unknown' else 0, axis=1
        )
        
        st.dataframe(surgeon_revenue_data, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Team roles analysis
        if fig_roles:
            st.subheader("👥 Team Roles by Department")
            st.markdown("**Description:** Distribution of surgical team roles (consultants, main surgeons, assistants) across departments.")
            st.plotly_chart(fig_roles, use_container_width=True)
            
            # Team roles data table
            role_data = []
            for dept in df['department'].unique():
                dept_data = df[df['department'] == dept]
                consultants = dept_data['consultant'].value_counts().sum()
                main_surgeons = dept_data['main_surgeon'].value_counts().sum()
                assistants = dept_data['assistant'].value_counts().sum()
                
                role_data.append({
                    'Department': dept,
                    'Consultants': consultants,
                    'Main Surgeons': main_surgeons,
                    'Assistants': assistants
                })
            
            role_df = pd.DataFrame(role_data)
            
            # Add percentage columns for each role
            role_df['Total Roles'] = role_df['Consultants'] + role_df['Main Surgeons'] + role_df['Assistants']
            role_df['Consultants %'] = (role_df['Consultants'] / role_df['Total Roles'] * 100).round(2)
            role_df['Main Surgeons %'] = (role_df['Main Surgeons'] / role_df['Total Roles'] * 100).round(2)
            role_df['Assistants %'] = (role_df['Assistants'] / role_df['Total Roles'] * 100).round(2)
            
            st.dataframe(role_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
    
    # Surgeon performance table (using comprehensive revenue calculation)
    st.subheader("📊 Detailed Surgeon Performance")
    st.markdown("**Description:** Comprehensive surgeon statistics using advanced revenue calculation (operation sequence pricing + surgeon role splits).")
    
    # Get comprehensive revenue metrics
    revenue_metrics, _ = calculate_comprehensive_revenue_metrics(df)
    surgeon_revenues = revenue_metrics['surgeon_revenues']
    
    # Get all surgeons who have revenue (from all roles)
    all_surgeons_with_revenue = [s for s in surgeon_revenues.keys() if s != 'Unknown' and s.strip()]
    
    # Build comprehensive surgeon statistics
    surgeon_stats_list = []
    for surgeon in all_surgeons_with_revenue:
        # Count operations where this surgeon participated in any role
        surgeon_ops = df[
            (df['main_surgeon'] == surgeon) | 
            (df['consultant'] == surgeon) | 
            (df['assistant'] == surgeon)
        ]
        
        if len(surgeon_ops) > 0:
            total_operations = len(surgeon_ops)
            unique_patients = surgeon_ops['patient_id'].nunique()
            departments = ', '.join(surgeon_ops['department'].dropna().unique())
            total_revenue = surgeon_revenues.get(surgeon, 0.0)
            avg_revenue_per_op = total_revenue / total_operations if total_operations > 0 else 0.0
            
            surgeon_stats_list.append({
                'main_surgeon': surgeon,
                'Total Operations': total_operations,
                'Unique Patients': unique_patients,
                'Departments': departments if departments else 'Unknown',
                'Total Revenue': round(total_revenue, 2),
                'Avg Revenue per Op': round(avg_revenue_per_op, 2)
            })
    
    # Convert to DataFrame
    surgeon_stats = pd.DataFrame(surgeon_stats_list)
    
    if surgeon_stats.empty:
        st.warning("No surgeon performance data available.")
    else:
        # Sort by operations
        surgeon_stats = surgeon_stats.sort_values('Total Operations', ascending=False)
        
        # Add department-based percentage columns using comprehensive mapping
        surgeon_dept_map = get_comprehensive_surgeon_dept_mapping(df)
        surgeon_stats['Primary Department'] = surgeon_stats['main_surgeon'].map(surgeon_dept_map).fillna('Unknown')
        
        # Calculate department totals for percentages using comprehensive metrics
        dept_operations = df.groupby('department')['patient_id'].count()
        dept_patients = df.groupby('department')['patient_id'].nunique()
        dept_comprehensive_revenue = {}
        for dept in df['department'].unique():
            if pd.notna(dept):
                dept_df = df[df['department'] == dept]
                dept_metrics, _ = calculate_comprehensive_revenue_metrics(dept_df)
                dept_comprehensive_revenue[dept] = dept_metrics['total_adjusted_revenue']
        
        # Add percentage columns
        surgeon_stats['Operations % in Dept'] = surgeon_stats.apply(
            lambda row: (row['Total Operations'] / dept_operations[row['Primary Department']] * 100).round(2)
            if row['Primary Department'] in dept_operations else 0, axis=1
        )
        
        surgeon_stats['Patients % in Dept'] = surgeon_stats.apply(
            lambda row: (row['Unique Patients'] / dept_patients[row['Primary Department']] * 100).round(2)
            if row['Primary Department'] in dept_patients else 0, axis=1
        )
        
        surgeon_stats['Revenue % in Dept'] = surgeon_stats.apply(
            lambda row: (row['Total Revenue'] / dept_comprehensive_revenue[row['Primary Department']] * 100).round(2)
            if row['Primary Department'] in dept_comprehensive_revenue and row['Primary Department'] != 'Unknown' else 0, axis=1
        )
        
        st.dataframe(surgeon_stats, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # 5. Advanced Insights
    st.header("🔬 Advanced Insights")
    st.markdown("**Deep analysis revealing patterns in patient complexity and departmental revenue performance.**")
    
    # Operation complexity distribution
    st.subheader("🔍 Patient Operation Complexity")
    st.markdown("**Description:** Distribution showing how many operations each patient received within the selected date range.")
    
    patient_op_counts = df.groupby('patient_id').size()
    complexity_dist = patient_op_counts.value_counts().sort_index()
    
    # Create readable labels for operations
    operation_labels = []
    for count in complexity_dist.index:
        if count == 1:
            operation_labels.append("One Operation")
        elif count == 2:
            operation_labels.append("Two Operations")
        elif count == 3:
            operation_labels.append("Three Operations")
        else:
            operation_labels.append(f"{count} Operations")
    
    fig_complexity = px.bar(
        x=operation_labels,
        y=complexity_dist.values,
        title='🔎 Patient Operation Complexity Distribution',
        labels={'x': 'Operations per Patient', 'y': 'Number of Patients'}
    )
    st.plotly_chart(fig_complexity, use_container_width=True)
    
    # Complexity data table with percentage
    complexity_data = pd.DataFrame({
        'Operations per Patient': operation_labels,
        'Number of Patients': complexity_dist.values
    })
    
    # Add percentage column
    total_patients = complexity_data['Number of Patients'].sum()
    complexity_data['Percentage (%)'] = (complexity_data['Number of Patients'] / total_patients * 100).round(2)
    
    st.dataframe(complexity_data, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Average revenue per department
    st.subheader("💰 Department Revenue Analysis")
    st.markdown("**Description:** Average revenue per operation by department within the selected date range, calculated by taking the mean price of all operations.")
    
    dept_revenue = df.groupby('department')['price'].mean().sort_values(ascending=False)
    fig_dept_revenue = px.bar(
        x=dept_revenue.index,
        y=dept_revenue.values,
        title='💰 Average Revenue per Department',
        labels={'x': 'Department', 'y': 'Average Revenue ($)'}
    )
    fig_dept_revenue.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_dept_revenue, use_container_width=True)
    
    # Department revenue data table
    dept_revenue_data = dept_revenue.reset_index()
    dept_revenue_data.columns = ['Department', 'Average Revenue ($)']
    dept_revenue_data['Average Revenue ($)'] = dept_revenue_data['Average Revenue ($)'].round(2)
    st.dataframe(dept_revenue_data, use_container_width=True, hide_index=True)
    
    # Raw data view
    with st.expander("📊 View Raw Data"):
        st.dataframe(df, use_container_width=True)
        
        # Download filtered data
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name=f"surgical_data_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    # Welcome screen
    st.info("👆 Please upload a surgical operations data file using the sidebar to begin analysis.")
    
    st.markdown("""
    ### 📋 Expected Data Format
    
    Your file should contain surgical operations data with columns such as:
    
    **Required/Recommended Columns:**
    - `Patient ID` or `PatientID` - Unique identifier for patients
    - `Operation` or `Operation 1`, `Operation 2`, `Operation 3` - Surgery names
    - `Department` or `Dept` - Medical department
    - `Main Surgeon` or `Surgeon` - Primary surgeon name
    - `Date` or `Surgery Date` - Operation date
    - `Price`, `Cost`, or `Revenue` - Financial information
    
    **Optional Columns:**
    - `Consultant` - Consulting physician
    - `Assistant` - Assistant surgeon
    - `Anesthesiologist` - Anesthesia provider
    - `Diagnosis` - Medical diagnosis
    - `Pre Authorization` - Pre-auth status
    
    **Supported File Types:** Excel (.xlsx, .xls) and CSV (.csv)
    """)

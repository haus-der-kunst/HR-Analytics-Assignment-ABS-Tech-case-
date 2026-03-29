import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_excel('ABS Tech Case 2026_Data.xlsx')

# Select variables
df_shrink = df[['Employee.Name', 'PerfScore', 'SpecialProjectsCount', 'ProjSelf', 'ProjColl',
                'ProjLead', 'Feedback', 'TechLev', 'AIUse', 'AIConf', 'InnoCont', 'Trust', 'Network',
                'TeamIden','OrgIden', 'PsySafe', 'EmpSatisfaction', 'WLF', 'JobStr', 'CarOpp']].copy()

# Invert Job Stress (high value = better wellbeing)
df_shrink['JobStr_Inv'] = 6 - df_shrink['JobStr']

# Define metric groups
core_cols = ['PerfScore', 'SpecialProjectsCount', 'Feedback', 'ProjSelf', 'ProjColl', 'ProjLead']
tech_cols = ['TechLev', 'AIUse', 'AIConf', 'InnoCont']
impact_cols = ['Trust', 'Network', 'TeamIden', 'OrgIden']
well_cols = ['PsySafe', 'EmpSatisfaction', 'WLF', 'JobStr_Inv']
pot_cols = ['CarOpp']

# Scaling
scaler = MinMaxScaler()

all_metrics = list(set(core_cols + tech_cols + impact_cols + well_cols + pot_cols))

df_norm = df_shrink.copy()
df_norm[all_metrics] = scaler.fit_transform(df_norm[all_metrics])

# Calculate dimension scores using mean
df_norm['CoreContribution'] = df_norm[core_cols].mean(axis=1)
df_norm['TechReadiness'] = df_norm[tech_cols].mean(axis=1)
df_norm['OrgImpact'] = df_norm[impact_cols].mean(axis=1)
df_norm['Wellbeing'] = df_norm[well_cols].mean(axis=1)
df_norm['Potential'] = df_norm[pot_cols].mean(axis=1)

# Final Talent Score
df_norm['TalentScore'] = (
    0.3 * df_norm['CoreContribution'] +
    0.3 * df_norm['TechReadiness'] +
    0.2 * df_norm['OrgImpact'] +
    0.1 * df_norm['Wellbeing'] +
    0.1 * df_norm['Potential']
)

print(df_norm.info())

# Export dataframe
df_export = df_norm[[
    'Employee.Name',
    'CoreContribution',
    'TechReadiness',
    'OrgImpact',
    'Wellbeing',
    'Potential',
    'TalentScore'
]].copy()

# Add termination info
df_export['Termd'] = df['Termd']
df_export['TermReason'] = df['TermReason']
df_export['EmploymentStatus'] = df['EmploymentStatus']
df_export['Department'] = df['Department']  # Add Department for later use

# Streamlit
st.set_page_config(layout="centered")
st.title("Talent Score")

# Mapping Configuration
mapping = {
    'Employee.Name': 'Employee Name',    
    'CoreContribution': 'Core Contribution',
    'TechReadiness': 'Technological Readiness',
    'OrgImpact': 'Organizational Impact',
    'Wellbeing': 'Well-being',
    'Potential': 'Potential',
    'TalentScore': 'Total Talent Score',
    'TermReason': 'Reason for Termination',
    'LowestScore': 'Lowest Rated from Talent Score'
}

metrics = ['CoreContribution', 'TechReadiness', 'OrgImpact', 'Wellbeing', 'Potential', 'TalentScore']
columns_basic_en = [mapping[m] for m in ['Employee.Name'] + metrics]

st.markdown(f"""
<div style='background-color: var(--secondary-background-color); 
            padding: 15px; 
            border-radius: 8px; 
            border-left: 5px solid var(--primary-color);'>
    <strong>Note:</strong> The default view displays the <b>'Average Score'</b>. <br>
    Average values are indicated by the <b>●</b> markers and their corresponding values. <br>
    To view an individual employee's performance, please enter their name in the search box below.
</div>
""", unsafe_allow_html=True)

st.write("")

# Employee Search
search_name = st.text_input("Search Employee (First or Last Name)", "")

def match_name(full_name, query):
    if query == "": return True
    parts = [p.strip().lower() for p in full_name.split(",")]
    return any(query.lower() in part for part in parts)

matched_df = df_export[df_export['Employee.Name'].apply(lambda x: match_name(x, search_name))]

# Bar Chart
if search_name != "" and len(matched_df) > 0:
    selected_employee = matched_df.iloc[0]
    bar_values = selected_employee[metrics].values
    employee_label = selected_employee['Employee.Name']
    colors = ["lightgray"]*5 + ["#FFBF00"]
else:
    bar_values = df_export[metrics].mean().values
    employee_label = "Average Score"
    colors = ["#D1E7DD"]*5 + ["#28A745"] 

fig, ax = plt.subplots(figsize=(10, 6))
display_labels = [mapping[m] for m in metrics]
bars = ax.bar(display_labels, bar_values, color=colors)

ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title(f"Analysis of: '{employee_label}'")
plt.xticks(rotation=20, ha='right')

# Value labels on bars
for i, v in enumerate(bar_values):
    ax.text(i, v - 0.05, f"{v:.2f}", ha='center', color='black', fontsize=9)

# Average markers
mean_values = df_export[metrics].mean().values
for i, mean_val in enumerate(mean_values):
    ax.plot(i, mean_val, marker='o', color='#2E8B57', markersize=6)
    ax.text(i, mean_val + 0.02, f"{mean_val:.2f}", ha='center', color='green', fontweight='bold', fontsize=9)

st.pyplot(fig)

# Employee List & Tables
st.divider()
st.title("Employee List")

st.markdown(f"""
<div style='background-color: var(--secondary-background-color); 
            padding: 15px; 
            border-radius: 8px; 
            border-left: 5px solid var(--primary-color);
            color: var(--text-color);'>
    Above Average: (≥) Average Talent Score. <br> 
    Below Average: (<) Average Talent Score. <br>
    Employees in the list are ranked in <b>descending order</b> based on their <b>Talent Score</b>.
</div>
""", unsafe_allow_html=True)

st.write("")

avg_talent_score = df_export['TalentScore'].mean()
above_avg = df_export[df_export['TalentScore'] >= avg_talent_score]
below_avg = df_export[df_export['TalentScore'] < avg_talent_score]

def process_and_display(df, title, is_above=True):
    header = f"### '{'Above' if is_above else 'Below'} Average' {mapping['TalentScore']}"
    st.markdown(header)
    
    if df.empty:
        st.write("No employees found.")
        return

    # Basic List
    df_sorted = df.sort_values(by='TalentScore', ascending=False)
    display_list = [[row['Employee.Name']] + row[metrics].tolist() for _, row in df_sorted.iterrows()]
    st.dataframe(pd.DataFrame(display_list, columns=columns_basic_en))
    st.info(f"Found {len(df)} employees in this category.")

    # Voluntarily Terminated List
    terminated = df[df['EmploymentStatus'] == 'Voluntarily Terminated']
    if not terminated.empty:
        term_list = []
        for _, row in terminated.iterrows():
            core_scores = row[metrics[:-1]]
            min_idx = core_scores.idxmin()
            min_val = core_scores[min_idx]
            # Mapping the lowest score metric name
            mapped_min_idx = mapping.get(min_idx, min_idx)
            
            term_list.append([
                row['Employee.Name'],
                row['TermReason'],
                f"{mapped_min_idx} ({min_val:.2f})",
                row['TalentScore']
            ])
        
        term_df = pd.DataFrame(term_list, columns=[
            mapping['Employee.Name'], 
            mapping['TermReason'], 
            mapping['LowestScore'], 
            'TalentScore'
        ])
        term_df = term_df.sort_values(by='TalentScore', ascending=False).drop(columns='TalentScore')
        
        st.markdown(f"#### Voluntarily Terminated '{'Above' if is_above else 'Below'} Average' Employees")
        st.dataframe(term_df)
        st.warning(f"Found {len(term_df)} voluntarily terminated employees.")

# Render Tables
process_and_display(above_avg, "Above", is_above=True)
st.write("")
process_and_display(below_avg, "Below", is_above=False)

def terminated_dept_df(df, group_col='Talent'):
    records = []
    talent_map = {1: 'Talent', 0: 'Non-Talent'}
    
    for group_value, group_name in zip([1,0], ['Talent', 'Non-Talent']):
        group_df = df[(df[group_col]==group_value) & (df['EmploymentStatus']=='Voluntarily Terminated')].copy()
        for dept, count in group_df['Department'].value_counts().items():
            records.append({'Talent Status': group_name, 'Department': dept, 'Count': count})
    
    return pd.DataFrame(records)

# Prepare DataFrame
df_export['Talent'] = (df_export['TalentScore'] >= df_export['TalentScore'].mean()).astype(int)
terminated_df = terminated_dept_df(df_export, group_col='Talent')

st.divider()
st.subheader("Voluntarily Terminated by Department")

if not terminated_df.empty:
    fig, ax = plt.subplots(figsize=(10,6))
    
    sns.barplot(
        data=terminated_df,
        x='Department',
        y='Count',
        hue='Talent Status',
        palette={'Talent': '#FFBF00', 'Non-Talent': 'lightgray'},
        ax=ax
    )
    
    # Value labels on bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', fontsize=9, color='black', weight='bold')
    
    ax.set_ylabel("Number of Employees")
    ax.set_xlabel("Department")
    plt.xticks(rotation=45, ha='right')
    ax.legend(title='Talent Status')
    plt.tight_layout()
    
    st.pyplot(fig)
else:
    st.write("No voluntarily terminated employees found.")

st.divider()
st.subheader("Voluntarily Terminated by Termination Reason")

# Talent label
df_export['Talent Status'] = df_export['Talent'].map({1: 'Talent', 0: 'Non-Talent'})

# Voluntarily Terminated
vt_df = df_export[df_export['EmploymentStatus'] == 'Voluntarily Terminated']

# TermReason count
termreason_df = (
    vt_df.groupby(['Talent Status','TermReason'])
    .size()
    .reset_index(name='Count')
)

if not termreason_df.empty:
    
    fig, ax = plt.subplots(figsize=(10,6))

    sns.barplot(
        data=termreason_df,
        x='TermReason',
        y='Count',
        hue='Talent Status',
        hue_order=['Talent','Non-Talent'],
        palette={'Talent': '#FFBF00', 'Non-Talent': 'lightgray'},  
        ax=ax
    )

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f'{int(height)}',
                (p.get_x() + p.get_width()/2, height),
                ha='center',
                va='bottom',
                fontsize=9,
                color='black',
                weight='bold'
            )

    ax.set_ylabel("Number of Employees")
    ax.set_xlabel("Termination Reason")
    plt.xticks(rotation=45, ha='right')
    ax.legend(title='Talent Status')
    plt.tight_layout()

    st.pyplot(fig)

else:
    st.write("No voluntarily terminated employees found.")


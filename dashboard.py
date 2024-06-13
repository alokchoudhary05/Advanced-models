from flask import Blueprint, render_template, request
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
import io
import base64

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/', methods=['GET', 'POST'])
def dashboard():
    # Load the dataset
    df = pd.read_csv('Dataset/Original_Dataset.csv')
    df = shuffle(df, random_state=42)


    # Convert DataFrame to HTML for rendering
    data_html = df.head().to_html(classes='table table-striped', index=False)
    
    # Frequency of each disease
    disease_counts = df['Disease'].value_counts()
    fig1 = px.bar(disease_counts, x=disease_counts.index, y=disease_counts.values, color=disease_counts.index)
    fig1.update_layout(xaxis_title='Disease', yaxis_title='Count', xaxis={'categoryorder':'total descending'})
    graph1 = fig1.to_html(full_html=False)
    
    # Word Cloud of Symptoms
    symptom_columns = df.columns[1:]
    all_symptoms = df[symptom_columns].values.flatten()
    all_symptoms = [symptom for symptom in all_symptoms if pd.notna(symptom)]
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_symptoms))
    img = io.BytesIO()
    plt.figure(figsize=(8,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(img, format='png')
    img.seek(0)
    wordcloud_img = base64.b64encode(img.getvalue()).decode()

    # Frequency Plot of Top Symptoms
    symptom_list = df[symptom_columns].values.flatten()
    symptom_list = [symptom for symptom in symptom_list if pd.notna(symptom)]
    symptom_counts = pd.Series(symptom_list).value_counts().head(20)
    fig2 = px.bar(symptom_counts, x=symptom_counts.values, y=symptom_counts.index, orientation='h', color=symptom_counts.index)
    fig2.update_layout(xaxis_title='Count', yaxis_title='Symptom')
    graph2 = fig2.to_html(full_html=False)
    
    # Symptom Distribution per Disease
    diseases = df['Disease'].unique()
    symptom_counts_per_disease = {}
    selected_disease = request.form.get('disease', diseases[0])
    symptom_counts = df[df['Disease'] == selected_disease].iloc[:, 1:].notna().sum()
    fig3 = px.bar(symptom_counts.sort_values(), orientation='h', labels={'index': 'Symptoms', 'value': 'Count'}, title=f'Symptom Distribution for {selected_disease}', color=symptom_counts.index)
    graph3 = fig3.to_html(full_html=False)


    # Disease-Symptom Network Graph
    B = nx.Graph()
    diseases = df['Disease'].unique()
    symptoms = pd.melt(df, id_vars=['Disease'], value_vars=df.columns[1:]).dropna()['value'].unique()

    B.add_nodes_from(diseases, bipartite=0)
    B.add_nodes_from(symptoms, bipartite=1)

    edges = []
    for index, row in df.iterrows():
        for symptom in df.columns[1:]:
            if pd.notna(row[symptom]):
                edges.append((row['Disease'], row[symptom]))

    B.add_edges_from(edges)

    pos = nx.spring_layout(B)
    edge_x = []
    edge_y = []
    for edge in B.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    node_x = []
    node_y = []
    node_text = []
    for node in B.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition='top center', hoverinfo='text', marker=dict(color=[], size=10, line=dict(width=2)))
    node_trace.marker.color = ['blue' if node in diseases else 'red' for node in B.nodes()]

    fig4 = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(title='', titlefont_size=16, showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40),
                                     annotations=[ dict(text="Alok Choudhary", showarrow=False, xref="paper", yref="paper")],
                                     xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False)))
    graph4 = fig4.to_html(full_html=False)



    # Pie Chart of Top Symptoms
    symptom_counts = df.iloc[:, 1:].stack().value_counts().head(10)
    fig5 = px.pie(symptom_counts, values=symptom_counts.values, names=symptom_counts.index, title='')
    fig5 = fig5.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#000000', width=2)))
    graph5 = fig5.to_html(full_html=False)
    

    # Parallel Categories Diagram
    fig6 = px.parallel_categories(df, color_continuous_scale=px.colors.sequential.Inferno, title="")
    graph6 = fig6.to_html(full_html=False)







    # Visualization on Encoded Dataset
    df_v1 = pd.read_csv('Dataset/df_v1.csv')
    df_v1 = shuffle(df_v1, random_state=42)

    # Create a multiselect widget to choose symptoms for the 3D scatter plot
    fig7 = px.scatter_3d(df_v1, x='family_history', y='muscle_weakness', z='silver_like_dusting', color='Disease', title="")
    graph7 = fig7.to_html(full_html=False)



    # Flows Between Diseases and Symptoms (Sankey Diagram)
    nodes = list(df_v1['Disease'].unique()) + list(df_v1.columns[1:10])
    links = []
    for disease in df_v1['Disease'].unique():
        for symptom in df_v1.columns[1:10]:
            count = df_v1[(df_v1['Disease'] == disease) & (df_v1[symptom] == 1)].shape[0]
            if count > 0:
                links.append({'source': nodes.index(symptom), 'target': nodes.index(disease), 'value': count})

    fig8 = go.Figure(go.Sankey(node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=nodes ),
                               link=dict(source=[link['source'] for link in links], target=[link['target'] for link in links], value=[link['value'] for link in links])))
    fig8.update_layout(title_text="", font_size=10)
    graph8 = fig8.to_html(full_html=False)


    # Parallel Relationships Between Diseases and Symptoms
    fig9 = px.parallel_categories(df_v1, dimensions=['Disease', 'family_history', 'muscle_weakness', 'silver_like_dusting'],
                                  title="", color_continuous_scale='viridis')
    graph9 = fig9.to_html(full_html=False)



    # Bubble Chart of Disease Symptom Counts
    df_v1['symptom_count'] = df_v1.iloc[:, 1:].sum(axis=1)
    fig10 = px.scatter(df_v1, x='Disease', y='symptom_count', size='symptom_count', color='Disease')
    graph10 = fig10.to_html(full_html=False)


    # Create the graph
    G = nx.Graph()
    # Add nodes and edges
    diseases = df_v1['Disease'].unique()
    symptoms = df_v1.columns[1:]
    # Add disease nodes
    for disease in diseases:
        G.add_node(disease, type='disease')
    # Add symptom nodes and edges
    for symptom in symptoms:
        G.add_node(symptom, type='symptom')
        for disease in diseases:
            if df_v1[df_v1['Disease'] == disease][symptom].sum() > 0:
                G.add_edge(disease, symptom)
    # Define node positions using a spring layout
    pos = nx.spring_layout(G)
    # Create edge traces
    edge_trace = go.Scatter( x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)

    # Create node traces
    node_trace = go.Scatter( x=[], y=[], text=[], mode='markers+text', textposition='top center', hoverinfo='text',
        marker=dict( showscale=True, colorscale='YlGnBu', size=10, colorbar=dict( thickness=15, title='Node Connections', xanchor='left', titleside='right'), color=[]))

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += (node,)
        node_trace['marker']['color'] += (G.degree(node),)

    # Create the plot
    fig11 = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(title='', titlefont_size=16, showlegend=False, hovermode='closest', margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(text="Network graph of diseases and symptoms (by Alok Choudhary)", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    graph11 = fig11.to_html(full_html=False)




    return render_template('dashboard.html', data_html=data_html, graph1=graph1, wordcloud_img=wordcloud_img, graph2=graph2, 
                           symptom_counts_per_disease=symptom_counts_per_disease, diseases=diseases, graph3=graph3, selected_disease=selected_disease, graph4=graph4,
                           graph5=graph5, graph6=graph6, graph7=graph7, graph8=graph8, graph9=graph9, graph10=graph10, graph11=graph11)



if __name__ == "__main__":
    dashboard_bp.run(debug=True)
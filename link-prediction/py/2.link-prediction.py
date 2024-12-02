#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[1]:


import pandas as pd
import unidecode
from typing import Optional
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# In[2]:


DATA_DIR = '../data/'


# In[3]:


def most_common_ignore_na(series: pd.Series) -> Optional[str]:
    """
    Returns the most common value in a series, ignoring null values.
    If all values are null, returns None.

    Args:
        series (pd.Series): A pandas Series containing values for a specific attribute.

    Returns:
        Optional[str]: The most common value in the series, or None if all values are null.
    """
    mode = series.dropna().mode()  # Calculate the mode, ignoring nulls
    if not mode.empty:
        return mode[0]  # Return the most common value if it exists
    else:
        return None  # Return None if all values are null


# In[4]:


# standadize string function
standardize_string = lambda x: unidecode.unidecode(x.lower())


# ## Read Data

# In[5]:


def read_dataset(
    data_dir,
    scope,
    dataset
):
    
    data_path = f'{data_dir}processed/{scope}/{dataset}.csv'
    df = pd.read_csv(data_path)
    return df


# In[6]:


scope = 'restritivo'
scope = 'aplicacoes'

df_book_raw = read_dataset(data_dir=DATA_DIR, scope=scope, dataset='livros')
df_user = read_dataset(data_dir=DATA_DIR, scope=scope, dataset='gerais')
df_per = read_dataset(data_dir=DATA_DIR, scope=scope, dataset='periodicos')
df_cap = read_dataset(data_dir=DATA_DIR, scope=scope, dataset='capitulos')
df_eventos = read_dataset(data_dir=DATA_DIR, scope=scope, dataset='eventos')


# ## Describe Data

# In[7]:


# length of datasets
print(f'Shape "gerais": {df_user.shape}')
print(f'Shape "livros": {df_book_raw.shape}')
print(f'Shape "periodicos": {df_per.shape}')
print(f'Shape "capitulos": {df_cap.shape}')
print(f'Shape "eventos": {df_eventos.shape}')


# In[8]:


# columsn of datasets
print(f'Columns "gerais": {list(df_user.columns)}')
print(f'Columns "livros": {list(df_book_raw.columns)}')
print(f'Columns "periodicos": {list(df_per.columns)}')
print(f'Columns "capitulos": {list(df_cap.columns)}')
print(f'Columns "eventos": {list(df_eventos.columns)}')


# ### GERAIS

# In[9]:


df_user.head()


# ### LIVROS

# In[10]:


df_book_raw.head()


# In[11]:


print('### Book name Describe ###')
print(f'Number of rows: {len(df_book_raw['TITULO-DO-LIVRO'])}')
print(f'Number of rows (dropna): {len(df_book_raw['TITULO-DO-LIVRO'].dropna())}')
print(f'Number of unique books names: {len(df_book_raw['TITULO-DO-LIVRO'].unique())}')
print(f'Number of unique books names (lower): {len(df_book_raw['TITULO-DO-LIVRO'].str.lower().unique())}')
print(f'Number of unique books names (standardize): {len(df_book_raw['TITULO-DO-LIVRO'].apply(standardize_string).unique())}')

print('\n### ISBN Describe ###')
print(f'Number of rows: {len(df_book_raw['ISBN'])}')
print(f'Number of rows (dropna): {len(df_book_raw['ISBN'].dropna())}')
print(f'Number of unique books ISBN: {len(df_book_raw['ISBN'].dropna().unique())}')
print(f'Number of unique books ISBN (lower): {len(df_book_raw['ISBN'].dropna().str.lower().unique())}')
print(f'Number of unique books ISBN (standardize): {len(df_book_raw['ISBN'].dropna().apply(standardize_string).unique())}')


# ### PERIODICOS

# In[12]:


df_per.head()


# In[13]:


df_per[['TITULO-DO-ARTIGO', 'TITULO-DO-PERIODICO-OU-REVISTA']] = df_per[['TITULO-DO-ARTIGO', 'TITULO-DO-PERIODICO-OU-REVISTA']].astype('str')


# In[14]:


print('### Paper name Describe ###')
print(f'Number of rows: {len(df_per['TITULO-DO-ARTIGO'])}')
print(f'Number of rows (dropna): {len(df_per['TITULO-DO-ARTIGO'].dropna())}')
print(f'Number of unique paper names: {len(df_per['TITULO-DO-ARTIGO'].unique())}')
print(f'Number of unique paper names (lower): {len(df_per['TITULO-DO-ARTIGO'].str.lower().unique())}')
print(f'Number of unique paper names (standardize): {len(df_per['TITULO-DO-ARTIGO'].apply(standardize_string).unique())}')

print('\n### Paper name Describe ###')
print(f'Number of rows: {len(df_per['TITULO-DO-ARTIGO'])}')
print(f'Number of rows (dropna): {len(df_per[['TITULO-DO-ARTIGO', 'TITULO-DO-PERIODICO-OU-REVISTA']].dropna())}')
print(f'Number of unique paper + journal names: {len(df_per[['TITULO-DO-ARTIGO', 'TITULO-DO-PERIODICO-OU-REVISTA']].drop_duplicates())}')
print(f'Number of unique paper + journal names (lower): {len(df_per[['TITULO-DO-ARTIGO', 'TITULO-DO-PERIODICO-OU-REVISTA']].drop_duplicates().apply(lambda x: (x['TITULO-DO-ARTIGO'].lower(), x['TITULO-DO-PERIODICO-OU-REVISTA'].lower()), axis =1, result_type='expand'))}')
print(f'Number of unique paper + journal names (standardize): {len(df_per[['TITULO-DO-ARTIGO', 'TITULO-DO-PERIODICO-OU-REVISTA']].drop_duplicates().apply(lambda x: (standardize_string(x['TITULO-DO-ARTIGO']), standardize_string(x['TITULO-DO-PERIODICO-OU-REVISTA'])), axis =1, result_type='expand'))}')


print('\n### DOI Describe ###')
print(f'Number of rows: {len(df_per['DOI'])}')
print(f'Number of rows (dropna): {len(df_per['DOI'].dropna())}')
print(f'Number of unique paper DOI: {len(df_per['DOI'].dropna().unique())}')
print(f'Number of unique paper DOI (lower): {len(df_per['DOI'].dropna().str.lower().unique())}')
print(f'Number of unique paper DOI (standardize): {len(df_per['DOI'].dropna().apply(standardize_string).unique())}')


# ### CAPITULOS

# In[15]:


df_cap.head()


# In[16]:


df_cap['TITULO-DO-LIVRO'] = df_cap['TITULO-DO-LIVRO'].astype('str')


# In[17]:


print('### Book name Describe ###')
print(f'Number of rows: {len(df_cap['TITULO-DO-LIVRO'])}')
print(f'Number of rows (dropna): {len(df_cap['TITULO-DO-LIVRO'].dropna())}')
print(f'Number of unique books names: {len(df_cap['TITULO-DO-LIVRO'].unique())}')
print(f'Number of unique books names (lower): {len(df_cap['TITULO-DO-LIVRO'].str.lower().unique())}')
print(f'Number of unique books names (standardize): {len(df_cap['TITULO-DO-LIVRO'].apply(standardize_string).unique())}')

print('\n### ISBN Describe ###')
print(f'Number of rows: {len(df_cap['ISBN'])}')
print(f'Number of rows (dropna): {len(df_cap['ISBN'].dropna())}')
print(f'Number of unique books ISBN: {len(df_cap['ISBN'].dropna().unique())}')
print(f'Number of unique books ISBN (lower): {len(df_cap['ISBN'].dropna().str.lower().unique())}')
print(f'Number of unique books ISBN (standardize): {len(df_cap['ISBN'].dropna().apply(standardize_string).unique())}')


# ### EVENTOS

# In[18]:


df_eventos.head()


# In[19]:


print('### Event name Describe ###')
print(f'Number of rows: {len(df_eventos['TITULO-DO-TRABALHO'])}')
print(f'Number of rows (dropna): {len(df_eventos['TITULO-DO-TRABALHO'].dropna())}')
print(f'Number of unique work names: {len(df_eventos['TITULO-DO-TRABALHO'].unique())}')
print(f'Number of unique work names (lower): {len(df_eventos['TITULO-DO-TRABALHO'].str.lower().unique())}')
print(f'Number of unique work names (standardize): {len(df_eventos['TITULO-DO-TRABALHO'].apply(standardize_string).unique())}')

print('\n### ISBN Describe ###')
print(f'Number of rows: {len(df_eventos['ISBN'])}')
print(f'Number of rows (dropna): {len(df_eventos['ISBN'].dropna())}')
print(f'Number of unique event work ISBN: {len(df_eventos['ISBN'].dropna().unique())}')
print(f'Number of unique event work ISBN (lower): {len(df_eventos['ISBN'].dropna().str.lower().unique())}')
print(f'Number of unique event work ISBN (standardize): {len(df_eventos['ISBN'].dropna().apply(standardize_string).unique())}')

print('\n### DOI Describe ###')
print(f'Number of rows: {len(df_eventos['DOI'])}')
print(f'Number of rows (dropna): {len(df_eventos['DOI'].dropna())}')
print(f'Number of unique event work DOI: {len(df_eventos['DOI'].dropna().unique())}')
print(f'Number of unique event work DOI (lower): {len(df_eventos['DOI'].dropna().str.lower().unique())}')
print(f'Number of unique event work DOI (standardize): {len(df_eventos['DOI'].dropna().apply(standardize_string).unique())}')


# ## Prepare Data

# ### LIVROS

# In[20]:


# create column with standardize book name
df_book_raw['STANDARDIZE-TITULO-DO-LIVRO'] = df_book_raw['TITULO-DO-LIVRO'].apply(standardize_string)
df_book_raw.head()


# In[21]:


# remove ' first char and transform to int
df_book_raw.LattesID = df_book_raw.LattesID.apply(lambda x: x.replace("'",'')).astype('int')


# In[22]:


# create df with id -> standardize_book_name
df_book_id = pd.DataFrame({'STANDARDIZE-TITULO-DO-LIVRO': df_book_raw['STANDARDIZE-TITULO-DO-LIVRO'].unique()})\
    .reset_index(drop=False)\
        .rename({'index': 'book_id'}, axis=1)
        
df_book_id.head()


# In[23]:


df_book_raw = df_book_raw.merge(
    df_book_id,
    on = 'STANDARDIZE-TITULO-DO-LIVRO'
)
df_book_raw.head()


# In[24]:


# create df with user and the book written
df_user_book = df_book_raw[['LattesID', 'book_id']]


# In[25]:


# list with all book atributes
BOOK_ATR = [
    "NATUREZA",
    "IDIOMA",
    "TIPO",
    "NOME-DA-EDITORA",
    "CIDADE-DA-EDITORA",
    "PAIS-DE-PUBLICACAO",
    "ISBN",
    "NOME-GRANDE-AREA-DO-CONHECIMENTO2",
    "NOME-DA-AREA-DO-CONHECIMENTO2",
    "NOME-DA-SUB-AREA-DO-CONHECIMENTO2",
    "NOME-DA-ESPECIALIDADE2",
    "NOME-GRANDE-AREA-DO-CONHECIMENTO3",
    "NOME-DA-AREA-DO-CONHECIMENTO3",
    "NOME-DA-SUB-AREA-DO-CONHECIMENTO3",
    "NOME-DA-ESPECIALIDADE3",
    'STANDARDIZE-TITULO-DO-LIVRO'
]


# In[26]:


# create df with book_id, standardize name and the most common atributes
df_book = df_book_raw[BOOK_ATR + ['book_id']].groupby('book_id').agg(lambda x: most_common_ignore_na(x))
df_book.reset_index(drop=False, inplace = True) 


# In[27]:


df_book.head()


# In[28]:


df_user_book.head()


# ### PERIODICOS

# In[29]:


# create column with standardize names
df_per['TITULO-DO-ARTIGO'] = df_per['TITULO-DO-ARTIGO'].apply(standardize_string)
df_per['TITULO-DO-PERIODICO-OU-REVISTA'] = df_per['TITULO-DO-PERIODICO-OU-REVISTA'].apply(standardize_string)


# In[30]:


# create df with id -> (TITULO-DO-ARTIGO, TITULO-DO-PERIODICO-OU-REVISTA)
df_per_id = df_per[['TITULO-DO-ARTIGO','TITULO-DO-PERIODICO-OU-REVISTA']].drop_duplicates()\
    .reset_index(drop=False)\
        .rename({'index':'paper_id'}, axis=1)
        
df_per_id.head()


# In[31]:


df_per = df_per.merge(
    df_per_id,
    on = ['TITULO-DO-ARTIGO','TITULO-DO-PERIODICO-OU-REVISTA']
)
df_per.head()


# In[32]:


# create df with user and the paper written
df_user_per = df_per[['LattesID','paper_id']]
df_user_per


# In[33]:


df_per.columns.to_list()


# In[34]:


PAPER_ATR = ['ANO-DO-ARTIGO',
 'TITULO-DO-ARTIGO',
 'NATUREZA',
 'IDIOMA',
 'ISSN',
 'TITULO-DO-PERIODICO-OU-REVISTA',
 'PAGINA-INICIAL',
 'PAGINA-FINAL',
 'VOLUME',
 'FASCICULO',
 'SERIE',
 'LOCAL-DE-PUBLICACAO',
 'PAIS-DE-PUBLICACAO',
 'DOI',
 'NOME-GRANDE-AREA-DO-CONHECIMENTO',
 'NOME-DA-AREA-DO-CONHECIMENTO',
 'NOME-DA-SUB-AREA-DO-CONHECIMENTO',
 'NOME-DA-ESPECIALIDADE']


# In[35]:


# from tqdm import tqdm

# df_per[PAPER_ATR] = df_per[PAPER_ATR].astype('str')
# # Divide o dataframe em grupos
# grouped = df_per[PAPER_ATR + ['paper_id']].groupby('paper_id')

# # Lista para armazenar os resultados
# results = []

# # Itera sobre cada grupo com a barra de progresso
# for paper_id, group in tqdm(grouped, desc="Processing Groups"):
#     # Aplica a função e armazena o resultado
#     agg_result = group.agg(lambda x: most_common_ignore_na(x))
#     agg_result['paper_id'] = paper_id
#     results.append(agg_result)

# # Concatena os resultados em um dataframe
# df_per = pd.concat(results, ignore_index=True)


# In[36]:


# create df with book_id, standardize name and the most common atributes
df_per[PAPER_ATR] = df_per[PAPER_ATR].astype('str')
df_per = df_per[PAPER_ATR + ['paper_id']].groupby('paper_id').agg(lambda x: most_common_ignore_na(x))
df_per.reset_index(drop=False, inplace = True) 


# In[37]:


df_per.head()


# ### CAPITULOS

# In[38]:


df_cap.head(1)


# In[39]:


df_cap.LATTES_ID = df_cap.LATTES_ID.apply(lambda x: x.replace("'", '')) 
df_cap.head(1)


# In[40]:


# create column with standardize names
df_cap['TITULO-DO-CAPITULO-DO-LIVRO'] = df_cap['TITULO-DO-CAPITULO-DO-LIVRO'].apply(standardize_string)
df_cap['TITULO-DO-LIVRO'] = df_cap['TITULO-DO-LIVRO'].apply(standardize_string)


# In[41]:


# create df with id -> (TITULO-DO-ARTIGO','TITULO-DO-PERIODICO-OU-REVISTA)
df_cap_id = df_cap[['TITULO-DO-CAPITULO-DO-LIVRO', 'TITULO-DO-LIVRO']].drop_duplicates()\
    .reset_index(drop=False)\
        .rename({'index':'chapter_id'}, axis=1)
        
df_cap_id.head()


# In[42]:


df_cap = df_cap.merge(
    df_cap_id,
    on = ['TITULO-DO-CAPITULO-DO-LIVRO', 'TITULO-DO-LIVRO']
)
df_cap.head()


# In[43]:


# create df with user and the chapter written
df_user_cap = df_cap[['LATTES_ID','chapter_id']]
df_user_cap = df_user_cap.rename({'LATTES_ID': 'LattesID'}, axis =1)
df_user_cap


# In[44]:


df_cap.columns.to_list()


# In[45]:


CHAPTER_ATR = [ 'ANO',
 'TITULO-DO-CAPITULO-DO-LIVRO',
 'TITULO-DO-LIVRO',
 'IDIOMA',
 'TIPO',
 'NOME-DA-EDITORA',
 'CIDADE-DA-EDITORA',
 'PAIS-DE-PUBLICACAO',
 'ISBN',
 'PAGINA-INICIAL',
 'PAGINA-FINAL',
 'NOME-PARA-CITACAO',
 'NOME-COMPLETO-DO-AUTOR',
 'NOME-GRANDE-AREA-DO-CONHECIMENTO',
 'NOME-DA-AREA-DO-CONHECIMENTO',
 'NOME-DA-SUB-AREA-DO-CONHECIMENTO',
 'NOME-DA-ESPECIALIDADE',
 'NOME-GRANDE-AREA-DO-CONHECIMENTO2',
 'NOME-DA-AREA-DO-CONHECIMENTO2',
 'NOME-DA-SUB-AREA-DO-CONHECIMENTO2',
 'NOME-DA-ESPECIALIDADE2',
 'NOME-GRANDE-AREA-DO-CONHECIMENTO3',
 'NOME-DA-AREA-DO-CONHECIMENTO3',
 'NOME-DA-SUB-AREA-DO-CONHECIMENTO3',
 'NOME-DA-ESPECIALIDADE3'
]


# In[46]:


# create df with book_id, standardize name and the most common atributes
df_cap[CHAPTER_ATR] = df_cap[CHAPTER_ATR].astype('str')
df_cap = df_cap[CHAPTER_ATR + ['chapter_id']].groupby('chapter_id').agg(lambda x: most_common_ignore_na(x))
df_cap.reset_index(drop=False, inplace = True) 


# ### EVENTOS

# In[47]:


df_eventos.head(1)


# In[48]:


# create column with standardize names
df_eventos['STANDARDIZE-TITULO-DO-TRABALHO'] = df_eventos['TITULO-DO-TRABALHO'].astype('str').apply(standardize_string)
df_eventos['STANDARDIZE-NOME-DO-EVENTO'] = df_eventos['NOME-DO-EVENTO'].astype('str').apply(standardize_string)


# In[49]:


# create df with id -> 'STANDARDIZE-TITULO-DO-TRABALHO', 'STANDARDIZE-NOME-DO-EVENTO'
df_eventos_id = df_eventos[['STANDARDIZE-TITULO-DO-TRABALHO', 'STANDARDIZE-NOME-DO-EVENTO']].drop_duplicates()\
    .reset_index(drop=False)\
        .rename({'index':'paper_id'}, axis=1)
        
df_eventos_id.head()


# In[50]:


df_eventos = df_eventos.merge(
    df_eventos_id,
    on = ['STANDARDIZE-TITULO-DO-TRABALHO', 'STANDARDIZE-NOME-DO-EVENTO']
)
df_eventos.head()


# In[51]:


# create df with user and the paper written
df_user_eventos = df_eventos[['LattesID','paper_id']]
df_user_eventos


# In[52]:


df_eventos.columns.to_list()


# In[53]:


EVENT_ATR = [
    'ANO-DO-TRABALHO',
 'TITULO-DO-TRABALHO',
 'NATUREZA',
 'IDIOMA',
 'TITULO-DOS-ANAIS-OU-PROCEEDINGS',
 'PAGINA-INICIAL',
 'PAGINA-FINAL',
 'CLASSIFICACAO-DO-EVENTO',
 'NOME-DO-EVENTO',
 'CIDADE-DO-EVENTO',
 'ANO-DE-REALIZACAO',
 'PAIS-DO-EVENTO',
 'ISBN',
 'DOI',
 'NOME-GRANDE-AREA-DO-CONHECIMENTO',
 'NOME-DA-AREA-DO-CONHECIMENTO',
 'NOME-DA-SUB-AREA-DO-CONHECIMENTO',
 'NOME-DA-ESPECIALIDADE',
 'STANDARDIZE-TITULO-DO-TRABALHO',
 'STANDARDIZE-NOME-DO-EVENTO' 
]


# In[ ]:


# create df with book_id, standardize name and the most common atributes
df_eventos[EVENT_ATR] = df_eventos[EVENT_ATR].astype('str')
df_eventos = df_eventos[EVENT_ATR + ['paper_id']].groupby('paper_id').agg(lambda x: most_common_ignore_na(x))
df_eventos.reset_index(drop=False, inplace = True) 


# ### COAUTORSHIP

# In[ ]:


display(df_user_per.head(1))
display(df_user_eventos.head(1))
display(df_user_cap.head(1))
display(df_user_book.head(1))


# In[ ]:


# transforma tudo em str
df_user_per.paper_id = df_user_per.paper_id.astype('str')
df_user_eventos.paper_id = df_user_eventos.paper_id.astype('str')
df_user_cap.chapter_id = df_user_cap.chapter_id.astype('str')
df_user_book.book_id = df_user_book.book_id.astype('str')

display(df_user_per.head(1))
display(df_user_eventos.head(1))
display(df_user_cap.head(1))
display(df_user_book.head(1))


# In[ ]:


df_user_per.paper_id = df_user_per.paper_id.apply(lambda x: f'periodicos_{x}')
df_user_eventos.paper_id = df_user_eventos.paper_id.apply(lambda x: f'eventos_{x}')
df_user_cap.chapter_id = df_user_cap.chapter_id.apply(lambda x: f'capitulos_{x}')
df_user_book.book_id = df_user_book.book_id.apply(lambda x: f'livros_{x}')

display(df_user_per.head(1))
display(df_user_eventos.head(1))
display(df_user_cap.head(1))
display(df_user_book.head(1))


# In[ ]:


df_user_per.rename({'paper_id':'work_id'}, axis=1, inplace=True)
df_user_eventos.rename({'paper_id':'work_id'}, axis=1, inplace=True)
df_user_cap.rename({'chapter_id':'work_id'}, axis=1, inplace=True)
df_user_book.rename({'book_id':'work_id'}, axis=1, inplace=True)


display(df_user_per.head(1))
display(df_user_eventos.head(1))
display(df_user_cap.head(1))
display(df_user_book.head(1))


# In[ ]:


df_co = pd.concat([df_user_per, df_user_eventos, df_user_cap, df_user_book])
df_co


# In[ ]:


# drop duplicates (check in the future)
df_co.drop_duplicates(inplace=True)


# In[ ]:


len(df_co.LattesID.unique())


# ## Create Graph

# ### COAUTHORSHIP

# In[ ]:


print(df_co.shape)
print(df_co.drop_duplicates(subset=['work_id']).shape)


# In[ ]:


import pandas as pd
import networkx as nx
from itertools import combinations

# Identificar pares de coautores para cada trabalho
pairs = []
gp = df_co.groupby('work_id')
for work_id, group in tqdm(gp, total = len(gp)):
    users = list(group['LattesID'])
    if len(users) > 1:  # Existem coautores apenas se houver mais de um usuário
        pairs.extend(combinations(users, 2))

# Contabilizar as coautorias
coauthor_df = pd.DataFrame(pairs, columns=['author1', 'author2'])
coauthor_df['count'] = 1
coauthor_weighted = coauthor_df.groupby(['author1', 'author2']).sum().reset_index()

# Construir o grafo
G = nx.Graph()
for _, row in tqdm(coauthor_weighted.iterrows(), total = len(coauthor_weighted)):
    G.add_edge(row['author1'], row['author2'], weight=row['count'])

# Exibir algumas informações do grafo
print("Nós do grafo:", len(G.nodes()))
print("Arestas do grafo (com pesos):", len(G.edges(data=True)))


# In[ ]:


coauthor_weighted


# In[ ]:


import matplotlib.pyplot as plt
import networkx as nx

min_weight_edge = -1


# Configurar o layout do grafo
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.1, seed=42)  # 'k' controla a distância entre os nós

# Configurar as arestas com pesos (exibir apenas arestas com peso > 1 para melhor visualização)
edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > min_weight_edge]
weights = [G[u][v]['weight'] for u, v in edges]

# Desenhar nós
nx.draw_networkx_nodes(G, pos, node_size=20, node_color='skyblue', alpha=0.7)

# Desenhar arestas ponderadas
nx.draw_networkx_edges(G, pos, edgelist=edges, width=[w * 0.2 for w in weights], edge_color='gray', alpha=0.5)

# Adicionar rótulos aos nós (apenas alguns, para evitar poluição visual)
sampled_nodes = dict(list(pos.items())[:50])  # Limita a visualização de rótulos a 50 nós
# nx.draw_networkx_labels(G, sampled_nodes, font_size=6, font_color='black', font_family='sans-serif')

# Ocultar eixos e mostrar o grafo
plt.axis('off')
plt.title("Grafo de Coautorias")
plt.show()


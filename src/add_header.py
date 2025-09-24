import pandas as pd
import os
from typing import List

files = ['areas', 'atuacoes', 'capitulos', 'enderecos', 'eventos', 'formacoes',
    'gerais', 'linhas', 'livros', 'participacoesEventos', 'periodicos', 'projetos']

scopes = ['abrangente', 'aplicacoes', 'restritivo']

DATA_DIR = 'data/'

def make_unique(lst: List) -> List:
    """
    Transforms repeated elements in a list of strings into unique strings,
    adding an incremental numeric suffix for each subsequent occurrence of a repeated element.

    Parameters:
    -----------
    lst : list of str
    A list of strings where some elements may be repeated.

    Returns:
    --------
    list of str
    A new list where each element is unique. For repeated elements,
    a numeric suffix is added to differentiate them (e.g., 'a', 'a2', 'a3').

    Example:
    --------
    >>> make_unique(['a', 'a', 'b', 'a', 'b'])
    ['a', 'a2', 'b', 'a3', 'b2']
    """
    count_dict = {}
    result = []
    
    for item in lst:
        if item in count_dict:
            count_dict[item] += 1
            result.append(f"{item}{count_dict[item]}")
        else:
            count_dict[item] = 1
            result.append(item)
    
    return result


for scope in scopes:
    for file in files:

        try:
            # paths
            header_file_path = DATA_DIR + 'raw/headers/' + file + '.cab'
            csv_file_path = DATA_DIR + 'raw/' + scope +'/' + file + '.csv'
            
            # Load header from .cab file
            with open(header_file_path, 'r') as file_header:
                header = file_header.read().strip().split(',')

            # Transform the string inside the list into a new list separated by "\t"
            header = header[0].split('\t')
            
            # Make columns unique
            header = make_unique(header)
            
            # Load csv file without header
            df = pd.read_csv(csv_file_path, header=None, delimiter='\t', names = header, low_memory=False)

            # Some headers have unnecessary columns to the right.
            # Therefore, the header will have the same size as the number of columns in the dataframe,
            # removing columns to the right.
            columns_lenght = df.shape[1]
            header = header[:columns_lenght]
            
            # Detect how many badly formatted lines were skipped in each dataframe
            linhas_lidas = len(df)
            total_linhas = sum(1 for line in open(csv_file_path))
            linhas_puladas = total_linhas - linhas_lidas
            
            print(f'Unread lines in {scope} {file}: {linhas_puladas} of {total_linhas}')

            # Insert the header read from the .cab file
            df.columns = header
            
            outdir = DATA_DIR + 'processed/' + scope + '/'
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            
            df.to_csv(outdir + file + '.csv', index=False)

        except Exception as e:
            print('ERROR!!!')
            print(f'scope = {scope}')
            print(f'file = {file}')
            print(len(header))
            print(df.shape)
            # Code to be executed for any other exception
            print(f"An error occurred: {e}")
            raise 
        

header_orientacoes = ['LattesID',
             'NATUREZA',
             'STATUS',
             'ANO',
             'NomeDoOrientador',
             'CODIGO-INSTITUICAO',
             'NOME-INSTITUICAO',
             'CODIGO-CURSO',
             'FLAG-BOLSA',
             'CODIGO-AGENCIA-FINANCIADORA',
             'NOME-DA-AGENCIA',
             'TITULO',
             'NumeroIdOrientado',
             'NOME-CURSO',
             'NomeGrandeAreaDoConhecimento',
             'NomeDaAreaDoConhecimento',
             'NomeDaSubAreaDoConhecimento',
             'TIPO-DE-ORIENTACAO-CONCLUIDA',
             'TIPO-DE-ORIENTACAO']

for scope in scopes:

    try:
        # paths
        csv_file_path = DATA_DIR  + 'raw/orientacoes/' + scope + '/orientacoes.csv'

        # Load csv file without header
        df = pd.read_csv(csv_file_path, header=None, delimiter='\t', names = header_orientacoes)
        
        outdir = DATA_DIR  + 'processed/' + scope + '/'
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        
        df.to_csv( outdir + 'orientacoes.csv', index=False)
    
    except Exception as e:
        # Code to be executed for any other exception
        print(f"An error occurred: {e}")

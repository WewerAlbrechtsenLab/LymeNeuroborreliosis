import pandas as pd
import numpy as np
import math
import itertools as it
import plotly.graph_objects as go
from scipy.stats import zscore, variation
from natsort import natsorted, ns
import seaborn as sns
from matplotlib.pyplot import gcf
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import plotly.express as px
from umap import UMAP
from inmoose.pycombat import pycombat_norm
from plotly.subplots import make_subplots
import pimmslearn.sampling
from pimmslearn.sklearn.ae_transformer import AETransformer


def prep_data(info_filepath, proteome_filepath, run='training'):
    proteome = pd.read_csv(proteome_filepath, sep='\t')
    info = pd.read_excel(info_filepath)[['sample_id', 'subject','diagnosis', 'tissue_info', 'batch_number', 'study_cohort', 'sample_date', 'dob', 'sex', 'origin']]
    info.sample_id = info.sample_id.astype(str)
    pool = proteome.columns[(proteome.columns.str.contains('pool', case = False))].tolist()
    blank = proteome.columns[proteome.columns.str.contains('blank|empty', case = False)].tolist()
    samples = proteome.columns[(~proteome.columns.str.contains('Protein|Genes'))&(proteome.columns.isin(pool+blank)==False)].tolist()
    # subset to quantity columns
    pg_sub = proteome[['Protein.Group','Protein.Names','Genes']+samples+pool+blank]
    print('Number of pool:{}, blank:{}, study samples:{}'.format(len(pool), len(blank), len(samples)))

    # clean sample-, gene- and uniprot id
    #pg_sub.loc[(pg_sub['PG.Genes'].isnull()),'PG.Genes']=pg_sub.loc[(pg_sub['PG.Genes'].isnull())]['PG.ProteinNames'].str.split('_').str[-2].tolist()
    pg_sub.loc[:,'name']=pg_sub['Genes'].str.split(';', expand = True)[0]
    pg_sub.loc[:,'uniprot_id'] = proteome['Protein.Group'].str.split(';').str[0]
    pg_sub.loc[(pg_sub.name.isnull())|(pg_sub.name==''),'name']= pg_sub.loc[(pg_sub.name.isnull())|(pg_sub.name=='')]['Protein.Names'].str.split(';').str[0].str.split('_').str[0]

    # remove empty rows and rows without majority proteinid
    print('Number of rows (proteins) in original data, shape: {}'.format(pg_sub.shape[0]))
    pg_sub = pg_sub.dropna(subset = pg_sub.columns.tolist()[2:-2], how = 'all', axis =0)
    print('Empty rows removed, shape: {}'.format(pg_sub.shape[0]))
    pg_sub = pg_sub.loc[(pg_sub.name.notnull())&(pg_sub.name!='')] # empty
    print('Empty id-name removed, shape: {}'.format(pg_sub.shape[0]))
    pg_sub.columns = pg_sub.columns.str.split('\\').str[-1].str.split('.').str[0]

    # data transform
    # transform to long format
    print(pg_sub[pg_sub.name.str.contains('HS71')].head())

    pg_long = pg_sub.drop(['Protein','Genes'], axis = 1).melt(id_vars=['name','uniprot_id'])
    pg_long.columns = ['name','uniprot_id','samples','LFQ_intensity']
    # add clean name and plate/batch info
    #OBSinfo['samples'] = info['replicate'].astype(str)+'_'+info['analytical_sample external_id']
    print(pg_long.shape)

    # add sample type
    pg_long['type']='sample'
    pg_long.loc[pg_long.samples.str.contains('pool', case = False),'type']='pool'
    pg_long.loc[pg_long.samples.str.contains('blank|empty', case = False),'type']='blank'
   
    # remove nans
    pg_long.dropna(inplace = True)
    # add cohort, tissue and batch information
    substrings = pg_long.samples.str.split('_', expand = True).iloc[:,3:]
    pg_long['batch'] = substrings.apply(lambda row: find_string_value(row, ['Plate','plate']), axis=1)
    pg_long['cohort'] = substrings.apply(lambda row: find_string_value(row, ['Cohort','cohort']), axis=1)
    pg_long['tissue'] = substrings.apply(lambda row: find_string_value(row, ['Plasma','plasma','CSF','csf','Serum','serum']), axis=1)
    pg_long.tissue.replace({'Serum':'serum','Plasma':'plasma','SerumPool':'serum'}, inplace = True)
    pg_long.loc[pg_long['type']=='blank','tissue']='blank'
    # extract sample id
    pg_long['sample_id']= pg_long.samples.str.split('_').str[-1]
    # correct sample ids
    to_correct = pg_long.loc[(pg_long.sample_id.str.contains('-CSF|2022'))&(pg_long.samples.str.contains('Pool')==False)].samples.unique().tolist()
    pg_long.loc[pg_long.sample_id.str.contains('-CSF'),'sample_id']=pg_long.loc[pg_long.sample_id.str.contains('-CSF')].sample_id.str.split('-').str[0]
    pg_long.loc[pg_long.sample_id.str.contains('2022'),'sample_id']=pg_long.loc[pg_long.sample_id.str.contains('2022')].samples.str.split('-CSF').str[0].str.split('_').str[-1]
    pg_long.loc[pg_long.sample_id.str.contains('012000'),'sample_id'] = pg_long.loc[pg_long.sample_id.str.contains('012000')].sample_id.str[1:]
    # correct cohort description
    pg_long.replace({'LNBCohort3':3,'LNB2ndCohort':2}, inplace = True)
    # merge with infofile
    pg_long = pd.merge(pg_long, info, on = 'sample_id', how = 'left')
    # add tissue informartion from info file
    pg_long.loc[pg_long.tissue_info.notnull(),'tissue']= pg_long.loc[pg_long.tissue_info.notnull()].tissue_info.tolist()
    # Calculate number of proteins
    count_df = pg_long.groupby(['samples','type']).size().reset_index()
    count_df.columns = ['samples','type','proteins']
    count_df = pd.merge(count_df, pg_long[['samples','batch','tissue','cohort']].drop_duplicates(), on = 'samples')
    count_df= count_df.sort_values(['type','proteins'], ascending = [False, False])


    # remove samples with few proteins
    exclusion_list = []
    # samples below wiskers or 2std:
    for sample_type in ['pool','blank','sample']:
        sub = count_df.loc[count_df['type']==sample_type]
        if sub.shape[0]==0:
            print('There are no samples in category: {}'.format(sample_type))
        else:
            Q1 = sub.proteins.quantile(0.25)
            Q3 = sub.proteins.quantile(0.75)
            IQR = Q3 - Q1
            lower_lim = Q1-1.5*IQR
            print('Sample category: {}\nNumber of samples with a protein number below 1.5IQR from Q1 ({}): {}'.format(sample_type,round(lower_lim),sub.loc[(sub.proteins<lower_lim)].shape[0]))
            if sample_type=='sample':
                exclusion_list.append(i for i in sub.loc[(sub.proteins<lower_lim)].samples.unique().tolist())
    exclusion_list = [item for sublist in exclusion_list for item in sublist]

    print(pg_long.shape)
    #pg_long = pg_long.loc[~pg_long.samples.isin(exclusion_list)]
    print(pg_long.shape)
    pg_long = pg_long.loc[pg_long['type']!='blank']
    pg_long['run']=run
    return(pg_long, exclusion_list, pg_sub)

# Function to find the value containing a specific string in each row
def find_string_value(row, target_string):
    for value in row:
        for string in target_string:
            if string in str(value):
                return value
    return None

from datetime import datetime

# Function to fix year for date-like strings
def fix_year_if_needed(date_val):
    if isinstance(date_val, (pd.Timestamp, datetime)):  # Check if the value is already a datetime object
        return date_val
    else:
        # Handle string or non-datetime values
        day, month, year = map(int, date_val.split('-'))
        if year <= 24:
            year += 2000  # Years <= 24 treated as 20xx
        else:
            year += 1900  # Years > 24 treated as 19xx
        return pd.Timestamp(year, month, day)

def number_of_proteins(pg_long, index_cols, output_path):
    count_df = pg_long.groupby(index_cols).size().reset_index()
    count_df.columns = index_cols+['proteins']
    count_df= count_df.sort_values(['sample_type','proteins'], ascending = [False, False])
    # plot no proteins
    fig1 = px.bar(count_df, x = 'id',y='proteins', color = 'sample_type', template = 'simple_white',
                hover_name='id', hover_data=['proteins','id','sample_type'])
    fig1.update_xaxes(showticklabels=False)
    fig1.show()

    fig2 = px.histogram(count_df, x="proteins", color="sample_type",
                    marginal="box", # or violin, rug
                    hover_data=count_df.columns,
                    facet_col='sample_type',facet_col_wrap = 1,
                    template = 'simple_white')
    fig2.update_layout(bargap = 0.1)
    fig2.update_xaxes(range=[0,count_df.proteins.max()])
    fig2.update_yaxes(matches=None)
    fig2.show()  

    # plot no proteins
    fig3 = px.box(count_df, x = 'sample_type',y='proteins', color = 'sample_type', template = 'simple_white',
                hover_name='sample_type', hover_data=['proteins','sample_type','id'])#, 
                #color_discrete_map=px.colors.qualitative.Prism)
    fig3.show()

    fig3.write_image('{}/proteinnumbers_astral.pdf'.format(output_path), width = 1000)
    fig2.write_image('{}/proteinnumbers_marginal_astral.pdf'.format(output_path), width = 1000)


def dynamic_range(pg, col_type, hilights_lst, save_plot = False):
    dr_df = pg.copy()
    dr_df_melt = dr_df.groupby('sample_type').mean().reset_index().melt(id_vars=['sample_type'])
    dr_df_melt.columns = ['sample_type','name','LFQ_intensity']

    dr_df_melt['completeness']=dr_df.groupby('sample_type').apply(lambda x: x.notnull().mean()).reset_index().melt(id_vars='sample_type').value.tolist()
    dr_df_melt['missingness']=1-dr_df_melt['completeness']
    dr_df = dr_df_melt.sort_values(['sample_type','LFQ_intensity'], ascending = [True,False])
    no_groups = dr_df.sample_type.unique().shape[0]
    dr_df['rank'] = int(no_groups)*list(range(1,int(dr_df.shape[0]/no_groups)+1))

    dr_df['genename']=dr_df.name.str.split('~').str[0]
    print(len(highlights))
    dr_df.loc[dr_df.genename.isin(highlights)==False,'genename']=''
    print(dr_df.genename.unique().shape)

    # DR for samples only
    fig = px.scatter(dr_df.loc[dr_df.sample_type=='SA'], x = 'rank', y = 'LFQ_intensity', hover_name='genename',color = 'completeness', 
                    text = 'genename',
                    hover_data = ['name','LFQ_intensity','rank','completeness'], 
                    template = 'simple_white', 
                    labels={
                        "LFQ_intensity": "log10(LFQ)",
                        "rank": "Proteins ranked by abundance","completeness": "Completeness"
                    },
                    color_continuous_scale='Blues', title="Dynamic range", height = 1000, width = 1000)
    fig.update_layout(font_family='arial')
    fig.update_traces(textposition='top right')
    if save_plot:
        fig.write_image('../output/v3/DR_sa_astral.pdf',height = 1000, width = 1000)
    fig.show()

def filter_missingness(df, feat_prevalence=.2, axis=0):
    N = df.shape[axis]
    minimum_freq = N * feat_prevalence
    freq = df.notna().sum(axis=axis)
    mask = freq >= minimum_freq
    print(f"Drop {(~mask).sum()} along axis {axis}.")
    freq = freq.loc[mask]
    if axis == 0:
        df = df.loc[:, mask]
    else:
        df = df.loc[mask]
    return df

def pimms_imputation(df_wide):
    val_X, train_X = pimmslearn.sampling.sample_data(df_wide.stack(),
                                           sample_index_to_drop=0,
                                           weights=df_wide.notna().sum(),
                                           frac=0.1,
                                           random_state=42,)

    val_X, train_X = val_X.unstack(), train_X.unstack()
    val_X = pd.DataFrame(pd.NA, index=train_X.index,
                     columns=train_X.columns).fillna(val_X)

    model_selected = 'VAE'  # 'DAE'
    model = AETransformer(
        model=model_selected,
        hidden_layers=[512,],
        latent_dim=50,
        out_folder='runs/scikit_interface',
        batch_size=10,
    )

    model.fit(train_X, val_X,
            epochs_max=50,
            cuda=False)
    
    df_imputed = model.transform(df_wide)
    
    # plot distribution

    # Identify NaN positions
    nan_mask = df_wide.isna()

    # Create a long-format DataFrame for plotting
    data = []
    for col in df_wide.columns:
        # Original non-NaN values
        data.extend([(val, col, 'Original') for val in df_wide[col].dropna()])
        # Replaced NaN values
        data.extend([(df_imputed[col][i], col, 'Replaced') for i in df_wide.index if nan_mask[col][i]])

    plot_df = pd.DataFrame(data, columns=['Value', 'Column', 'Type'])

    # Plot using Plotly Express
    fig = px.histogram(plot_df, x="Value", color="Type", 
                    barmode="overlay", title="Histogram of Original & Replaced Values",
                    template = 'simple_white')
    fig.show()

    return df_imputed

def batch_correction(processed_data, plate_lst):
    corrected_data = pycombat_norm(processed_data.T,plate_lst).T
    #corrected_data = processed_data.copy()
    return corrected_data

def umap_plot(processed_data, protein_cols, index_cols, info, savefig=False, output_path='', title = 'not_specified'):
    # processed and imputed
    umap_2d = UMAP(n_components=2, init='random', random_state=0)
    umap_df = pd.DataFrame(umap_2d.fit_transform(processed_data[protein_cols]))

    sub = pd.merge(info, processed_data, on = 'filename', how='right')

    for col in index_cols:
        fig = px.scatter(
        umap_df, x=0, y=1,
        color=sub[col],
        template = 'simple_white', height = 500, width = 700)
        if savefig:
            fig.write_image(output_path+"umap_plot_{}_{}.pdf".format(col,title))
        fig.show()


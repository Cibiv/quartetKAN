import numpy as np
import pandas as pd
import tensorflow.keras as keras

import argparse
import logging
import os

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from pylab import *

matplotlib.use('Agg')

# log to stdout
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s %(levelname)s: %(message)s',
    datefmt = '%d.%m.%y %H:%M:%S',
    handlers = [logging.StreamHandler()]
)


# pass arguments with flags
parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model", help = "Model to test.")
parser.add_argument("-t", "--test", default = '../data/processed/zone/test/1000bp', help = "Path to test directory")
parser.add_argument("-l", "--seqlen", default = '1000bp', help = "Sequence Length")
parser.add_argument("-o", "--offset", type = int, default = 4, help = "Index of first feature column.")
parser.add_argument("-n", "--no_feat", type = int, default = 15, help = "Number of feature columns.")
args = vars(parser.parse_args())


def test(model):
    # define output path
    output = args['model'].replace('models', 'results')
    if output.endswith('/'):
        output = output[:-1]

    # initialize table dictionary
    tmp_accuracies = {'pProb': [], 'qProb': [], 'tree': [], 'accuracy': []}

    # loop through test files
    for file in os.listdir(args["test"]):
        if not file.endswith('.csv'):
            continue 
        logging.info(file)

        # read file and split Farris and Felsenstein alignments
        df = pd.read_csv(f'{args["test"]}/{file}', usecols = [i for i in range(args['offset'], args['offset'] + 1 + args['no_feat'])])
        far = df[df['label'] == 1]
        fel = df[df['label'] == 0]

        # evaluate model on data
        score_far = model.evaluate(x = far.iloc[:, :-1].values, y = far.iloc[:, -1:].values, verbose = 0)[1]
        score_fel = model.evaluate(x = fel.iloc[:, :-1].values, y = fel.iloc[:, -1:].values, verbose = 0)[1]
        
        # append properties of alignments and respective accuracy to table dict
        pProb = float(file[file.rfind("_p") + 2:file.rfind("_q")])
        qProb = float(file[file.rfind("_q") + 2:file.rfind(".csv")])

        tmp_accuracies['pProb'].append(pProb)
        tmp_accuracies['pProb'].append(pProb)
        tmp_accuracies['qProb'].append(qProb)
        tmp_accuracies['qProb'].append(qProb)
        tmp_accuracies['tree'].append('farris')
        tmp_accuracies['tree'].append('felsenstein')
        tmp_accuracies['accuracy'].append(score_far)
        tmp_accuracies['accuracy'].append(score_fel)

    # create dataframe from table dict
    df = pd.DataFrame.from_dict(tmp_accuracies)
    df['zone'] = np.where(df['pProb'] >= (-18*df['qProb'] + 24*df['qProb']**2 + np.sqrt(243*df['qProb'] - 567*df['qProb']**2 + 648*df['qProb']**3 - 288*df['qProb']**4))/(9 - 24*df['qProb'] +32*df['qProb']**2),1,0)

    acc_far = df[df['tree']=='farris']['accuracy'].mean()
    acc_far_z = df[(df['tree']=='farris') & (df['zone']==1)]['accuracy'].mean()
    acc_far_n = df[(df['tree']=='farris') & (df['zone']==0)]['accuracy'].mean()
    acc_fel = df[df['tree']=='felsenstein']['accuracy'].mean()
    acc_fel_z = df[(df['tree']=='felsenstein') & (df['zone']==1)]['accuracy'].mean()
    acc_fel_n = df[(df['tree']=='felsenstein') & (df['zone']==0)]['accuracy'].mean()
    acc_avg = df['accuracy'].mean()
    acc_avg_z = df[df['zone']==1]['accuracy'].mean()
    acc_avg_n = df[df['zone']==0]['accuracy'].mean()

    logging.info(f'Acc. on Far: {acc_far:.4f} (in_zone: {acc_far_z:.4f}, out_zone: {acc_far_n:.4f})')
    logging.info(f'Acc. on Fel: {acc_fel:.4f} (in_zone: {acc_fel_z:.4f}, out_zone: {acc_fel_n:.4f})')
    logging.info(f'Acc. on avg: {acc_avg:.4f} (in_zone: {acc_avg_z:.4f}, out_zone: {acc_avg_n:.4f})')

    # split Farris and Felsenstein data
    df_fels = df.loc[df['tree'] == 'felsenstein'].drop('tree', axis = 1)
    df_farris = df.loc[df['tree'] == 'farris'].drop('tree', axis = 1)

    # sort values 
    df_fels = df_fels.sort_values(['pProb'], ascending = [True])
    df_farris = df_farris.sort_values(['pProb'], ascending = [True])

    # generate table containing branch parameters and accuracy for Farris/Felsenstein trees of these parameters
    df_output = pd.merge(df_farris, df_fels, on = ['pProb', 'qProb', 'zone'], how = 'outer')
    df_output = df_output.rename(index = str, columns = {"accuracy_x": "Far", "accuracy_y": "Fel"})
    df_output = df_output[['pProb', 'qProb', 'zone', 'Far', 'Fel']]

    # save accuracy table to file
    name = f'{output}_test_seqLen_{args["seqlen"]}.csv'
    df_output['Far'] = df_output['Far'] * 100
    df_output['Fel'] = df_output['Fel'] * 100
    df_output.to_csv(name, index = False)
    logging.info(f'Saved test table to {name}.')

    # generate 2d arrays with inference accuracies for the different p-q-combis
    far_nn, fel_nn = testToAccuracy(df_output)
    far_nn = getPivot(far_nn)
    fel_nn = getPivot(fel_nn)

    # generate heatmap
    make_figure(far_nn, fel_nn, f'{output}_heatmap_permuted_dataset_seqLen_{args["seqlen"]}.png')


# get accuracies for tree-type, p- and q-value
def testToAccuracy(df):
    far_df = df
    far_df['acc'] = far_df['Far'] / 100
    far_df = far_df.drop(['zone','Fel', 'Far'], axis = 1)
    fel_df = df
    fel_df['acc'] = fel_df['Fel'] / 100
    fel_df = fel_df.drop(['zone','Fel', 'Far'], axis = 1)

    return far_df, fel_df


# generate 2d array with rows containing accuracies for fixed p parameter and columns those of fixed q parameter
def getPivot(file):
    df = file
    df = df.sort_values(['pProb', 'qProb'], ascending = [True, True])
    df['acc'] = df['acc'].round(2)
    df = df.pivot(index = "pProb", columns = "qProb", values = "acc")
    return df


# generate heatmaps
def make_figure(far, fel, name):

    # set figure and colorbar size
    fig = plt.figure(figsize = (18, 7.5))
    cbar_ax = fig.add_axes([.125, 0.02, .775, .05])

    # plot accuracy for tree inference of alignments of Farris trees as heatmap
    subplot(1, 2, 1)
    ax = sns.heatmap(far, annot = False, cmap = "YlOrRd_r", square = True, cbar = True, center = 0.5, vmin = 0, vmax = 1, cbar_ax = cbar_ax, cbar_kws = {"orientation": "horizontal"})
    ax.text((ax.get_xlim()[1]) / 2, ax.get_ylim()[0] + 0.5, "Farris", fontsize = 22, horizontalalignment = 'center')
    ax.invert_yaxis()
    ax.set_xlabel("q (3 branches)", fontsize = 24)
    ax.set_ylabel("p (2 branches)", fontsize = 24)
    plt.yticks(np.arange(0, 15, 2), ("0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"), rotation = 0, fontsize = "16")
    plt.xticks(np.arange(0, 15, 2), ("0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"), rotation = 0, fontsize = "16")

    # plot accuracy for tree inference of alignments of Felsenstein trees as heatmap
    subplot(1, 2, 2)
    ax = sns.heatmap(fel, annot = False, cmap = "YlOrRd_r", square = True, cbar = False, center = 0.5, vmin = 0, vmax = 1)
    ax.text((ax.get_xlim()[1]) / 2, ax.get_ylim()[0] + 0.5, "Felsenstein", fontsize = 22, horizontalalignment = 'center')
    ax.invert_yaxis()
    ax.set_xlabel("q (3 branches)", fontsize = 24)
    ax.set_ylabel("p (2 branches)", fontsize = 24)
    plt.yticks(np.arange(0, 15, 2), ("0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"), rotation = 0, fontsize = "16")
    plt.xticks(np.arange(0, 15, 2), ("0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"), rotation = 0, fontsize = "16")
    
    # set cbar ticks
    cbar_ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar_ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize = 24)
    
    # adjust the spacing of subplots
    subplots_adjust(left  = 0.125, right = 0.9, bottom = 0.25, top = 0.9, wspace = 0.4, hspace = 0.2)

    # save plot
    fig.savefig(name, bbox_inches = 'tight', dpi = 100)
    logging.info(f'Saved heatmap to {name}.')


# load model and test it
model = keras.saving.load_model(args['model'])
test(model)

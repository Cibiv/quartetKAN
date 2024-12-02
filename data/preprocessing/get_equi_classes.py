import pandas as pd
import numpy as np
import itertools
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="input dataset")
ap.add_argument("-x", "--offset", type=int, default=4, help="index of column aaaa (0 based)")
ap.add_argument("-d", "--degree", type=int, default=4, help="offset")
ap.add_argument("-o", "--output", help="output dataset")
args = vars(ap.parse_args())

# gather the site-pattern frequencies as well as the possibilities to permute the seq order
sp = ['aaaa','aaab','aaba','aabb','aabc','abaa','abab','abac','abba','abbb','abbc','abca','abcb','abcc','abcd']
perm = pd.read_csv('../raw/zone/permutation_list.csv',sep=';', dtype=str)
sp_ind = list(perm.columns)

# get all possibilities to combine the site-pattern freqs
sp_combis = []
for x in itertools.combinations_with_replacement(sp_ind,args['degree']):
    sp_combis.append(','.join(["%s" for i in range(args['degree'])]) %(x))

# print the info on how many combinations exist
print('INFO: there are {} site-pattern combinations'.format(len(sp_combis)))

# define routine to sort the site-patterns in a comination
def get_sorted_list(x,y):
    z=[x[i] for i in y]
    z=list(map(int,z))
    z.sort()
    return ','.join(list(map(str,z)))

# form groups of site-pattern combis which can be generated from each other by reordering of the sequences
reord_dict = dict()
still_to_check = sp_combis.copy()

ind=0
for i in range(len(sp_combis)):
    y = sp_combis[i]
    if y in still_to_check:
        y = y.split(',')
        reord_dict[ind] = set(perm.apply(lambda x: get_sorted_list(x,y), axis=1))
        for j in reord_dict[ind]:
            if j in still_to_check:
                still_to_check.remove(j)
        ind+=1

if still_to_check!=[]:
    print('ERROR: There are still site-pattern combis to check.')

print('INFO: {} groups of reordered site-pattern combis were found'.format(len(reord_dict)))

# reform dataset in groups
input = pd.read_csv(args['input'])
tmp_df = pd.DataFrame(0, index=np.arange(len(input)), columns=[str(i) for i in range(len(reord_dict))])
df = pd.concat([input.iloc[:,:args['offset']], tmp_df, input[['label']]], axis=1)

for i in range(len(reord_dict)):
    for combi in reord_dict[i]:
        combi = combi.split(',')
        summand = np.ones(len(input))
        for j in range(args['degree']):
            summand = summand * input[sp[int(combi[j])]]
        df[str(i)] = df[str(i)] + summand

df.to_csv(args['output'], index=False)

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 10:37:19 2022

@author: clair
"""

import numpy as np
import pandas as pd
import re
import time, sys
import pickle

def updt(total, progress):
    """
    Displays or updates a console progress bar.

    Original source: https://stackoverflow.com/a/15860757/1391441
    """
    barLength, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\r[{}] {:.0f}% {}".format(
        "#" * block + "-" * (barLength - block), round(progress * 100, 0),
        status)
    sys.stdout.write(text)
    sys.stdout.flush()
    
    
#Make functions to process the data

def open_CIF_file (UniProtID, directory="\\"):
    """
    Given a UniProt ID open the corresponding CIF file, for CIF files in a give directory
    
    """
    filename="AF-"+str(UniProtID)+"-F1-model_v2.cif"
    #for linux
    #file=open(directory/filename, "r")
    #for windows
    file=open(directory+filename, "r")
    return file
    
def close_CIF_file (UniProtID, directory=""):
    """
    Close CIF file
    
    """
    filename="AF-"+str(UniProtID)+"-F1-model_v2.cif"
    filename.close()
    
       

def distances_CA (residue_number, UniProtID, directory=""):
    """
    For a given residue number in a CIF file, calculate the Euclidean distances from its CA carbon to all other residue CA 
    carbons
    
    Output: array of distances of shape (x, 2) where x is the number of amino acids in the sequence
    And an array of the locations
    """
    file=open_CIF_file(UniProtID, directory)
    locations = []
    residues=[]
    lines = file.readlines()
    for line in lines:
        if "ATOM" in line:
            if "CA" in line:
                parts=str.split(line)
                array=np.array([int(parts[8]), float(parts[10]), float(parts[11]), float(parts[12])])
                locations.append(array)
                residues.append(parts[-1])
    
    location=np.stack(locations, axis=0)
    residue=np.stack(residues, axis=0)
    diff=location[:, 1:4]-location[residue_number-1, 1:4]
    
    Euc=np.array([])
    Euc=np.array([np.linalg.norm(diff[i, :]) for i in range(diff.shape[0])])
    distances=np.vstack((location[:,0], Euc))
    
    return distances, location, residue
            
        
    
def cutoff (radius, distance_CA):
    """
    radius: in Angstroms, keep all values under this distance in the distance vector
    distance_CA: distance array of shape (x, 2)
    
    Output: array of distances of shape (w, 2) with w<=x, and an array of the indexes of these for the full data
    
    """
    residue=distance_CA[0]
    distances=distance_CA[1]
    new_array=np.where(distances<radius)

    cutoff=np.vstack((residue[new_array], distances[new_array]))
    return cutoff, new_array
    
    
def make_adj (UniProtID, residue_number, radius, directory=""):
    """
    Given an uniprot ID and distance array of shape (x, 2)
    
    Output: Adjacency matrix as a numpy array of dimension (x, x)
    
    """
    distances, location, residues = distances_CA (residue_number, UniProtID, directory)
    cut, new_array = cutoff (radius, distances)
    
    accepted_loc=location[new_array]
    resi=residues[new_array]
    #print (resi)
    #print (accepted_loc[:, 0])
    
    rowlist=[]
    for i in range(accepted_loc.shape[0]):
        diff=accepted_loc[:, 1:4]-accepted_loc[i, 1:4]
        row=np.array([np.linalg.norm(diff[j, :]) for j in range(diff.shape[0])])
        rowlist.append(row)
    
    adj_matrix=np.stack(rowlist, axis=0)
    node_features=np.vstack((accepted_loc[:, 0], resi))
    
    return adj_matrix, node_features

def sparse_adj (adj_matrix, node_features):
    """
    Given an adjacency matrix and node features make a sparse adjacency matrix with distances
    
    """
    list1=[]
    list2=[]
    list3=[]
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[0]):
            if adj_matrix[i, j] != 0:
                list1.append(node_features[0, i])
                list2.append(node_features[0, j])
                list3.append(adj_matrix[i, j])
                
    sparse=np.column_stack([list1, list2, list3])
    return sparse
            
    
    

def load_kmers (file, directory, sheet_name):
    """
    Loads a csv or excel spreadsheet of UniProt IDs, sites, and k-mers

    Output: numpy array of UniProt IDs, sites and k-mers
    
    """
    sheet = pd.read_excel(file, sheet_name=sheet_name)
    data=sheet[['UniProt ID', 'Phosphorylation Index', 'k-mer']]
    data=data.to_numpy(dtype=str)
    print(data.shape)
    return data
    
def check_seq_match (cif_directory, UniProtID, seq_file='UP000005640_Ref_genome_1_seq_per_gene.fasta'):
    """
    Checks if the sequence matches between the CIF and reference sequence
    
    """
    file=open_CIF_file (UniProtID, cif_directory)
    
    lines = file.readlines()
    marker=0
    string=';'
    for line in lines:
        if "_struct_ref.pdbx_seq_one_letter_code" in line:
            marker=1
        elif "#" in line:
            marker=0
        elif marker != 0:
            string=string+line
        else:
            marker=0
    
    
    CIFstring=string.replace(";", "").replace("\n", '')
    
    seqfile=open(seq_file, "r")
    marker=0
    string2=''
    lines2=seqfile.readlines()
    for line in lines2:
        if (">" in line) & (UniProtID in line):
            marker=1
        elif (marker==1) & (">" not in line):
            string2=string2+line
        elif (marker==1) & (">" in line):
            marker=0
        else:
            marker=0
    
    Refstring=string2.replace("\n", '')
    #print(repr(Refstring))
    #print(repr(CIFstring))
    
    if Refstring==CIFstring:
        return True
    else:
        return False
     
def main_adj_matrices (file, k_directory, cif_directory, cutoff_radius, output_name, sheet_name="k-mers"):
    
    """
    For a file of k-mers, creates local adjacency matrices centered at each of the sites using alphafold predicted CIF files
    
    Output: Saves each file as a numpy file
    
    """
    kmers= load_kmers(file, k_directory, sheet_name)
    
    adjs=[]
    sp_adjs=[]
    node_feat=[]
    ID=[]
    No_match=[]
    
    for i in range(kmers.shape[0]):
        UniProtID=kmers[i, 0]
        resi=int(kmers[i,1])
        #print(UniProtID, resi)
        try:
            if check_seq_match (cif_directory, UniProtID) == True:

                distances, location, residues = distances_CA (resi, UniProtID, cif_directory)
                cut, new_array = cutoff (cutoff_radius, distances)
                ID.append(np.array([UniProtID, resi]))
        
                local_adj, node_features=make_adj(UniProtID, resi, cutoff_radius, cif_directory)
                adjs.append(local_adj)
                node_feat.append(node_features)
        
                sparse=sparse_adj(local_adj, node_features)
                sp_adjs.append(sparse)
                      
                updt(len(kmers[:, 0]), i) #Update progress bar
                #close_CIF_file (UniProtID, cif_directory)
        
            else: 
                print(UniProtID+" does not match ref seq")
                No_match.append(UniProtID)
        except:
            print("UniProtID not in structures")
            No_match.append(UniProtID)
            
    
    print("loop complete")    
    with open("/home/cstg/GraphPhos/"+output_name+"_adjs.pickle", "wb") as b:
        pickle.dump(adjs, b)
    print("Adjacency matrices saved")
    #np.save(output_name+"_sparse_adjs", sp_adjs)
    #print("Sparse adjacency matrices saved")
    with open("/home/cstg/GraphPhos/"+output_name+"_IDs.pickle", "wb") as c:
        pickle.dump(ID, c)
    print("IDs saved")
    with open("/home/cstg/GraphPhos/"+output_name+"_seqs_not_processed.pickle", "wb") as d:
        pickle.dump(No_match, d)
    print("Seqs not processed saved")

    #np.save(output_name+"_node_features", node_feat)
    with open("/home/cstg/GraphPhos/"+output_name+"node_features.pickle", 'wb') as f:
        pickle.dump(node_feat, f)
    print("Node features saved")
    
        
    print("Conversion complete")
    
main_adj_matrices('20220720_negative_dataset.xlsx', "", '/home/cstg/GraphPhos/nobackup/users/cstg/GraphPhos/cif/', 20, "Phos_20_neg_full")
            


"""
Datasets.
"""

import pandas as pd
from torch.utils.data import Dataset

from Bio import SeqIO

import random

#ALLOWED = set("ACGT-")     # Used to remove ambiguous base symbols
ALLOWED = set("ACGT-NVHDBWSMKYR")

REPLACEMENT = {
                "R": ("A", "G"),
                "Y": ("C", "T"),
                "K": ("G", "T"),
                "M": ("A", "C"),
                "S": ("C", "G"),
                "W": ("A", "T"),
                "B": ("C", "G", "T"),
                "D": ("A", "G", "T"),
                "H": ("A", "C", "T"),
                "V": ("A", "C", "G"),
                "N": ("A", "C", "G", "T"),
            }

#Remove ambiguous nucleotides to let the sequence be compatible with the code
def clean_sequence(seq):
    seq = seq.upper()  # Convert to uppercase for consistency
    for base in seq:
        if base not in ALLOWED:
            seq = seq.replace(base, "")
            continue
    return seq

def extract_species_id(header, dataset_format):
    """
    Extract species ID from header based on dataset format.
    
    Parameters
    ----------
    header : str
        The sequence header/identifier
    dataset_format : str
        The dataset name (e.g., 'Inga', 'beetle', 'fish', 'birds', etc.)
    
    Returns
    -------
    str : The extracted species ID
    """
    if dataset_format in ["beetle", "fish"]:
        # Format: "label_id,species_id"
        return header.split(",")[1] if "," in header else header
    elif dataset_format in ["birds", "fishes", "bats", "insect_dataset_species", 
                            "insect_dataset_genus", "unseen_insect_dataset", "unseen", "fish_12S", "fish_12S_Noise"]:
        # Format: "label_id|species_id"
        if dataset_format in ["unseen_insect_dataset", "fish_12S", "fish_12S_Noise", "insect_dataset_species", "insect_dataset_genus"]:  # As they come from a .mat file and labels starts from 1
            return int(header.split("|")[1]) - 1 if "|" in header else header
        else:
            return header.split("|")[1] if "|" in header else header
    elif dataset_format in ["Cypraeidae", "Drosophila", "Inga"] or dataset_format.startswith("GNe"):
        # Format: "xxx|species_id|xxx"
        return header.split("|")[1] if "|" in header else header
    else:
        raise NotImplementedError(f"Dataset format {dataset_format} not supported for species ID extraction.")


class DNADataset(Dataset):
    '''
    A PyTorch Dataset class for loading DNA sequences from a fasta file.
    It supports various dataset formats and can handle K-Fold Cross Validation.
    '''

    def __init__(
        self,
        file_path,
        dataset_format,
        choose_indexes_set=None,        # Used for K-Fold Cross Validation
        max_len=0,     # placeholder value. If 0, computes the actual max len in the dataset
        label_set=None,
        print_label_set=False,
        replace_ambiguous_bases_with_random=True,   # Whether to replace ambiguous bases with random choices instead of removing them
        number_of_augmentations_for_ambiguous_bases=0,   # Number of augmented sequences to generate for each original sequence with ambiguous bases (only used if replace_ambiguous_bases_with_random is True)
    ):
        self.max_len = max_len
        self.dataset_format = dataset_format
        self.replace_ambiguous_bases_with_random = replace_ambiguous_bases_with_random
        self.number_of_augmentations_for_ambiguous_bases = number_of_augmentations_for_ambiguous_bases

        self.choose_indexes_set = choose_indexes_set


        datas_type = "Test"
        if label_set is None:
            datas_type = "Train"


        if dataset_format not in ["fishes", "birds", "Cypraeidae", "Drosophila", "bats", "Inga", "beetle", "fish", "insect_dataset_species", "insect_dataset_genus", "unseen_insect_dataset", "unseen", "fish_12S", "fish_12S_Noise"] and not dataset_format.startswith("GNe"):
            raise NotImplementedError(f"Dataset {dataset_format} not supported.")


        print("\n----------------------------------------")
        print("Loading dataset from file:", file_path)

        seq_ids = []
        id_seq_map = {}
        augmented_sequences_generated = 0
        cleaned_sequences = []
        if dataset_format in ["birds", "fishes", "bats", "fish_12S", "fish_12S_Noise"]:

            cleaned_sequences = []
            for record in SeqIO.parse(file_path, "fasta"):
                header = record.description
                label_id = header.split("|")[0]
                seq_id = header.split("|")[1]
                

                if dataset_format in ["unseen_insect_dataset", "fish_12S", "fish_12S_Noise"]:  # As they come from a .mat file and labels starts from 1
                #if seq_id.isdigit():
                    seq_id = int(seq_id) - 1

                cleaned_sequence = clean_sequence(str(record.seq))

                #if cleaned_sequence in cleaned_sequences:
                #    if datas_type == "Train":
                #        continue

                cleaned_sequences.append(cleaned_sequence)

                seq_ids.append(seq_id)


                if header in id_seq_map:
                    len_map = len(id_seq_map)
                    label_id = f"{label_id}_{len_map}"
                    header = str(label_id) + "|" + str(seq_id)

                id_seq_map.update({header: cleaned_sequence})

        elif dataset_format in ["Cypraeidae", "Drosophila", "Inga"] or dataset_format.startswith("GNe"):

            for record in SeqIO.parse(file_path, "fasta"):
                header = record.description
                label_id = header.split("|")[0]
                seq_id = header.split("|")[1]

                seq_ids.append(seq_id)

                cleaned_sequence = clean_sequence(str(record.seq))

                cleaned_sequences.append(cleaned_sequence)

                if header in id_seq_map:
                    len_map = len(id_seq_map)
                    label_id = f"{label_id}_{len_map}"
                    header = str(label_id) + "|" + str(seq_id)

                id_seq_map.update({header: cleaned_sequence})

        elif dataset_format in ["beetle", "fish"]:

            if self.choose_indexes_set is None:
                raise ValueError(f"You can't avoid specifying the indices set when dealing with dataset {dataset_format}!")

            for idx, record in enumerate(SeqIO.parse(file_path, "fasta")):

                if idx in self.choose_indexes_set:
                    header = record.description
                    label_id = header.split(",")[0]
                    seq_id = header.split(",")[1]
                    seq_ids.append(seq_id)
                    
                    cleaned_sequence = clean_sequence(str(record.seq))

                    cleaned_sequences.append(cleaned_sequence)

                    if header in id_seq_map:
                        len_map = len(id_seq_map)
                        label_id = f"{label_id}_{len_map}"
                        header = str(label_id) + "|" + str(seq_id)

                    id_seq_map.update({header: cleaned_sequence})

        elif dataset_format in ["unseen", "insect_dataset_species", "insect_dataset_genus"]:
            
            #if self.choose_indexes_set is None and label_set is not None:       # Only for test set
                #raise ValueError(f"You can't avoid specifying the indices set when dealing with dataset {dataset_format} in test mode!")

            cleaned_sequences = []
            for idx, record in enumerate(SeqIO.parse(file_path, "fasta")):
                
                #if self.choose_indexes_set is None or idx in self.choose_indexes_set:   # For train set or for test set with specified indices

                header = record.description
                label_id = header.split("|")[0]
                seq_id = header.split("|")[1]

                if dataset_format in ["insect_dataset_species", "insect_dataset_genus"]:  # As they come from a .mat file and labels starts from 1
                    #if seq_id.isdigit():
                    seq_id = int(seq_id) - 1

                cleaned_sequence = clean_sequence(str(record.seq))

                #if cleaned_sequence in cleaned_sequences:
                #    if datas_type == "Train":
                #        continue

                cleaned_sequences.append(cleaned_sequence)

                seq_ids.append(seq_id)

                if id_seq_map.get(header) is not None and id_seq_map[header] != cleaned_sequence:
                    print(f"Warning: The same header {header} has different sequences after cleaning. This should not happen. Check the dataset for duplicates or inconsistencies.")


                if header in id_seq_map:
                    len_map = len(id_seq_map)
                    label_id = f"{label_id}_{len_map}"
                    header = str(label_id) + "|" + str(seq_id)

                id_seq_map.update({header: cleaned_sequence})

        else:
            raise NotImplementedError(f"Dataset {dataset_format} not supported.")
        

        self.barcodes = cleaned_sequences

        if max_len == 0:
            # Define the maximum length among all the retrieved barcodes
            max_len = max([len(barcode) for barcode in self.barcodes])
            print(f"Maximum length for the dataset barcodes: {max_len}")
            self.max_len = max_len
        # ------------------------------------------------------------
        
        print("Number of entries in the dataset:", len(seq_ids))
        print("Number of augmented sequences generated:", augmented_sequences_generated)

        self.seq_ids = seq_ids

        if label_set is None:
            # For training set
            self.labels, self.label_set = pd.factorize(pd.Series(seq_ids), sort=True)      # was sort=True
            print(f"The dataset has {len(self.label_set)} different labels.")
            if print_label_set:
                print("Label set:", self.label_set)
            #print(f"After factorization, there are {len(self.labels)} labels in the dataset.")
        else:
            # For test set, using the label set from training set
            self.label_set = label_set
            label_to_idx = {lbl: i for i, lbl in enumerate(label_set)}
            self.labels = [label_to_idx.get(lbl, -1) for lbl in seq_ids]
            #print(f"The dataset has {len(self.label_set)} different labels.")

            #print("Check if train set misses any label from test set:")
            missing_labels = set(seq_ids) - set(self.label_set)
            if len(missing_labels) > 0:
                print(f"Warning: The following labels are missing in the training set and will be assigned label -1: {missing_labels}")
            #else:
            #    print("All labels from test set are present in training set.")

        self.num_labels = len(self.label_set)

        self.id_seq_map = id_seq_map

        print("----------------------------------------\n")


    def __len__(self):
        return len(self.barcodes)
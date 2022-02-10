import os

import pandas as pd
import numpy as np
import torch
import plot_data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path = "/home/luo00042/M2SSD/SAMCNet/"

def get_sampler(sample_dataset, label_name, labels, transformed_samples=5):
    class_sample_count = [len([x[0] for x in list(sample_dataset.groupby([label_name, 'Sample'])) if x[0][0] == labels[i]]) * transformed_samples for i in range(len(labels))]
    sample_labels = np.asarray([[x[1][label_name].cat.codes.iloc[0]] * transformed_samples for x in list(sample_dataset.groupby(['Sample']))]).ravel()

    num_samples = sum(class_sample_count)

    class_weights = [num_samples/class_sample_count[i] for i in range(len(class_sample_count))]
    weights = [class_weights[sample_labels[i]] for i in range(int(num_samples))]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))

    return sampler

def read_tsv(in_file, use_margins=False, set_categories=True, sep='\t'):
    df = pd.read_csv(in_file, sep=sep, header=0, low_memory=False)
    #df = df[(df.Phenotype != 'Potential Artifact') & (df.Phenotype != 'Unclassified') & (df.Phenotype != 'Monocyte') & (df.Phenotype != 'Neutrophyl') & (df.Phenotype != 'Plasma Cell')]
    
    if use_margins:
        df = filter_margins(df)

    if set_categories:
        # Set necessary columns as categorical to later retrieve distinct category codes
        df.Sample = pd.Categorical(df.Sample)
        df.Pathology = pd.Categorical(df.Pathology)
        df.Phenotype = pd.Categorical(df.Phenotype)
        df.Status = pd.Categorical(df.Status)
        df.HLA1_FUNCTIONAL_threeclass = pd.Categorical(df.HLA1_FUNCTIONAL_threeclass)

    return df

def read_margin_samples(in_dir):
    mixed_dfs = []
    for filename in os.listdir(in_dir + '/' + 'mixed'):
        filepath = in_dir + '/' + 'mixed' + '/' + filename
        df = read_tsv(filepath, set_categories=False, sep=',')
        df['Sample'] = filename
        mixed_dfs.append(df)
    mixed_df = pd.concat(mixed_dfs)
    mixed_df['Margin'] = 'Mixed'

    intact_dfs = []
    for filename in os.listdir(in_dir + '/' + 'intact'):
        filepath = in_dir + '/' + 'intact' + '/' + filename
        df = read_tsv(filepath, set_categories=False, sep=',')
        df['Sample'] = filename
        intact_dfs.append(df)
    intact_df = pd.concat(intact_dfs)
    intact_df['Margin'] = 'Intact'

    df = pd.concat([mixed_df, intact_df])

    # Set necessary columns as categorical to later retrieve distinct category codes
    df.Sample = pd.Categorical(df.Sample)
    df.Pathology = pd.Categorical(df.Pathology)
    df['Phenotype'] = pd.Categorical(df.ManualPhenotype)
    df.Margin = pd.Categorical(df.Margin)
    df.HLA1_FUNCTIONAL_threeclass = pd.Categorical(df.HLA_label)
    df['Y'] = df.Y

    return df

def read_disease_samples(in_dir):
    disease1_dfs = []
    dir = os.path.join(in_dir, 'Anon_Group1')
    for filename in os.listdir(dir):
        filepath = os.path.join(dir, filename)
        df = read_tsv(filepath, set_categories=False, sep='\t')
        df['Sample'] = filename + '_g1'
        sample_list = []
        for col in df.columns[2:-1]:
            samples = df[df[col] == 'pos']
            samples['Phenotype'] = col
            sample_list.append(samples)
        df = pd.concat(sample_list)
        df = df.drop(columns=df.columns[2:-2])
        disease1_dfs.append(df)
    disease1_df = pd.concat(disease1_dfs)
    disease1_df['Disease'] = 'Group1'

    disease2_dfs = []
    dir = os.path.join(in_dir, 'Anon_Group2')
    for filename in os.listdir(dir):
        filepath = os.path.join(dir, filename)
        df = read_tsv(filepath, set_categories=False, sep='\t')
        df['Sample'] = filename + '_g2'
        sample_list = []
        for col in df.columns[2:-1]:
            samples = df[df[col] == 'pos']
            samples['Phenotype'] = col
            sample_list.append(samples)
        df = pd.concat(sample_list)
        df = df.drop(columns=df.columns[2:-2])
        disease2_dfs.append(df)
    disease2_df = pd.concat(disease2_dfs)
    disease2_df['Disease'] = 'Group2'

    df = pd.concat([disease1_df, disease2_df])

    # Set necessary columns as categorical to later retrieve distinct category codes
    df = df.rename(columns={'Cell.X.Position':'X', 'Cell.Y.Position':'Y'})
    df.Sample = pd.Categorical(df.Sample)
    df.Phenotype = pd.Categorical(df.Phenotype)
    df.Disease = pd.Categorical(df.Disease)

    return df

# def read_tumor_core_samples(in_file, tumor_core_file):

def read_dataset(in_file, sub_path, dataset = None):
    in_file = path + in_file
    sub_path = path + sub_path
    if dataset is None:
        data = pd.read_csv(sub_path, sep=',', header=0, low_memory=False)
        df = read_tsv(in_file, set_categories=False)

    else:
        data = pd.read_csv(sub_path, header=0, low_memory=False)
        df = read_tsv(in_file, set_categories=False, sep=',')

    core_data = pd.DataFrame()
    for sample in data.Sample:
        core_data = core_data.append(df[df.Sample == sample])

    # df = pd.concat([responder_samples, non_responder_samples])
    df = core_data
    df = df[(df['Phenotype'] != "Unclassified") & (df['Phenotype'] != "Potential Artifact")]

    # Set necessary columns as categorical to later retrieve distinct category codes
    df.Sample = pd.Categorical(df.Sample)
    df.Phenotype = pd.Categorical(df.Phenotype)
    # df.Tumor_Core = pd.Categorical(df.Tumor_Core)
    df.Status = pd.Categorical(df.Status)

    return df

def sample_region(dataset, index, label_name):
    '''
    Samples a region (POV) in the dataset.

    Input:
        dataset:    DataFrame consisting of the input data.
        index:      Index of the distinct region to sample.
    '''
    data_at_idx = dataset[dataset.Sample.cat.codes == index]
    label = (int)(data_at_idx[label_name].cat.codes.iloc[0])

    features_at_idx = np.vstack(
        (np.array(data_at_idx.X), 
        np.array(data_at_idx.Y), 
        np.array(data_at_idx.Phenotype.cat.codes))
    ).T

    return torch.FloatTensor(features_at_idx), label

def filter_margins(df):
    mixed = {
        'Mel10': [
            'region_001', 'region_002', 'region_004', 'region_006', 'region_007', 'region_008',
            'region_009', 'region_010', 'region_021', 'region_022', 'region_023', 'region_024',
            'region_025', 'region_026',
        ],
        'Mel12': [

        ],
        'Mel14': [
            'region_003', 'region_004', 'region_005', 'region_007', 'region_008', 'region_011',
            'region_012'
        ]
    }
    intact = {
        'Mel10': [
            'region_005',
        ],
        'Mel12': [
            'region_001', 'region_002', 'region_003', 'region_004', 'region_005', 'region_009',
            'region_011', 'region_012', 'region_022', 'region_023', 'region_024', 'region_025',
            'region_026', 'region_031', 'region_033'
        ],
        'Mel14': [
            'region_001', 'region_002', 'region_006', 'region_009', 'region_010', 'region_013',
            'region_032'
        ]
    }
    margins = []
    for point in df.Sample:
        mixed_label = [mixed_label for mixed_label in mixed if mixed_label in point]
        intact_label = [intact_label for intact_label in intact if intact_label in point]
        if mixed_label:
            mixed_region = [mixed_region for mixed_region in mixed[mixed_label[0]] if mixed_region in point]
            if mixed_region:
                margins.append('Mixed')
                continue
        if intact_label:
            intact_region = [intact_region for intact_region in intact[intact_label[0]] if intact_region in point]
            if intact_region:
                margins.append('Intact')
                continue
        margins.append('Unclassified')
    df['Margin'] = margins
    filtered_df = df[df['Margin'] != 'Unclassified']
    filtered_df.Margin = pd.Categorical(filtered_df.Margin)

    return filtered_df

class PathologyDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset, num_points=None, transforms=None, transformed_samples=1, label_name='Margin', plot_data=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        dataset.Sample = dataset.Sample.cat.remove_unused_categories()
        self.dataset = dataset
        self.transforms = transforms
        self.transformed_samples = transformed_samples
        self.num_points = num_points
        self.label_name = label_name
        self.plot_data = plot_data

    def __len__(self):
        return len(self.dataset.Sample.cat.categories) * self.transformed_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = np.int(idx / self.transformed_samples) # Rescale idx into the df shape
        sample, label = sample_region(self.dataset, index=idx, label_name=self.label_name)

        ## NOTE: For sanity checking
        if self.plot_data:
            data_at_idx = self.dataset[self.dataset.Sample.cat.codes == idx]
            plot_data.plot_data(sample, label, data_at_idx, n_dist=50)
            print()

        if self.transforms:
            sample = self.transforms(sample)
        
        if self.num_points != None and self.num_points < len(sample):
            indices = np.random.choice(len(sample), self.num_points, replace=False)
            sample = sample[indices] # Random sampling

        return sample, label
    
def collate_fn_pad(batch):
    '''
    Zero-pads batch of variable length, so that DataLoader doesn't explode
    '''
    seq = [a_tuple[0] for a_tuple in batch]
    labels = torch.tensor([a_tuple[1] for a_tuple in batch])
    lengths = torch.tensor([ t.shape[0] for t in seq ]).to(device)
    seq = torch.nn.utils.rnn.pad_sequence(seq, padding_value=-1)

    return seq, labels, lengths
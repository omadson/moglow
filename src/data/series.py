import numpy as np
from torch.utils.data import Dataset

class SeriesDataset(Dataset):
    """
    Trajectories dataset. 
    """

    def __init__(self, df, data_columns, label_column, serie_column, seqlen, dropout):
        """
        Args:
        data: dataframe with 
        seqlen: number of autoregressive body poses and previous control values
        n_lookahead: number of future control-values
        dropout: (0-1) dropout probability for previous poses
        """
        self.seqlen = seqlen
        self.dropout=dropout
        # seqlen_control = seqlen + 1

        aggregated_series = (
            df
            .groupby(serie_column)
            .agg(list)
            .reset_index()
        )
        self.labels = (
            aggregated_series
            .apply(lambda x: x[label_column][0], axis=1)
            .values
        )
        serie_example_label = aggregated_series[serie_column][0]

        n_samples = aggregated_series.shape[0]
        n_frames = df.query(f'{serie_column} == @serie_example_label').shape[0]
        n_features = len(data_columns)

        data = df[data_columns].values.reshape(n_samples, n_frames, n_features)
                    
        # Joint positions for n previous frames
        autoreg = self.concat_sequence(self.seqlen, data[:,:n_frames-1,:])
                    
        # Control for n previous frames + current frame
        # control = self.concat_sequence(seqlen_control, control_data)

        # conditioning
        
        print("autoreg:" + str(autoreg.shape))
        # print("control:" + str(control.shape))      
        # new_cond = np.concatenate((autoreg,control),axis=2)

        # joint positions for the current frame
        x_start = seqlen
        new_x = self.concat_sequence(1, data[:,x_start:n_frames,:])
        self.x = new_x
        # self.cond = new_cond
        self.cond = autoreg
        
        #TODO TEMP swap C and T axis to match existing implementation
        self.x = np.swapaxes(self.x, 1, 2)
        self.cond = np.swapaxes(self.cond, 1, 2)
        
        print("self.x:" + str(self.x.shape))        
        print("self.cond:" + str(self.cond.shape))
        
    def n_channels(self):
        return self.x.shape[1], self.cond.shape[1]
		
    def concat_sequence(self, seqlen, data):
        """ 
        Concatenates a sequence of features to one.
        """
        nn,n_timesteps,n_feats = data.shape
        L = n_timesteps-(seqlen-1)
        inds = np.zeros((L, seqlen)).astype(int)

        #create indices for the sequences we want
        rng = np.arange(0, n_timesteps)
        for ii in range(0,seqlen):  
            inds[:, ii] = rng[ii:(n_timesteps-(seqlen-ii-1))]

        #slice each sample into L sequences and store as new samples 
        cc = data[:,inds,:].copy()

        #print ("cc: " + str(cc.shape))

        #reshape all timesteps and features into one dimention per sample
        dd = cc.reshape((nn, L, seqlen*n_feats))
        #print ("dd: " + str(dd.shape))
        return dd
                                                                                                                               
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        """
        Returns poses and conditioning.
        If data-dropout sould be applied, a random selection of the previous poses is masked.
        The control is not masked
        """
        
        if self.dropout > 0.:
            n_feats, tt = self.x[idx,:,:].shape
            cond_masked = self.cond[idx,:,:].copy()
            
            keep_pose = np.random.rand(self.seqlen, tt) < (1-self.dropout)

            #print(keep_pose)
            n_cond = cond_masked.shape[0]-(n_feats*self.seqlen)
            mask_cond = np.full((n_cond, tt), True)

            mask = np.repeat(keep_pose, n_feats, axis = 0)
            mask = np.concatenate((mask, mask_cond), axis=0)
            #print(mask)

            cond_masked = cond_masked*mask
            sample = {'x': self.x[idx,:,:], 'cond': cond_masked}
        else:
            sample = {
                'x': self.x[idx,:,:],
                'cond': self.cond[idx,:,:],
                'label': self.labels[idx]
            }
            
        return sample
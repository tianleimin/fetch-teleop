#!/usr/bin/env python
# coding: utf-8

# Preprocess collected csv data to use in offline RL

# imports
import pandas as pd
import numpy as np
from itertools import product


# function for preprocessing the raw csv, including computing combined rewards
# takes in a str of the name of the raw csv to be processed (without '.csv')
def preprocess(raw_data):
    # data type of non-numerical columns in the raw csv files
    int_cols = ['episode']
    cat_cols = ['status', 'handover quality', 'handover type', 'arm status', 'base status', 'handover status']
    # coded actions for arm status and handover status action pairs
    action_encoding = {
        'STATIONARY+MIDDLE' : 0,
        'REACHING+MIDDLE' : 1,
        'TUCKING+MIDDLE' : 2,
        'STATIONARY+LEFT' : 3,
        'REACHING+LEFT' : 4,
        'TUCKING+LEFT' : 5,
        'STATIONARY+RIGHT' : 6,
        'REACHING+RIGHT' : 7,
        'TUCKING+RIGHT' : 8,
    }
    # rename header of preprocessed csvs to use with the Minerva UI
    new_header = ['episode','observation:0','observation:1','observation:2','observation:3','observation:4','observation:5','observation:6','observation:7','observation:8','observation:9','observation:10','observation:11','observation:12','observation:13','observation:14','observation:15','observation:16','observation:17','observation:18','observation:19','observation:20','observation:21','observation:22','observation:23','observation:24','observation:25','observation:26','observation:27','observation:28','observation:29','observation:30','observation:31','observation:32','observation:33','observation:34','observation:35','observation:36','observation:37','observation:38','observation:39','observation:40','observation:41','observation:42','observation:43','observation:44','observation:45','observation:46','observation:47','observation:48','observation:49','observation:50','observation:51','observation:52','observation:53','observation:54','observation:55','observation:56','observation:57','observation:58','observation:59','observation:60','observation:61','observation:62','observation:63','observation:64','observation:65','observation:66','observation:67','observation:68','observation:69','observation:70','observation:71','observation:72','observation:73','observation:74','observation:75','observation:76','observation:77','observation:78','observation:79','observation:80','observation:81','observation:82','observation:83','observation:84','observation:85','observation:86','observation:87','observation:88','observation:89','observation:90','observation:91','observation:92','observation:93','observation:94','observation:95','observation:96','observation:97','observation:98','observation:99','observation:100','action:0','reward']

    # read csv into dataframe
    df = pd.read_csv((raw_data + '.csv'), header = 0)
    # specify data type of columns
    for column in df:
        if column in int_cols:
            df[column] = df[column].astype('int')
        elif column in cat_cols:
            df[column] = df[column].astype('category')
        else:
            df[column] = df[column].astype('float')
    
    # fill missing values in the manually entered handover quality and type columns
    df['handover quality'] = df['handover quality'].cat.add_categories('NEUTRAL')
    df['handover quality'].fillna('NEUTRAL', inplace =True)
    df['handover type'] = df['handover type'].cat.add_categories('NEITHER')
    df['handover type'].fillna('NEITHER', inplace =True) 

    # normalise categorical emotion prediction values
    # categorical emotion values are network weights at output layer
    # the category label shown in GUI visualisation is the argmax of all 8 classes' weights (e.g., argmax(-8,-3) is -3)
    # normalise categorical emotion values to [0,1] by (60+x)/60 as the min is 60
    df.iloc[:,28:36] = df.iloc[:,28:36].apply(lambda x: (60+x)/ 60, axis=0) # OAK-D emotions
    df.iloc[:,38:46] = df.iloc[:,38:46].apply(lambda x: (60+x)/ 60, axis=0) # Fetch emotions
    
    # replace Fetch predictions with OAK-D predictions when it's not facing the participant
    # use status label to find when the robot is facing the participant
    facing_participant = ['TO PARTICIPANT', 'PARTICIPANT HANDOVER']
    not_facing_participant = ['TO OPERATOR', 'ROTATE TO OPERATOR', 'ROTATE TO PARTICIPANT', 'OPERATOR HANDOVER']
    emo_list = ['neutral','happy','sad','surprise','fear','disgust','anger','contempt','valence','arousal']
    #df1 = df.iloc[30020:30070]
    # replace Fetch emotions with OAK-D emotions
    for index, row in df.iterrows():
        if row['status'] in not_facing_participant:
            for emo in emo_list:
                df.loc[index,str(emo+' (fetch)')] = df.loc[index,str(emo+' (global)')]
        else:
            pass
    
    # state observations are OAK-D pose estimations in columns [48:148] and an added task progress (numerical)
    df['task progress'] = df['episode'] / max(df['episode'])
    # state observation in a new dataframe
    df_state = df.iloc[:,48:149]
    
    # actions are categorical values in the 'arm status' and 'handover status' columns
    # 'base status' excluded for simplicity
    # create a new column of action pairs
    df['action pair'] = df['arm status'].astype('str') + '+' + df['handover status'].astype('str')
    df['action pair'] = df['action pair'].astype('category')
    # code action pairs into IDs
    df['action pair ID'] = df['action pair'].map(action_encoding)
    df['action pair ID'] = df['action pair ID'].astype('int')
    
    # emotional reward
    # OAK-D categorical emotion sum, there are 8 emotioanl classes in total
    df['emotion (global)'] = (df['neutral (global)'] + df['happy (global)'] + df['contempt (global)'])/3 \
                             - 0.2 * (df['sad (global)'] + df['surprise (global)'] + df['fear (global)'] \
                             + df['disgust (global)'] + df['anger (global)'])
    # Fetch categorical emotion sum
    df['emotion (fetch)'] = (df['neutral (fetch)'] + df['happy (fetch)'] + df['contempt (fetch)'])/3 \
                             - 0.2 * (df['sad (fetch)'] + df['surprise (fetch)'] + df['fear (fetch)'] \
                             + df['disgust (fetch)'] + df['anger (fetch)'])

    # combined emotional reward
    df['emotional reward'] = (df['emotion (global)'] + df['arousal (global)'] + df['valence (global)'] \
                            + df['emotion (fetch)'] + df['arousal (fetch)'] + df['valence (fetch)'])/6
    
    # handover quality reward
    # replacing categorical labels with values
    df['handover reward'] = df['handover quality'].map({'GOOD':1.0, 'BAD':-1.0, 'NEUTRAL':0.0})
    
    # combined reward
    df['combined reward'] = df['emotional reward'] + df['handover reward'].astype('float')
    
    # combined reward but the handover quality is treated as an episode reward
    # get step index within each episode
    df['step'] = np.arange(len(df))
    ep_start_step = df['step'][(np.diff(df['episode'].values, prepend=-1) > 0)].values
    df['offset'] = ep_start_step[df['episode'].values]
    df.loc[:,'step'] = df['step'] - df['offset'] + 1
    # create a step progress fraction within each episode
    max_step_by_episode = df.groupby('episode')['step'].max().values
    df['fraction'] = df['step'] / np.repeat(max_step_by_episode, max_step_by_episode)
    # add handover quality reward scaled by step progress fraction so that later steps get more of the episodic reward
    df['combined reward async'] = df['emotional reward'] + df['handover reward'].astype('float')*df['fraction']
    
    # only keeping columns relevant
    df_time = df[['time (s)', 'episode', 'step']]
    df_tidy = pd.concat([df_time, df_state, df['action pair ID'], df['combined reward async']], axis=1)
    df_minerva = pd.concat([df['episode'], df_state, df['action pair ID'], df['combined reward async']], axis=1)
    
    # only use part of the data    
    # State machine for overall robot status:
    # To participant -> Participant handover -> Rotate to operator -> To operator -> Operator handover -> Rotate to participant
    # extract rows with 'status' = 'PARTICIPANT HANDOVER'
    df_seg = df.loc[df['status'] == 'PARTICIPANT HANDOVER']
    # only keeping columns relevant
    df_seg_time = df_seg[['time (s)', 'episode', 'step']]
    df_seg_state = df_seg.iloc[:,48:149]
    df_seg_action = df_seg['action pair ID']
    df_seg_reward = df_seg['combined reward async']
    df_seg_tidy = pd.concat([df_seg_time, df_seg_state, df_seg_action, df_seg_reward], axis=1)
    df_seg_minerva = pd.concat([df_seg['episode'], df_seg_state, df_seg_action, df_seg_reward], axis=1)
    
    
    # save in the format of Minerva UI
    processed_minerva_f = raw_data + '_processed_minerva' + '.csv'
    df_minerva.columns = new_header
    df_minerva.to_csv(processed_minerva_f, index=False)
    print('Preprocessing finished: output file', processed_minerva_f)
    #df_seg_minerva.columns = new_header
    #df_seg_minerva.to_csv(processed_minerva_f, index=False)


# function for combining processed files with an optional episode offset
# takes in a list of all processed csv to be concatenated
def combcsv(participants, ep_offset = False):
    # episode offset for concatenating processed files
    offset = 0
    # list of processed df 
    list_comb = []
    # name of combined dataset file
    combined_csv = 'data/combined_minerva.csv'
    # concatenate
    for participant in participants:
        proc_data_each = 'data/' + participant + '_processed_minerva' + '.csv'
        df_each = pd.read_csv(proc_data_each, header = 0)
        # offset episode counter at the start of current csv to continue after the previous csv
        if ep_offset:
            df_each['episode'] = df_each['episode'] + offset
            offset = max(df_each['episode']) + 1
        # restart episode counter for each participant at 0
        # Minerva distinguishes ep No.0 of P1 and ep No.0 of P2 as different episodes
        else:
            pass
        list_comb.append(df_each)
        print('Read processed data:', participant)
    print('Concatenating...')
    df_comb = pd.concat(list_comb)
    print('Saving concatenated dataset...')
    df_comb.to_csv(combined_csv, index=False)


# main loop
# list of csv files to be processed
participants = ['p1_2022-07-25', 'p2_2022-08-03', 'p3_2022-08-04', 'p4_2022-08-10', 'p5_2022-08-12', 'p6_2022-08-15', 'p7_2022-08-15']
if __name__ == "__main__":
    # preprocess individual participants
    for participant in participants:
        raw_data = 'data/' + participant
        preprocess(raw_data)
    print('All raw data processed.')
    # concatenate preprocessed data into one csv with optional episode count offset
    combcsv(participants, ep_offset = False)
    print('All processed files concatenated.')


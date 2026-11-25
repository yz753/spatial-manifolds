from tabnanny import verbose
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix
import pynapple as nap
from spatial_manifolds.toroidal import *
from spatial_manifolds.behaviour_plots import *
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import distance
from spatial_manifolds.circular_decoder import circular_decoder, cross_validate_decoder, cross_validate_decoder_time, circular_nanmean, circular_nansem
from scipy.stats import gaussian_kde
from astropy.convolution import convolve, Gaussian1DKernel
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.cm as cm
from scipy.signal import find_peaks
from scipy.signal import correlate
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator 
from spatial_manifolds.tuning_scores.grid_score import autocorr2d
from spatial_manifolds.data.curation import curate_clusters
from scipy.stats import zscore
from spatial_manifolds.util import gaussian_filter_nan
from spatial_manifolds.predictive_grid import compute_travel_projected, wrap_list
from spatial_manifolds.behaviour_plots import *
from spatial_manifolds.behaviour_plots import trial_cat_priority
from spatial_manifolds.anaylsis_parameters import *
import hdbscan
import umap
from sklearn.preprocessing import StandardScaler
from spatial_manifolds.anaylsis_parameters import tl, vr_tl, mcvr_tl, bs, time_bs, rm_figsize, disqualifying_brain_areas_for_grid_cells, other_areas
 

import matplotlib as mpl

# ── dataset layout ───────────────────────────────────────────────────────────
# clark2025 replaces the old COHORT12 tree. Layout:
#   {SOURCE}/M{mouse}/D{day}/{session}/sub-M{mouse}_ses-D{day}_typ-{session}_beh.nwb
#   {SOURCE}/M{mouse}/D{day}/{session}/sub-M{mouse}_ses-D{day}_typ-{session}_srt-kilosort4_clusters.npz
# The old tree used sub-M{mouse}_ses-D{day}_typ-{session}_*. Per-cluster anatomy
# now lives outside the dataset tree, hence its own constant.
DEFAULT_SOURCE_PATH = '/Volumes/datasetsSD/Ephys_data/clark2025/'
BRAIN_LOCATIONS_CSV = '/Users/harryclark/Downloads/all_cluster_brain_locations_chris.csv'

mpl.rcParams['font.family'] = 'Arial'


def cell_classification_vr(mouse, day, percentile_threshold=99, source_path=None):
    if source_path is None:
        source_path = DEFAULT_SOURCE_PATH
    session = 'VR'
    vr_folder = f'{source_path}M{mouse}/D{day}/{session}/'
    ramp_path = vr_folder + 'tuning_scores/ramps.parquet'
    speed_path = vr_folder + 'tuning_scores/speed_correlation.parquet'
    spatial_path = vr_folder + 'tuning_scores/spatial_information.parquet'
 
    ramp_table = pd.read_parquet(ramp_path)
    speed_table = pd.read_parquet(speed_path)
    spatial_table = pd.read_parquet(spatial_path)

    ramp_classes = [('+','+'),('+','-'), ('+','/'),('-','+'),('-','-'),('-','/')]

    ramp_cells = pd.DataFrame()
    for class_idx, ramp_class in enumerate(ramp_classes):
        subset_ids1 = ramp_table[(ramp_table['trials'] == 'b+nb')
                                 & (ramp_table['outbound_sign'] == ramp_class[0])
                                ]['cluster_id'].values
        subset_ids2 = ramp_table[(ramp_table['trials'] == 'b+nb')
                                 & (ramp_table['homebound_sign'] == ramp_class[1])
                                 ]['cluster_id'].values
        subset_ids = np.intersect1d(subset_ids1, subset_ids2)
        
        tmp = pd.DataFrame()
        tmp['cluster_id'] = subset_ids
        tmp['ramp_class'] = np.repeat(''.join(ramp_class), len(subset_ids))
        speed_mod = []
        for id in subset_ids:
            speed_mod.append(speed_table[(speed_table['cluster_id'] == id) &
                                         (speed_table['trials'] == 'b+nb') &
                                         (speed_table['context'] == 'rz1')]['sig'].iloc[0])
        tmp['speed_modulated'] = speed_mod
        ramp_cells = pd.concat([ramp_cells, tmp])

    non_spatial_cells = spatial_table.query('sig == False')

    ramp_and_speed_cells = ramp_cells[ramp_cells['speed_modulated'] == True]

    if len(non_spatial_cells)<1:
        non_spatial_cells = pd.DataFrame(columns=spatial_table.columns)
    if len(ramp_cells)<1:
        ramp_cells = pd.DataFrame(columns=speed_table.columns)
    if len(non_spatial_cells)<1:
        ramp_and_speed_cells = pd.DataFrame(columns=speed_table.columns)
    
    return ramp_cells, ramp_and_speed_cells, non_spatial_cells

def cell_classification_anatomy(mouse, day, source_path=None, verbose=True):
    if source_path is None:
        source_path = DEFAULT_SOURCE_PATH
    brain_locations = pd.read_csv(BRAIN_LOCATIONS_CSV)

    _,_,_,_,_,clusters_VR = compute_vr_tcs(mouse, day, source_path=source_path)
    session = 'OF1'
    of1_folder = f'{source_path}M{mouse}/D{day}/{session}/'
    spatial_path = of1_folder + "tuning_scores/shifted_spatial_information.parquet"
    theta_path = of1_folder + "tuning_scores/theta_index.parquet"

    spatial_information_score_of1 = pd.read_parquet(spatial_path)
    theta_index_of1 = pd.read_parquet(theta_path)

    cluster_ids_values = spatial_information_score_of1.query('travel == 0').cluster_id

    MEC_cells = pd.DataFrame(columns=spatial_information_score_of1.columns)
    PARA_cells = pd.DataFrame(columns=spatial_information_score_of1.columns)
    PRE_cells = pd.DataFrame(columns=spatial_information_score_of1.columns)
    VIS_cells = pd.DataFrame(columns=spatial_information_score_of1.columns)
    SUB_cells = pd.DataFrame(columns=spatial_information_score_of1.columns)
    CERE_cells = pd.DataFrame(columns=spatial_information_score_of1.columns)
    other_cells = pd.DataFrame(columns=spatial_information_score_of1.columns)

    print(f'unique brain regions {np.unique(clusters_VR.brain_region)}')
    for index in cluster_ids_values:
        cluster_spatial_information_of1 = spatial_information_score_of1[spatial_information_score_of1.cluster_id==index]
        optimal_lag = cluster_spatial_information_of1.travel.values[np.nanargmax(cluster_spatial_information_of1.spatial_information)]

        brain_region = brain_locations[(brain_locations['mouse']==mouse) & (brain_locations['day']==day) & (brain_locations['cluster_id']==index)]['brain_region'].iloc[0]
        SC_x = brain_locations[(brain_locations['mouse']==mouse) & (brain_locations['day']==day) & (brain_locations['cluster_id']==index)]['coord_SCs_x'].iloc[0]
        SC_y = brain_locations[(brain_locations['mouse']==mouse) & (brain_locations['day']==day) & (brain_locations['cluster_id']==index)]['coord_SCs_y'].iloc[0]
        SC_z = brain_locations[(brain_locations['mouse']==mouse) & (brain_locations['day']==day) & (brain_locations['cluster_id']==index)]['coord_SCs_z'].iloc[0]
        probe_x = brain_locations[(brain_locations['mouse']==mouse) & (brain_locations['day']==day) & (brain_locations['cluster_id']==index)]['unit_location_x'].iloc[0]
        probe_y = brain_locations[(brain_locations['mouse']==mouse) & (brain_locations['day']==day) & (brain_locations['cluster_id']==index)]['unit_location_y'].iloc[0]

        theta_index = theta_index_of1[theta_index_of1.cluster_id==index]['theta_index'].iloc[0]

        cell = cluster_spatial_information_of1[(spatial_information_score_of1.travel == optimal_lag)]
        cell['mouse'] = mouse
        cell['day'] = day
        cell['brain_region'] = brain_region
        cell['optimal_travel_lag'] = optimal_lag
        cell['SC_x'] = SC_x
        cell['SC_y'] = SC_y
        cell['SC_z'] = SC_z
        cell['probe_x'] = probe_x
        cell['probe_y'] = probe_y
        cell['theta_index'] = theta_index
        
        if 'ENT' in brain_region:
            MEC_cells = pd.concat([MEC_cells, cell], ignore_index=True)
        elif 'PAR' in brain_region:
            PARA_cells = pd.concat([PARA_cells, cell], ignore_index=True)
        elif 'PRE' in brain_region:
            PRE_cells = pd.concat([PRE_cells, cell], ignore_index=True)
        elif 'VIS' in brain_region:
            VIS_cells = pd.concat([VIS_cells, cell], ignore_index=True)
        elif 'arb' in brain_region:
            CERE_cells = pd.concat([CERE_cells, cell], ignore_index=True)
        elif 'PFL' in brain_region:
            CERE_cells = pd.concat([CERE_cells, cell], ignore_index=True)
        elif 'FL' in brain_region:
            CERE_cells = pd.concat([CERE_cells, cell], ignore_index=True)
        elif 'root' in brain_region:
            CERE_cells = pd.concat([CERE_cells, cell], ignore_index=True)
        elif 'SIM' in brain_region:
            CERE_cells = pd.concat([CERE_cells, cell], ignore_index=True)
        elif 'SUB' in brain_region:
            SUB_cells = pd.concat([SUB_cells, cell], ignore_index=True)
        else:
            other_cells = pd.concat([other_cells, cell], ignore_index=True)

    if verbose:     
        print(f'there are {len(MEC_cells)} MEC cells')
        print(f'there are {len(PARA_cells)} PARA cells')
        print(f'there are {len(PRE_cells)} PRE cells')
        print(f'there are {len(VIS_cells)} VIS cells')
        print(f'there are {len(CERE_cells)} CERE cells')
        print(f'there are {len(other_cells)} other cells')
        print(f'there are {len(SUB_cells)} SUB cells')

    all_cells = pd.concat([MEC_cells, PARA_cells, PRE_cells, SUB_cells, VIS_cells, CERE_cells, other_cells], ignore_index=True)
    print(f'there are {len(all_cells)} all cells')

    return MEC_cells, PARA_cells, PRE_cells, SUB_cells, VIS_cells, CERE_cells, other_cells, all_cells



def classify_cells_both_sessions(mouse, day, percentile_threshold=95, source_path=None, verbose=False,
                                 disqualifying_regions=disqualifying_brain_areas_for_grid_cells, 
                                 use_optimal_travel=True, get_extra_info=False):
    """
    Classify cells using both OF1 and OF2 sessions.
    For each session, we attempt to classify cells as grid cells (GCs) or non-grid spatial cells (NGS) using the specified percentile threshold.
    We then combine the classifications from both sessions to create a unified classification scheme:
    - GCs: any cell classified as a GC in either OF1 or OF2.
    - NGS: any cell classified as NGS in either OF1 or OF2, but not classified as GC in either session.
    The function returns three DataFrames:
    - gcs_comb: combined grid cells from both sessions.
    - ngs_comb: combined non-grid spatial cells from both sessions, excluding any cells classified as GCs.
    - all_1: the full classification DataFrame from OF1, which
    """
    
    try:
        gcs_1, ngs_1, _, _, _, all_1 = cell_classification_of1(
            mouse, day, percentile_threshold=percentile_threshold,
            source_path=source_path, verbose=verbose, session='OF1', 
            disqualifying_brain_areas_for_grid_cells=disqualifying_regions,
                disqualifying_brain_areas_for_spatial_cells=disqualifying_regions,
                use_optimal_travel=use_optimal_travel, get_extra_info=get_extra_info
        )
    except FileNotFoundError:
        gcs_1, ngs_1, all_1 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        if verbose:
            print(f"  M{mouse} D{day}: OF1 files not found.")

    try:
        gcs_2, ngs_2, _, _, _, all_2 = cell_classification_of1(
            mouse, day, percentile_threshold=percentile_threshold,
            source_path=source_path, verbose=verbose, session='OF2',
            disqualifying_brain_areas_for_grid_cells=disqualifying_regions,
            disqualifying_brain_areas_for_spatial_cells=disqualifying_regions,
            use_optimal_travel=use_optimal_travel, get_extra_info=get_extra_info
        )
    except FileNotFoundError:
        gcs_2, ngs_2, all_2 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        if verbose:
            print(f"  M{mouse} D{day}: OF2 files not found.")

    # --- Explicit combined ID sets ---
    gc1_ids  = set(gcs_1['cluster_id']) if len(gcs_1)  > 0 else set()
    gc2_ids  = set(gcs_2['cluster_id']) if len(gcs_2)  > 0 else set()
    ngs1_ids = set(ngs_1['cluster_id']) if len(ngs_1)  > 0 else set()
    ngs2_ids = set(ngs_2['cluster_id']) if len(ngs_2)  > 0 else set()

    combined_gc_ids  = gc1_ids | gc2_ids
    combined_ngs_ids = (ngs1_ids | ngs2_ids) - combined_gc_ids

    # --- Merge OF2 scores into all_1 before extracting combined dataframes ---
    # Cells classified by OF2 would otherwise carry their (potentially negative) OF1 scores.
    if len(all_1) > 0:
        all_1 = all_1.copy()
        all_1['grid_score_of1'] = all_1['grid_score']
        all_1['spatial_information_score_of1'] = all_1['spatial_information_score']
        all_1['field_spacing_of1'] = all_1.get('field_spacing', pd.NA)
        all_1['orientation_of1'] = all_1.get('orientation', pd.NA)

        if len(all_2) > 0:
            of2_cols = ['grid_score', 'spatial_information_score']
            of2_rename = {
                'grid_score': 'grid_score_of2',
                'spatial_information_score': 'spatial_information_score_of2',
            }
            if 'field_spacing' in all_2.columns:
                of2_cols.append('field_spacing')
                of2_rename['field_spacing'] = 'field_spacing_of2'
            if 'orientation' in all_2.columns:
                of2_cols.append('orientation')
                of2_rename['orientation'] = 'orientation_of2'
            of2_lookup = all_2.set_index('cluster_id')[of2_cols].rename(columns=of2_rename)
            all_1 = all_1.join(of2_lookup, on='cluster_id')
        else:
            all_1['grid_score_of2'] = pd.NA
            all_1['spatial_information_score_of2'] = pd.NA
            all_1['field_spacing_of2'] = pd.NA
            all_1['orientation_of2'] = pd.NA

        all_1['grid_score_best'] = all_1[['grid_score_of1', 'grid_score_of2']].max(axis=1)
        all_1['spatial_information_score_best'] = all_1[['spatial_information_score_of1', 'spatial_information_score_of2']].max(axis=1)

    # --- Combined scheme: select rows from all_1 and annotate classification session ---
    if len(all_1) > 0:
        gcs_comb = all_1[all_1['cluster_id'].isin(combined_gc_ids)].copy().reset_index(drop=True)
        ngs_comb = all_1[all_1['cluster_id'].isin(combined_ngs_ids)].copy().reset_index(drop=True)

        # 'classified_by': both/OF1/OF2 depending on which sessions qualify the cell
        gcs_comb['classified_by'] = gcs_comb['cluster_id'].apply(
            lambda cid: 'both' if (cid in gc1_ids and cid in gc2_ids)
                        else ('OF1' if cid in gc1_ids else 'OF2')
        )
        ngs_comb['classified_by'] = ngs_comb['cluster_id'].apply(
            lambda cid: 'both' if (cid in ngs1_ids and cid in ngs2_ids)
                        else ('OF1' if cid in ngs1_ids else 'OF2')
        )
    else:
        gcs_comb = pd.DataFrame()
        ngs_comb = pd.DataFrame()

    return gcs_comb, ngs_comb, all_1



def cell_classification_of1(mouse, day, percentile_threshold=95, source_path=None, 
                            disqualifying_brain_areas_for_grid_cells=disqualifying_brain_areas_for_grid_cells,
                            disqualifying_brain_areas_for_spatial_cells=disqualifying_brain_areas_for_grid_cells, 
                            use_optimal_travel=True, get_extra_info=False, verbose=True, session='OF1'):
    if source_path is None:
        source_path = DEFAULT_SOURCE_PATH
    _,_,_,_,_,clusters_VR = compute_vr_tcs(mouse, day, source_path=source_path)
    last_ephys_time_bin = clusters_VR[clusters_VR.index[0]].count(bin_size=time_bs, time_units = 'ms').index[-1]

    brain_locations = pd.read_csv(BRAIN_LOCATIONS_CSV)
    of1_folder = f'{source_path}M{mouse}/D{day}/{session}/'
    shifted_grid_path = of1_folder + "tuning_scores/shifted_grid_score.parquet"
    shifted_grid_path = of1_folder + "tuning_scores/shifted_grid_score_manual_smooth.parquet"

    spatial_path = of1_folder + "tuning_scores/shifted_spatial_information.parquet"
    spatial_path = of1_folder + "tuning_scores/shifted_spatial_information_manual_smooth.parquet"

    speed_path = of1_folder + "tuning_scores/shifted_speed_correlation.parquet"
    theta_path = of1_folder + "tuning_scores/theta_index.parquet"
    
    head_direction_path = of1_folder + "tuning_scores/shifted_hd_information.parquet"
    
    clusters_OF1 = nap.load_file(of1_folder + f"sub-M{mouse}_ses-D{day}_typ-{session}_srt-kilosort4_clusters.npz")
    beh_OF1 = nap.load_file(of1_folder + f"sub-M{mouse}_ses-D{day}_typ-{session}_beh.nwb")

    if os.path.exists(shifted_grid_path):
        shifted_grid_scores_of1 = pd.read_parquet(shifted_grid_path)
    else:
        shifted_grid_scores_of1 = pd.read_parquet(of1_folder + "tuning_scores/shifted_grid_score.parquet")

    if os.path.exists(spatial_path):
        spatial_information_score_of1 = pd.read_parquet(spatial_path)
    else:
        spatial_information_score_of1 = pd.read_parquet(of1_folder + "tuning_scores/shifted_spatial_information.parquet")

    if os.path.exists(head_direction_path):
        head_direction_of1 = pd.read_parquet(head_direction_path)
    else:
        head_direction_of1 = pd.read_parquet(of1_folder + "tuning_scores/shifted_hd_mean_vector_length.parquet")
        head_direction_of1['hd_information'] = head_direction_of1['mean_vector_length'] 
        
    shifted_speed_score_of1 = pd.read_parquet(speed_path)
    theta_index_of1 = pd.read_parquet(theta_path)
    
    
    cluster_ids_values = shifted_grid_scores_of1.query('travel == 0').cluster_id

    non_grid_cells = pd.DataFrame(columns=shifted_grid_scores_of1.columns)
    grid_cells = pd.DataFrame(columns=shifted_grid_scores_of1.columns)
    non_spatial_cells = pd.DataFrame(columns=shifted_grid_scores_of1.columns)
    speed_cells = pd.DataFrame(columns=shifted_grid_scores_of1.columns)
    cells = pd.DataFrame(columns=shifted_grid_scores_of1.columns)

    # estimate optimal travel using spatial information of all retrohippocampal cells
    travel = np.arange(-50, 50, 2)
    travel_at_max = []
    for id in cluster_ids_values:
        id_scores = np.array(spatial_information_score_of1[spatial_information_score_of1.cluster_id == id].spatial_information)
        id_travels = np.array(spatial_information_score_of1[spatial_information_score_of1.cluster_id == id].travel)

        brain_region = brain_locations[(brain_locations['mouse']==mouse) & (brain_locations['day']==day) & (brain_locations['cluster_id']==id)]['brain_region'].iloc[0]
        if 'ENT' in brain_region or 'PAR' in brain_region or 'PRE' in brain_region:
            travel_at_max.append(id_travels[np.nanargmax(id_scores)])

    if len(travel_at_max) > 3:
        kde = gaussian_kde(travel_at_max, bw_method=0.3) # Adjust bw_method for smoothing
        x = np.linspace(-50, 50, 1000)
        kde_values = kde(x)
        optimal_travel = x[np.argmax(kde_values)]
        if verbose:
            print(f'optimal travel lag is {optimal_travel} cm based on kde of travel at max spatial information')
        # don't allow optimal_travel to be less than -30
        if optimal_travel < -30:
            optimal_travel = 0  
    else:
        optimal_travel = 0

    if use_optimal_travel:
        if verbose:
            print(f'using travel lag of {optimal_travel} cm')
    else:
        if verbose:
            print(f'using travel lag of 0 cm')
        optimal_travel = 0
    

    # now loop through the clusters and classify them
    for index in cluster_ids_values:
        brain_region = brain_locations[(brain_locations['mouse']==mouse) & (brain_locations['day']==day) & (brain_locations['cluster_id']==index)]['brain_region'].iloc[0]
        SC_x = brain_locations[(brain_locations['mouse']==mouse) & (brain_locations['day']==day) & (brain_locations['cluster_id']==index)]['coord_SCs_x'].iloc[0]
        SC_y = brain_locations[(brain_locations['mouse']==mouse) & (brain_locations['day']==day) & (brain_locations['cluster_id']==index)]['coord_SCs_y'].iloc[0]
        SC_z = brain_locations[(brain_locations['mouse']==mouse) & (brain_locations['day']==day) & (brain_locations['cluster_id']==index)]['coord_SCs_z'].iloc[0]
        probe_x = brain_locations[(brain_locations['mouse']==mouse) & (brain_locations['day']==day) & (brain_locations['cluster_id']==index)]['unit_location_x'].iloc[0]
        probe_y = brain_locations[(brain_locations['mouse']==mouse) & (brain_locations['day']==day) & (brain_locations['cluster_id']==index)]['unit_location_y'].iloc[0]
        theta_index = theta_index_of1[theta_index_of1.cluster_id==index]['theta_index'].iloc[0]

        if brain_region not in disqualifying_brain_areas_for_grid_cells:
            if verbose: 
                print(f'brain region {brain_region} qualifies for spatial cells')

            cluster_spatial_information_of1 = spatial_information_score_of1[spatial_information_score_of1.cluster_id==index]
            cluster_shifted_grid_scores_of1 = shifted_grid_scores_of1[shifted_grid_scores_of1.cluster_id==index]
            cluster_speed_correlation_of1 = shifted_speed_score_of1[shifted_speed_score_of1.cluster_id==index]
            cluster_optimal_lag = cluster_spatial_information_of1.travel.values[np.nanargmax(cluster_spatial_information_of1.spatial_information)]
            cluster_optimal_lag_grid_score = cluster_shifted_grid_scores_of1.travel.values[np.nanargmax(cluster_shifted_grid_scores_of1.grid_score)]

            grid_scores = np.array(cluster_shifted_grid_scores_of1[cluster_shifted_grid_scores_of1.cluster_id == index].grid_score).tolist()
            id_scores = np.array(spatial_information_score_of1[spatial_information_score_of1.cluster_id == index].spatial_information).tolist()
            hd_scores = np.array(head_direction_of1[head_direction_of1.cluster_id == index].hd_information).tolist()
            speed_scores = np.array(shifted_speed_score_of1[shifted_speed_score_of1.cluster_id == index].speed_correlation).tolist()
            id_travels = np.array(spatial_information_score_of1[spatial_information_score_of1.cluster_id == index].travel).tolist()
            
            grid_shuffle_scores = np.array(cluster_shifted_grid_scores_of1[cluster_shifted_grid_scores_of1.travel == 0].null_grid_score.iloc[0]).tolist()
            spatial_info_shuffle_scores = np.array(cluster_spatial_information_of1[cluster_spatial_information_of1.travel == 0].null_spatial_information.iloc[0]).tolist()

            percentile99_grid_score_of1 = np.nanpercentile(grid_shuffle_scores, percentile_threshold)
            percentile99_spatial_information_of1 = np.nanpercentile(spatial_info_shuffle_scores, percentile_threshold)

            #percentile99_speed_information_of1_pos = np.nanpercentile(cluster_speed_correlation_of1[cluster_speed_correlation_of1.travel == 0].null_speed_correlation.iloc[0], percentile_threshold)
            #percentile99_speed_information_of1_neg = np.nanpercentile(cluster_speed_correlation_of1[cluster_speed_correlation_of1.travel == 0].null_speed_correlation.iloc[0], 100-percentile_threshold)

            max_grid_score_of1 = cluster_shifted_grid_scores_of1[cluster_shifted_grid_scores_of1['travel'] == np.round(optimal_travel)]['grid_score'].iloc[0]
            zero_lag_grid_score_of1 = cluster_shifted_grid_scores_of1[cluster_shifted_grid_scores_of1['travel'] == 0]['grid_score'].iloc[0]
            spatial_info = cluster_spatial_information_of1[cluster_spatial_information_of1['travel'] == np.round(optimal_travel)]['spatial_information'].iloc[0]
            spatial_info_no_lag = cluster_spatial_information_of1[cluster_spatial_information_of1['travel'] == 0]['spatial_information'].iloc[0]

            speed_correlation = cluster_speed_correlation_of1[cluster_speed_correlation_of1['travel'] == 0]['speed_correlation'].iloc[0]
            head_direction_score = head_direction_of1[(head_direction_of1.cluster_id == index) & (head_direction_of1['travel'] == 0)]['hd_information'].iloc[0]

            cell = cluster_shifted_grid_scores_of1[cluster_shifted_grid_scores_of1['travel'] == np.round(optimal_travel)]

            if len(cell) == 0:
                if verbose:
                    print(f'cell is empty for cluster id {index}')
                    print(f'max_grid_score_of1 is {max_grid_score_of1}')

            cell = cell.reset_index(drop=True) # reset index to avoid issues with concatenation
            cell['mouse'] = mouse
            cell['day'] = day
            cell['brain_region'] = brain_region
            cell['session_travel_lag'] = optimal_travel
            cell['optimal_travel_lag'] = cluster_optimal_lag
            cell['optimal_travel_lag_grid_score'] = cluster_optimal_lag_grid_score
            cell['session_travel_lag_grid_score'] = max_grid_score_of1 
            cell['zero_lag_grid_score'] = zero_lag_grid_score_of1
            cell['spatial_information_score'] = spatial_info
            cell['spatial_information_score_no_lag'] = spatial_info_no_lag
            cell['SC_x'] = SC_x
            cell['SC_y'] = SC_y
            cell['SC_z'] = SC_z
            cell['probe_x'] = probe_x
            cell['probe_y'] = probe_y
            cell['theta_index'] = theta_index
            cell['head_direction_score'] = head_direction_score
            cell['speed_score'] = speed_correlation
            cell['firing_rate'] = clusters_VR.firing_rate[index]

            cell['firing_rate_OF1'] = clusters_OF1.rate[index]
            cell['firing_rate_VR'] = clusters_VR.rate[index]


            # print a warning if the firing rates are very different, by a factor of 10 at least
            if (cell['firing_rate_VR'].iloc[0] > 10*cell['firing_rate_OF1'].iloc[0]) or (cell['firing_rate_OF1'].iloc[0] > 10*cell['firing_rate_VR'].iloc[0]):
                #print(f'warning: firing rates are very different for cluster id {index}, VR: {cell["firing_rate_VR"].iloc[0]}, OF1: {cell["firing_rate_OF1"].iloc[0]}')
                disgard = True
            else:
                #print(f'firing rates are similar for cluster id {index}, VR: {cell["firing_rate_VR"].iloc[0]}, OF1: {cell["firing_rate_OF1"].iloc[0]}')
                disgard = False

            if get_extra_info:
                cell['grid_travel_scores'] = None
                cell = cell.astype({'grid_travel_scores': 'object'})

                cell['grid_shuffles_scores'] = None
                cell = cell.astype({'grid_shuffles_scores': 'object'})
                
                cell['head_direction_travel_scores'] = None
                cell = cell.astype({'head_direction_travel_scores': 'object'})
         
                cell['spatial_information_shuffles_scores'] = None
                cell = cell.astype({'spatial_information_shuffles_scores': 'object'})

                cell['speed_travel_scores'] = None
                cell = cell.astype({'speed_travel_scores': 'object'})

                cell['spatial_information_travel_scores'] = None
                cell = cell.astype({'spatial_information_travel_scores': 'object'})

                cell['spatial_information_travel'] = None
                cell = cell.astype({'spatial_information_travel': 'object'})

                cell.at[0, 'grid_travel_scores'] = grid_scores
                cell.at[0, 'grid_shuffles_scores'] = grid_shuffle_scores
                cell.at[0, 'head_direction_travel_scores'] = hd_scores
                cell.at[0, 'spatial_information_shuffles_scores'] = spatial_info_shuffle_scores
                cell.at[0, 'spatial_information_travel_scores'] = id_scores
                cell.at[0, 'speed_travel_scores'] = speed_scores
                cell.at[0, 'spatial_information_travel'] = id_travels

            if (max_grid_score_of1 > percentile99_grid_score_of1) and (spatial_info > percentile99_spatial_information_of1) and (not disgard):
                grid_cells = pd.concat([grid_cells, cell], ignore_index=True)
            elif (spatial_info > percentile99_spatial_information_of1) and (brain_region not in disqualifying_brain_areas_for_spatial_cells) and (not disgard):
                non_grid_cells = pd.concat([non_grid_cells, cell], ignore_index=True)
            #elif (speed_correlation > percentile99_speed_information_of1_pos) or (speed_correlation < percentile99_speed_information_of1_neg):
            #    speed_cells = pd.concat([speed_cells, cell], ignore_index=True)
            elif (not disgard):
                non_spatial_cells = pd.concat([non_spatial_cells, cell], ignore_index=True)

            if ((not disgard)):
                cells = pd.concat([cells, cell], ignore_index=True)
        else:
            if verbose:
                print(f'brain region {brain_region} is disqualifying for spatial cells')
    
    all_cells = cells.copy()
    non_grid_and_non_spatial_cells = pd.concat([non_grid_cells, non_spatial_cells], ignore_index=True)

    if verbose:
        print(f'there are {len(non_grid_and_non_spatial_cells)} non_grid and non_spatial_cells')
        print(f'there are {len(grid_cells)} grid_cells')
        print(f'there are {len(non_grid_cells)} non grid spatial cells')
        print(f'there are {len(non_spatial_cells)} non spatial cells')
        print(f'there are {len(speed_cells)} speed cells')
        print(f'there are {len(all_cells)} cells')
        if len(non_grid_cells)>0:
            print(f'for the non-grid spatial cells the unique locations are {np.unique(non_grid_cells.brain_region)}')  
        if len(grid_cells)>0:
            print(f'for the grid cells the unique locations are {np.unique(grid_cells.brain_region)}')

    return grid_cells, non_grid_cells, non_spatial_cells, speed_cells, non_grid_and_non_spatial_cells, all_cells



def HDBSCAN_grid_modules(gcs, all, mouse, day, figpath='', min_cluster_size=None, cluster_selection_epsilon=None,
                         curate_with_vr=True, curate_with_brain_region=True, source_path=None, plot_curate=False,
                         verbose=True, ax=None, show=False):
    if source_path is None:
        source_path = DEFAULT_SOURCE_PATH

    _min_cluster_size = min_cluster_size if min_cluster_size is not None else 3
    _epsilon = cluster_selection_epsilon if cluster_selection_epsilon is not None else 0.3
    
    if len(gcs) <= 1:
        return [], [], None
    
    gcs['field_spacing'] = pd.to_numeric(gcs['field_spacing'], errors='coerce')
    gcs['orientation'] = pd.to_numeric(gcs['orientation'], errors='coerce')
    gcs.dropna(subset=['field_spacing', 'orientation'], inplace=True)

    # Extract and preprocess features
    X = gcs[["field_spacing", "orientation"]].copy()

    # Standard scale the 'field_spacing'
    scaler = StandardScaler()
    X["field_spacing_scaled"] = scaler.fit_transform(X[["field_spacing"]])

    # Cyclic encoding for 'orientation' (range 0 to 60)
    X["orientation_sin"] = np.sin(2 * np.pi * X["orientation"] / 60)
    X["orientation_cos"] = np.cos(2 * np.pi * X["orientation"] / 60)

    # Scale sine and cosine components to balance feature influence
    scale_factor = 1 / np.sqrt(2)
    X["orientation_sin_scaled"] = X["orientation_sin"] * scale_factor
    X["orientation_cos_scaled"] = X["orientation_cos"] * scale_factor

    # Prepare the feature matrix
    features = X[["field_spacing_scaled", 
                  "orientation_sin_scaled", 
                  "orientation_cos_scaled"]]
    original_features = X[["field_spacing", 
                           "orientation"]]
    
    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
                min_cluster_size=_min_cluster_size,
                min_samples=1,
                cluster_selection_epsilon=_epsilon,
                allow_single_cluster=False,
                metric='chebyshev'
            ) 
    module_labels = clusterer.fit_predict(features)

    # if theres only one label, attempt to make a cluster around 10 from the centroid
    if np.unique(module_labels).size == 1 and np.unique(module_labels)[0] == -1:
        if verbose:
            print('Only one cluster found, attempting to create a new cluster around centroid')
        centroid = original_features.median().values # use median to avoid outliers
        distances = distance.cdist([centroid], original_features.values, metric='euclidean')[0]
        close_points = np.where(distances < 10)[0]
        if len(close_points) > _min_cluster_size:
            module_labels[close_points] = 0
            if verbose:
                print(f'Created a new cluster with {len(close_points)} points close to centroid at distance < 10')

    # merge labels if centroids are within 10 units (e.g., cm or degrees)
    centroids = original_features.groupby(module_labels).mean().values
    centroids = np.array(centroids)
    # Start with original labels
    merged_labels = module_labels.copy()
    for i, label_i in enumerate(np.unique(module_labels)):
        for j, label_j in enumerate(np.unique(module_labels)):
            if j <= i or label_i == -1 or label_j == -1:
                continue
            if (distance.euclidean(centroids[i], centroids[j]) < 10):
                if verbose:
                    print(f"Merging labels {label_i} and {label_j} with distance {distance.euclidean(centroids[i], centroids[j])}")
                merged_labels[merged_labels == label_j] = label_i
            else:
                if verbose:
                    print(f"Not merging labels {label_i} and {label_j} with distance {distance.euclidean(centroids[i], centroids[j])}")
    module_labels = merged_labels.copy()
    
    X["cluster"] = module_labels

    if verbose:
        print(f'Found {len(np.unique(module_labels))} modules with HDBSCAN for mouse {mouse} day {day}')
        print(f'for each module , the number of points is: {np.unique(module_labels, return_counts=True)[1]}')
    
    # Plot the results — into a provided axis or a new figure
    if ax is None:
        fig = plt.figure(figsize=(4.2, 3))
        ax_plot = fig.gca()
        standalone = True
    else:
        fig = ax.get_figure()
        ax_plot = ax
        standalone = False

    sns.scatterplot(data=X, x="field_spacing", y="orientation", hue="cluster", palette="tab10",
                    s=25, legend=False, linewidth=0, ax=ax_plot)

    # Highlight unassigned points (label -1)
    ax_plot.scatter(np.array(X['field_spacing'])[module_labels == -1],
                    np.array(X['orientation'])[module_labels == -1], s=21, color='black', label='Unassigned Points')
    ax_plot.scatter(all['field_spacing'], all['orientation'], s=20, color='tab:grey', alpha=0.5, zorder=-1)

    ax_plot.set_ylim(0, 60)
    if standalone:
        ax_plot.set_xlabel('Grid spacing (cm)')
        ax_plot.set_ylabel('Grid orientation ($^\circ$)')
        ax_plot.set_title(f'M{mouse}D{day}  ε={_epsilon}')
        plt.tight_layout()
        if figpath:
            os.makedirs(figpath, exist_ok=True)
            fig.savefig(os.path.join(figpath, f'M{mouse}D{day}_grid_modules_HDBSCAN.pdf'),
                       dpi=200, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

    if np.unique(module_labels).size == 1 and np.unique(module_labels)[0] == -1:
        module_labels[:] = 0  # Assign all points to a single cluster if no clusters were found
        return [], [], fig
    
    # put cluster ids into modules then rearange from smallest spacing to larger
    grid_module_cluster_ids = []
    grid_module_ids = []
    avg_spacings = []
    for mi, module_label in enumerate(np.unique(module_labels[module_labels != -1])):
        grid_ids = np.array(gcs['cluster_id'])
        cells = gcs[np.isin(gcs['cluster_id'], grid_ids[module_labels == module_label])]
        avg_spacings.append(np.nanmean(cells.field_spacing.values))
        grid_module_cluster_ids.append(cells['cluster_id'].tolist())
        grid_module_ids.append(mi)
        if verbose:
            print(f'for module {mi}, there are {len(cells)} cells with average spacing {np.nanmean(cells.field_spacing.values)}')
    grid_module_cluster_ids = [x for _, x in sorted(zip(avg_spacings, grid_module_cluster_ids))]
    grid_module_ids = [x for _, x in sorted(zip(avg_spacings, grid_module_ids))]
          
    return grid_module_ids, grid_module_cluster_ids, fig


def plot_grid_modules_rate_maps(gcs, grid_module_ids, grid_module_cluster_ids, mouse, day, figpath, source_path=None):
    print(mouse, day)
    if source_path is None:
        source_path = DEFAULT_SOURCE_PATH
    session = 'OF1'
    of1_folder = f'{source_path}M{mouse}/D{day}/{session}/'
    shifted_grid_path = of1_folder + "tuning_scores/shifted_grid_score.parquet"
    spatial_path = of1_folder + "tuning_scores/shifted_spatial_information.parquet"
    spikes_path = of1_folder + f"sub-M{mouse}_ses-D{day}_typ-{session}_srt-kilosort4_clusters.npz"
    beh_path = of1_folder + f"sub-M{mouse}_ses-D{day}_typ-{session}_beh.nwb"
    shifted_grid_scores_of1 = pd.read_parquet(shifted_grid_path)
    spatial_information_score_of1 = pd.read_parquet(spatial_path)
    beh_OF = nap.load_file(beh_path)
    clusters_OF = nap.load_file(spikes_path)

    shifted_grid_scores_of1 = shifted_grid_scores_of1.query('travel >= 0')
    spatial_information_score_of1 = spatial_information_score_of1.query('travel >= 0')
    cluster_ids_values = shifted_grid_scores_of1.query('travel == 0').cluster_id

    ncols = 10
    rows_per_module = {mi: int(np.ceil(len(module) / ncols)) for mi, module in zip(grid_module_ids, grid_module_cluster_ids)}
    nrows = sum(rows_per_module.values())+len(grid_module_cluster_ids)+1
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 1*nrows), squeeze=False)
    row_counter = 0
    for mi, module_ids in zip(grid_module_ids, grid_module_cluster_ids):
        cells = gcs[gcs['cluster_id'].isin(module_ids)]
        print(f'for module {mi}, there are {len(cells)} cells')
        counter = 0
        for j in range(rows_per_module[mi]):
            for i in range(ncols):
                if counter < len(cells):
                    index = cells['cluster_id'].values[counter]
                    score = cells['grid_score'].values[counter]
                    cluster_shifted_grid_scores = shifted_grid_scores_of1[shifted_grid_scores_of1.cluster_id==index]
                    travel = cluster_shifted_grid_scores.travel.values[np.nanargmax(cluster_shifted_grid_scores.grid_score)]
                    max_score = cluster_shifted_grid_scores.grid_score.values[np.nanargmax(cluster_shifted_grid_scores.grid_score)]
                    field_spacing = cluster_shifted_grid_scores.field_spacing.values[np.nanargmax(cluster_shifted_grid_scores.grid_score)]
                    
                    tcs = {}    
                    position = np.stack([beh_OF['P_x'], beh_OF['P_y']], axis=1)
                    beh_lag = compute_travel_projected(["P_x", "P_y"], position, position, travel)
                    position_lagged = np.stack([beh_lag['P_x'], beh_lag['P_y']], axis=1)
                    for cell in cells['cluster_id'].values:
                        tc = nap.compute_2d_tuning_curves(nap.TsGroup([clusters_OF[cell]]), position_lagged, nb_bins=(40,40))[0]
                        tc = gaussian_filter_nan(tc[0], sigma=(2.5,2.5))
                        tcs[cell] = tc
                    #ax[row_counter, i].text(0,-2, f'id: {index}, mgs: {np.round(max_score, decimals=1)}', size=7)
                    #ax[row_counter, i].text(0,44, f'fs:{int(field_spacing)}', size=7)
                    ax[row_counter, i].imshow(tcs[index], cmap='jet')
                    counter+=1
            row_counter += 1
        row_counter += 1

    for axi in ax.flatten():
        axi.axis('off')
    plt.tight_layout()
    plt.savefig(f'{figpath}/M{mouse}D{day}_GC_rate_maps_modules.pdf', dpi=1000)
    plt.close()    


def plot_grid_modules(gcs, grid_module_ids, grid_module_cluster_ids, mouse, day,
                      figpath='', source_path=None, session='OF1', ncols=10, verbose=False, show=False):
    """
    Plot OF rate maps for grid cells arranged by module (sorted smallest→largest spacing).

    Parameters
    ----------
    gcs : DataFrame
        Grid cell DataFrame with at least cluster_id and travel columns.
    grid_module_ids : list of int
        Module IDs sorted from smallest to largest average spacing
        (as returned by HDBSCAN_grid_modules).
    grid_module_cluster_ids : list of list of int
        cluster_ids belonging to each module, parallel to grid_module_ids.
    mouse, day : int
        Session identifiers.
    figpath : str
        Directory to save the figure. Pass '' to skip saving.
    session : str
        'OF1' or 'OF2'.
    ncols : int
        Number of columns in the rate-map grid.
    verbose : bool
        Print per-module cell counts.
    """
    if not grid_module_ids:
        print('No modules to plot.')
        return

    # Determine session-level optimal shift from gcs
    optimal_shift = float(gcs['travel'].iloc[0]) if len(gcs) > 0 and 'travel' in gcs.columns else 0

    # Load all rate maps in one pass
    try:
        tcs, _, _, _, _ = compute_of_tcs(
            mouse, day,
            apply_zscore=False,
            apply_guassian_filter=True,
            fill_nans_from_neighbors=True,
            source_path=source_path,
            optimal_shift=optimal_shift,
            session=session,
        )
    except Exception as e:
        print(f'Failed to load {session} rate maps for M{mouse}D{day}: {e}')
        return

    # Build layout: one block of rows per module, separated by a spacer row
    rows_per_module = {
        mi: max(int(np.ceil(len(ids) / ncols)), 1)
        for mi, ids in zip(grid_module_ids, grid_module_cluster_ids)
    }
    n_spacers = max(len(grid_module_ids) - 1, 0)
    nrows = sum(rows_per_module.values()) + n_spacers

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(ncols * 0.5, nrows * 0.55), squeeze=False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    row_cursor = 0
    for loop_idx, (mi, module_cluster_ids) in enumerate(zip(grid_module_ids, grid_module_cluster_ids)):
        module_gcs = gcs[gcs['cluster_id'].isin(module_cluster_ids)]
        if verbose:
            avg_sp = np.nanmean(module_gcs['field_spacing'].values) if 'field_spacing' in module_gcs.columns else float('nan')
            print(f'Module {mi}: {len(module_gcs)} cells, avg spacing {avg_sp:.1f} cm')

        counter = 0
        for row_offset in range(rows_per_module[mi]):
            for col in range(ncols):
                ax = axes[row_cursor + row_offset, col]
                if counter < len(module_cluster_ids):
                    cid = module_cluster_ids[counter]
                    if cid in tcs:
                        ax.imshow(tcs[cid], cmap='viridis', origin='lower')
                    counter += 1
                ax.axis('off')

        row_cursor += rows_per_module[mi]

        # Spacer row between modules
        if loop_idx < len(grid_module_ids) - 1:
            for col in range(ncols):
                axes[row_cursor, col].axis('off')
            row_cursor += 1

    # Turn off any remaining axes
    for r in range(row_cursor, nrows):
        for col in range(ncols):
            axes[r, col].axis('off')

    fig.suptitle(f'Grid modules M{mouse}D{day}  ({session})', fontsize=9)
    plt.tight_layout(rect=[0, 0, 0.3, 0.3])

    if figpath:
        os.makedirs(figpath, exist_ok=True)
        plt.savefig(f'{figpath}GC_rate_maps_modules_M{mouse}D{day}.pdf', dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_grid_modules_umap(gcs, grid_module_ids, grid_module_cluster_ids, mouse, day,
                           figpath='', source_path=None, session='OF1',
                           n_neighbors=15, min_dist=0.1, random_state=42,
                           ax=None, show=False, verbose=True):
    """
    UMAP embedding (2D) of OF spatial rate maps for grid cells, colored by the
    HDBSCAN module label from HDBSCAN_grid_modules — checks whether modules
    found on (field_spacing, orientation) also separate in rate-map shape space.

    Rate maps are computed with the SAME common per-session shift baked into
    `gcs['travel']` (the KDE-derived optimal_travel from
    classify_cells_both_sessions), not compute_of_tcs's default of shift=0.

    Returns (embedding, module_labels, cluster_ids) or (None, None, None) if
    there isn't enough data to embed.
    """
    if not grid_module_ids:
        print('No modules to embed.')
        return None, None, None

    optimal_shift = float(gcs['travel'].iloc[0]) if len(gcs) > 0 and 'travel' in gcs.columns else 0
    if verbose:
        print(f'plot_grid_modules_umap: using common session shift = {optimal_shift} cm (KDE optimal_travel, not 0)')

    try:
        tcs, _, _, _, _ = compute_of_tcs(
            mouse, day,
            apply_zscore=False,
            apply_guassian_filter=True,
            fill_nans_from_neighbors=True,
            source_path=source_path,
            optimal_shift=optimal_shift,
            session=session,
        )
    except Exception as e:
        print(f'Failed to load {session} rate maps for M{mouse}D{day}: {e}')
        return None, None, None

    # cluster_id -> module label (-1 = unassigned by HDBSCAN)
    cid_to_module = {}
    for mi, ids in zip(grid_module_ids, grid_module_cluster_ids):
        for cid in ids:
            cid_to_module[cid] = mi

    cluster_ids = [cid for cid in gcs['cluster_id'].tolist() if cid in tcs]
    if len(cluster_ids) < 4:
        print('Too few grid cells with rate maps for UMAP.')
        return None, None, None

    X = []
    for cid in cluster_ids:
        rm = np.nan_to_num(tcs[cid], nan=0.0).astype(np.float64).ravel()
        rng = rm.max() - rm.min()
        rm = (rm - rm.min()) / rng if rng > 0 else rm * 0
        X.append(rm)
    X = np.stack(X)

    labels = np.array([cid_to_module.get(cid, -1) for cid in cluster_ids])

    reducer = umap.UMAP(n_components=2, n_neighbors=min(n_neighbors, len(cluster_ids) - 1),
                        min_dist=min_dist, random_state=random_state)
    embedding = reducer.fit_transform(X)

    if ax is None:
        fig = plt.figure(figsize=(4.2, 3))
        ax_plot = fig.gca()
        standalone = True
    else:
        fig = ax.get_figure()
        ax_plot = ax
        standalone = False

    palette = plt.get_cmap('tab10').colors
    unassigned = labels == -1
    ax_plot.scatter(embedding[unassigned, 0], embedding[unassigned, 1],
                    s=25, color='black', label='Unassigned', zorder=1)
    for mi in sorted(set(labels[~unassigned].tolist())):
        m = labels == mi
        ax_plot.scatter(embedding[m, 0], embedding[m, 1],
                        s=25, color=palette[mi % len(palette)], label=f'Module {mi}',
                        edgecolors='k', linewidths=0.3, zorder=2)

    if standalone:
        ax_plot.set_xlabel('UMAP 1')
        ax_plot.set_ylabel('UMAP 2')
        ax_plot.set_title(f'M{mouse}D{day}  rate-map UMAP ({session})')
        ax_plot.legend(fontsize=6, frameon=False, ncol=2)
        plt.tight_layout()
        if figpath:
            os.makedirs(figpath, exist_ok=True)
            fig.savefig(os.path.join(figpath, f'M{mouse}D{day}_grid_modules_UMAP_{session}.pdf'),
                       dpi=200, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

    return embedding, labels, cluster_ids


def white_to_hex_cmap(hex_color, name='custom_cmap'):
    return LinearSegmentedColormap.from_list(name, ['#FFFFFF', hex_color])


def compute_vr_tcs_using_expected_spikes(mouse, day, apply_zscore=True, apply_guassian_filter=True, 
                                         source_path=None, bs_t=None, vr_type='VR', expected_spikes=None):
    if source_path is None:
        source_path = DEFAULT_SOURCE_PATH
    if bs_t is None:
        bs_t = time_bs
    if vr_type is 'MCVR':
        tl = mcvr_tl
    else:
        tl = vr_tl

    vr_folder = f'{source_path}M{mouse}/D{day}/{vr_type}/'
    spikes_path = vr_folder + f"sub-M{mouse}_ses-D{day}_typ-{vr_type}_srt-kilosort4_clusters.npz"
    beh_path = vr_folder + f"sub-M{mouse}_ses-D{day}_typ-{vr_type}_beh.nwb"
    beh = nap.load_file(beh_path)
    clusters = nap.load_file(spikes_path)
    #print(f'there are this many clusters before curation {len(clusters)}')
    clusters = curate_clusters(clusters)

    tns = beh['trial_number']
    dt = beh['travel']-((tns[0]-1)*tl)
    n_bins = int(int(((np.ceil(np.nanmax(dt))//tl)+1)*tl)/bs)
    max_bound = int(((np.ceil(np.nanmax(dt))//tl)+1)*tl)
    min_bound = 0
    dt_bins = np.arange(0,max_bound,bs)

    # trick to clip the tc to around the end of the ephys recording
    # take the cell with the highest firing rate, and find the last bin with a spike
    # then work backwards and clip at the end of the last appropriate trials
    tc = nap.compute_1d_tuning_curves(nap.TsGroup([clusters[clusters.index[np.nanargmax(clusters.firing_rate)]]]), 
                                        dt, 
                                        nb_bins=n_bins, 
                                        minmax=[min_bound, max_bound],
                                        ep=beh["moving"])[0]
    
    tc = gaussian_filter(np.nan_to_num(tc).astype(np.float64), sigma=2.5)
    last_ephys_bin = int(np.nonzero(tc)[0][-1] + (tl/bs) - np.nonzero(tc)[0][-1]%(tl/bs))
    last_ephys_time_bin = clusters[clusters.index[0]].count(bin_size=bs_t, time_units = 'ms').index[-1]
    #print(f'last_ephys_bin {last_ephys_bin}')
    #print(f'last_ephys_time_bin {last_ephys_time_bin}')

    # time binned variables for later
    ep = nap.IntervalSet(start=0, end=last_ephys_time_bin, time_units = 's')
    speed_in_time = beh['S'].bin_average(bin_size=bs_t, time_units = 'ms', ep=ep)
    dt_in_time = beh['travel'].bin_average(bin_size=bs_t, time_units = 'ms', ep=ep)-((tns[0]-1)*tl)
    pos_in_time = dt_in_time%tl
    trial_number_in_time = (dt_in_time//tl)+tns[0]

    tcs = {}
    tcs_time = {}
    autocorrs = {}
    for cell in clusters.index:
        if (expected_spikes is not None) and (cell in expected_spikes):
            print(f'using expected spikes for cell {cell}')
            # Generate spike counts for each bin
            spike_counts = np.random.poisson(expected_spikes[cell])

            # Get the bin indices for all spikes
            bin_indices = np.repeat(np.arange(len(spike_counts)), spike_counts)

            # For each spike, add a random offset within the bin to get continuous time
            spike_times = bin_indices * (time_bs/1000) + np.random.uniform(0, (time_bs/1000), size=len(bin_indices))
        else:
            spike_times = clusters[cell]
        
        tc = nap.compute_1d_tuning_curves(nap.TsGroup([spike_times]), 
                                        dt, 
                                        nb_bins=n_bins, 
                                        minmax=[min_bound, max_bound],
                                        ep=beh["moving"])[0]
        tc = np.array(tc)
        tc = np.nan_to_num(tc).astype(np.float64)
        if apply_guassian_filter:
            tc = gaussian_filter(tc, sigma=2.5)
        if apply_zscore:
            tc = zscore(tc)
        tc = tc[:last_ephys_bin] # only want bins with ephys data in it
        tcs[cell] = tc
        
        tc_time = clusters[cell].count(bin_size=bs_t, time_units = 'ms', ep=ep)
        tc_time = np.array(tc_time)
        tc_time = np.nan_to_num(tc_time).astype(np.float64)
        if apply_guassian_filter:
            tc_time = gaussian_filter(tc_time, sigma=2.5) # 
        if apply_zscore:
             tc_time = zscore(tc_time)
        tcs_time[cell] = tc_time

        lags = np.arange(0, 200, 1) # were looking at 10 timesteps back and 10 forward
        autocorr = []
        for lag in lags:
            if lag < 0:
                tc_offset = np.roll(tc, lag)
                tc_offset[lag:] = 0
            elif lag > 0:
                tc_offset = np.roll(tc, lag)
                tc_offset[:lag] = 0
            else:
                tc_offset = tc
            corr = stats.pearsonr(tc, tc_offset)[0]
            autocorr.append(corr)
        autocorr = np.array(autocorr)
        autocorrs[cell] = autocorr

    # drop beh trials from after last ephys bin
    beh_trials = beh['trials']
    beh_trials = beh_trials[:int(last_ephys_bin/(tl/bs))]

    return tcs, tcs_time, autocorrs, last_ephys_bin, beh, clusters



def compute_of_tcs_using_expected_spikes(mouse, day, apply_zscore=True, apply_guassian_filter=True, 
                                        source_path=None, bs_t=None, expected_spikes=None, optimal_shift=0):
    if source_path is None:
        source_path = DEFAULT_SOURCE_PATH
    if bs_t is None:
        bs_t = time_bs

    session = 'OF1'
    of1_folder = f'{source_path}M{mouse}/D{day}/{session}/'
    shifted_grid_path = of1_folder + "tuning_scores/shifted_grid_score.parquet"
    spatial_path = of1_folder + "tuning_scores/shifted_spatial_information.parquet"
    spikes_path = of1_folder + f"sub-M{mouse}_ses-D{day}_typ-{session}_srt-kilosort4_clusters.npz"
    beh_path = of1_folder + f"sub-M{mouse}_ses-D{day}_typ-{session}_beh.nwb"
    shifted_grid_scores_of1 = pd.read_parquet(shifted_grid_path)
    spatial_information_score_of1 = pd.read_parquet(spatial_path)
    beh_OF = nap.load_file(beh_path)
    clusters_OF = nap.load_file(spikes_path)
    clusters_OF = curate_clusters(clusters_OF)

    # trick to clip the tc to around the end of the ephys recording
    last_ephys_time_bin = clusters_OF[clusters_OF.index[0]].count(bin_size=bs_t, time_units = 'ms').index[-1]
    ep = nap.IntervalSet(start=0, end=last_ephys_time_bin, time_units = 's')
 
    tcs = {}
    tcs_time = {}
    for cell in clusters_OF.index:
        if (expected_spikes is not None) and (cell in expected_spikes):
            print(f'using expected spikes for cell {cell}')
            # Generate spike counts for each bin
            spike_counts = np.random.poisson(expected_spikes[cell])

            # Get the bin indices for all spikes
            bin_indices = np.repeat(np.arange(len(spike_counts)), spike_counts)

            # For each spike, add a random offset within the bin to get continuous time
            spike_times = bin_indices * (time_bs/1000) + np.random.uniform(0, (time_bs/1000), size=len(bin_indices))
        else:
            spike_times = clusters_OF[cell]
    
        # plot for cluster on ax
        position = np.stack([beh_OF['P_x'], beh_OF['P_y']], axis=1)
        beh_lag = compute_travel_projected(["P_x", "P_y"], position, position, optimal_shift)
        position_lagged = np.stack([beh_lag['P_x'], beh_lag['P_y']], axis=1)

        tc = nap.compute_2d_tuning_curves(nap.TsGroup([spike_times]),
                                          position_lagged, nb_bins=(40,40), ep=ep)[0]
        if apply_guassian_filter:
            tc = gaussian_filter_nan(tc[0], sigma=(2.5,2.5))
        if apply_zscore:
            tc = zscore(tc)
        tcs[cell] = tc
        tc_time = clusters_OF[cell].count(bin_size=bs_t, time_units = 'ms', ep=ep)
        tc_time = np.array(tc_time)
        tc_time = np.nan_to_num(tc_time).astype(np.float64)
        if apply_guassian_filter:
            tc_time = gaussian_filter(tc_time, sigma=2.5) # 
        if apply_zscore:
             tc_time = zscore(tc_time)
        tcs_time[cell] = tc_time

    return tcs, tcs_time, beh_OF, clusters_OF, ep
 
def get_theta_trace(
    mouse,
    day,
    cluster_id,
    time_bs=10,
    resample_bs=10,
    session_type='VR',
    source_path=None,
    bs_t=None,
    channel=None,
):
    if source_path is None:
        source_path = DEFAULT_SOURCE_PATH
    if bs_t is None:
        bs_t = time_bs
    # Load behaviour and spike data for this session so we can determine
    # the ephys recording length and build the epoch interval.
    session_folder = f'{source_path}M{mouse}/D{day}/{session_type}/'
    beh_path    = session_folder + f"sub-M{mouse}_ses-D{day}_typ-{session_type}_beh.nwb"
    spikes_path = session_folder + f"sub-M{mouse}_ses-D{day}_typ-{session_type}_srt-kilosort4_clusters.npz"
    beh      = nap.load_file(beh_path)
    clusters = nap.load_file(spikes_path)
    clusters = curate_clusters(clusters)

    last_ephys_time_bin = clusters[clusters.index[0]].count(bin_size=bs_t, time_units='ms').index[-1]
    ep = nap.IntervalSet(start=0, end=last_ephys_time_bin, time_units='s')

    # Load the broadband LFP file for this session.
    # The file contains one trace per electrode channel across the whole probe.
    # Path: {source_path}/LFP/M{mouse}/D{day}/{session_type}/sub-M{mouse}_ses-D{day}_typ-{session_type}_lfp.npz
    theta = nap.load_file(f'{source_path}LFP/M{mouse}/D{day}/{session_type}/sub-M{mouse}_ses-D{day}_typ-{session_type}_lfp.npz')

    # Look up which electrode channel is closest to this cell's recording site.
    # channel_id comes from the spike-sorting brain location table and identifies
    # the channel with the largest spike amplitude for this unit, which is used
    # as a proxy for the local LFP at that cell's location.
    channel_arrays = pd.read_csv(BRAIN_LOCATIONS_CSV)
    extrema_channel = channel_arrays[
        (channel_arrays['mouse'] == mouse) &
        (channel_arrays['day'] == day) &
        (channel_arrays['cluster_id'] == cluster_id)
    ]['channel_id'].values[0]

    # Allow overriding the channel if a specific electrode is preferred.
    if channel is not None:
        theta_trace = theta[channel]
    else:
        theta_trace = theta[extrema_channel]

    # Bin LFP at bs_t ms resolution, restricted to the ephys epoch.
    binned = theta_trace.bin_average(bin_size=bs_t, time_units='ms', ep=ep)

    # Optionally resample to a different bin size (e.g. load at 50 ms then
    # downsample to time_bs=10 ms) via linear interpolation.
    if resample_bs is not None and resample_bs != bs_t:
        binned_np = np.array(binned)
        if binned_np.size > 1:
            t_orig = np.arange(len(binned_np)) * (bs_t / 1000.0)
            t_new  = np.arange(t_orig[0], t_orig[-1] + 1e-9, resample_bs / 1000.0)
            binned_np = np.interp(t_new, t_orig, binned_np)
        binned = binned_np
        bs_t = resample_bs

    return binned


def compute_vr_tcs(mouse, day, apply_zscore=True, apply_guassian_filter=True, source_path=None, bs_t=None, vr_type='VR'):
    if source_path is None:
        source_path = DEFAULT_SOURCE_PATH
    if bs_t is None:
        bs_t = time_bs
    if vr_type is 'MCVR':
        tl = mcvr_tl
    else:
        tl = vr_tl

    vr_folder = f'{source_path}M{mouse}/D{day}/{vr_type}/'
    spikes_path = vr_folder + f"sub-M{mouse}_ses-D{day}_typ-{vr_type}_srt-kilosort4_clusters.npz"
    beh_path = vr_folder + f"sub-M{mouse}_ses-D{day}_typ-{vr_type}_beh.nwb"
    beh = nap.load_file(beh_path)
    clusters = nap.load_file(spikes_path)
    #print(f'there are this many clusters before curation {len(clusters)}')
    clusters = curate_clusters(clusters)

    tns = beh['trial_number']
    dt = beh['travel']-((tns[0]-1)*tl)
    n_bins = int(int(((np.ceil(np.nanmax(dt))//tl)+1)*tl)/bs)
    max_bound = int(((np.ceil(np.nanmax(dt))//tl)+1)*tl)
    min_bound = 0
    dt_bins = np.arange(0,max_bound,bs)

    # trick to clip the tc to around the end of the ephys recording
    # take the cell with the highest firing rate, and find the last bin with a spike
    # then work backwards and clip at the end of the last appropriate trials
    tc = nap.compute_1d_tuning_curves(nap.TsGroup([clusters[clusters.index[np.nanargmax(clusters.firing_rate)]]]), 
                                        dt, 
                                        nb_bins=n_bins, 
                                        minmax=[min_bound, max_bound],
                                        ep=beh["moving"])[0]
    
    tc = gaussian_filter(np.nan_to_num(tc).astype(np.float64), sigma=2.5)
    last_ephys_bin = int(np.nonzero(tc)[0][-1] + (tl/bs) - np.nonzero(tc)[0][-1]%(tl/bs))
    last_ephys_time_bin = clusters[clusters.index[0]].count(bin_size=bs_t, time_units = 'ms').index[-1]
    #print(f'last_ephys_bin {last_ephys_bin}')
    #print(f'last_ephys_time_bin {last_ephys_time_bin}')

    # time binned variables for later
    ep = nap.IntervalSet(start=0, end=last_ephys_time_bin, time_units = 's')
    speed_in_time = beh['S'].bin_average(bin_size=bs_t, time_units = 'ms', ep=ep)
    dt_in_time = beh['travel'].bin_average(bin_size=bs_t, time_units = 'ms', ep=ep)-((tns[0]-1)*tl)
    pos_in_time = dt_in_time%tl
    trial_number_in_time = (dt_in_time//tl)+tns[0]

    tcs = {}
    tcs_time = {}
    autocorrs = {}
    for cell in clusters.index:
        tc = nap.compute_1d_tuning_curves(nap.TsGroup([clusters[cell]]), 
                                        dt, 
                                        nb_bins=n_bins, 
                                        minmax=[min_bound, max_bound],
                                        ep=beh["moving"])[0]
        tc = np.array(tc)
        tc = np.nan_to_num(tc).astype(np.float64)
        if apply_guassian_filter:
            tc = gaussian_filter(tc, sigma=2.5)
        if apply_zscore:
            tc = zscore(tc)
        tc = tc[:last_ephys_bin] # only want bins with ephys data in it
        tcs[cell] = tc
        
        tc_time = clusters[cell].count(bin_size=bs_t, time_units = 'ms', ep=ep)
        tc_time = np.array(tc_time)
        tc_time = np.nan_to_num(tc_time).astype(np.float64)
        if apply_guassian_filter:
            tc_time = gaussian_filter(tc_time, sigma=2.5) # 
        if apply_zscore:
             tc_time = zscore(tc_time)
        tcs_time[cell] = tc_time

        lags = np.arange(0, 200, 1) # were looking at 10 timesteps back and 10 forward
        autocorr = []
        for lag in lags:
            if lag < 0:
                tc_offset = np.roll(tc, lag)
                tc_offset[lag:] = 0
            elif lag > 0:
                tc_offset = np.roll(tc, lag)
                tc_offset[:lag] = 0
            else:
                tc_offset = tc
            corr = stats.pearsonr(tc, tc_offset)[0]
            autocorr.append(corr)
        autocorr = np.array(autocorr)
        autocorrs[cell] = autocorr

    # drop beh trials from after last ephys bin
    beh_trials = beh['trials']
    beh_trials = beh_trials[:int(last_ephys_bin/(tl/bs))]

    return tcs, tcs_time, autocorrs, last_ephys_bin, beh, clusters

def compute_of_tcs(mouse, day, apply_zscore=True, apply_guassian_filter=True, fill_nans_from_neighbors=True,
                   source_path=None, bs_t=None, optimal_shift=0, session='OF1'):
    if source_path is None:
        source_path = DEFAULT_SOURCE_PATH 
    if bs_t is None:
        bs_t = time_bs

    of1_folder = f'{source_path}M{mouse}/D{day}/{session}/'
    shifted_grid_path = of1_folder + "tuning_scores/shifted_grid_score.parquet"
    spatial_path = of1_folder + "tuning_scores/shifted_spatial_information.parquet"
    spikes_path = of1_folder + f"sub-M{mouse}_ses-D{day}_typ-{session}_srt-kilosort4_clusters.npz"
    beh_path = of1_folder + f"sub-M{mouse}_ses-D{day}_typ-{session}_beh.nwb"
    shifted_grid_scores_of1 = pd.read_parquet(shifted_grid_path)
    spatial_information_score_of1 = pd.read_parquet(spatial_path)
    beh_OF = nap.load_file(beh_path)
    clusters_OF = nap.load_file(spikes_path)
    clusters_OF = curate_clusters(clusters_OF)

    # trick to clip the tc to around the end of the ephys recording
    last_ephys_time_bin = clusters_OF[clusters_OF.index[0]].count(bin_size=bs_t, time_units = 'ms').index[-1]
    ep = nap.IntervalSet(start=0, end=last_ephys_time_bin, time_units = 's')
 
    tcs = {}
    tcs_time = {}
    for cell in clusters_OF.index:

        # plot for cluster on ax
        position = np.stack([beh_OF['P_x'], beh_OF['P_y']], axis=1)
        beh_lag = compute_travel_projected(["P_x", "P_y"], position, position, optimal_shift)
        position_lagged = np.stack([beh_lag['P_x'], beh_lag['P_y']], axis=1)

        tc = nap.compute_2d_tuning_curves(nap.TsGroup([clusters_OF[cell]]), 
                                          position_lagged, nb_bins=(40,40), ep=ep)[0][0]

        if fill_nans_from_neighbors:
            tc = _fill_nans_from_neighbors(tc)
        if apply_guassian_filter:
            tc = gaussian_filter_nan(tc, sigma=(2.5,2.5))
        if apply_zscore: 
            tc = zscore(tc)
        tcs[cell] = tc
        tc_time = clusters_OF[cell].count(bin_size=bs_t, time_units = 'ms', ep=ep)
        tc_time = np.array(tc_time)
        tc_time = np.nan_to_num(tc_time).astype(np.float64)
        if apply_guassian_filter:
            tc_time = gaussian_filter(tc_time, sigma=2.5) # 
        if apply_zscore:
             tc_time = zscore(tc_time)
        tcs_time[cell] = tc_time

    return tcs, tcs_time, beh_OF, clusters_OF, ep


def get_time_binned_variables(mouse, day, apply_zscore=True, apply_guassian_filter=True, source_path=None, bs_t=None):
    if source_path is None:
        source_path = DEFAULT_SOURCE_PATH
    if bs_t is None:
        bs_t = time_bs

    vr_folder = f'{source_path}M{mouse}/D{day}/VR/'
    spikes_path = vr_folder + f"sub-M{mouse}_ses-D{day}_typ-VR_srt-kilosort4_clusters.npz"
    beh_path = vr_folder + f"sub-M{mouse}_ses-D{day}_typ-VR_beh.nwb"
    beh = nap.load_file(beh_path)
    clusters = nap.load_file(spikes_path)
    #print(f'there are this many clusters before curation {len(clusters)}')
    clusters = curate_clusters(clusters)

    tns = beh['trial_number']
    dt = beh['travel']-((tns[0]-1)*tl)
    n_bins = int(int(((np.ceil(np.nanmax(dt))//tl)+1)*tl)/bs)
    max_bound = int(((np.ceil(np.nanmax(dt))//tl)+1)*tl)
    min_bound = 0
    dt_bins = np.arange(0,max_bound,bs)

    # trick to clip the tc to around the end of the ephys recording
    # take the cell with the highest firing rate, and find the last bin with a spike
    # then work backwards and clip at the end of the last appropriate trials
    tc = nap.compute_1d_tuning_curves(nap.TsGroup([clusters[clusters.index[np.nanargmax(clusters.firing_rate)]]]), 
                                        dt, 
                                        nb_bins=n_bins, 
                                        minmax=[min_bound, max_bound],
                                        ep=beh["moving"])[0]
    
    tc = gaussian_filter(np.nan_to_num(tc).astype(np.float64), sigma=2.5)
    last_ephys_bin = int(np.nonzero(tc)[0][-1] + (tl/bs) - np.nonzero(tc)[0][-1]%(tl/bs))
    last_ephys_time_bin = clusters[clusters.index[0]].count(bin_size=bs_t, time_units = 'ms').index[-1]
    #print(f'last_ephys_bin {last_ephys_bin}')
    #print(f'last_ephys_time_bin {last_ephys_time_bin}')

    # time binned variables for later
    ep = nap.IntervalSet(start=0, end=last_ephys_time_bin, time_units = 's')
    speed_in_time = beh['S'].bin_average(bin_size=bs_t, time_units = 'ms', ep=ep)
    dt_in_time = beh['travel'].bin_average(bin_size=bs_t, time_units = 'ms', ep=ep)-((tns[0]-1)*tl)
    pos_in_time = dt_in_time%tl
    trial_number_in_time = (dt_in_time//tl)+tns[0]

    trial_type_in_time = []
    for tn in trial_number_in_time:
        trial = beh['trials'][beh['trials']['number'] == tn]
        trial_type_in_time.append(trial['type'].values[0])
    trial_type_in_time = np.array(trial_type_in_time)

    return speed_in_time, dt_in_time, pos_in_time, trial_number_in_time, trial_type_in_time


def get_sorted_trials_and_colors(beh, last_ephys_bin, tl, bs, ignore_performance=False):
    sorted_cats = beh['trials'][:int(last_ephys_bin/(tl/bs))].groupby(by=['context','type','performance'])
    sorted_cats = sort_dict_by_priority(sorted_cats, trial_cat_priority)

    sorted_trial_indices = []
    sorted_trial_colors = []
    sorted_block_sizes = []
    for group, cat_indices in zip(sorted_cats.keys(), sorted_cats.values()):
        if ignore_performance:
            group = (group[0], group[1], 'hit')
        c = get_color_for_group(group)
        sorted_trial_colors.extend(np.repeat(c, len(cat_indices)).tolist())
        sorted_trial_indices.extend(cat_indices.tolist())
        sorted_block_sizes.append(len(cat_indices))
    sorted_trial_colors = np.array(sorted_trial_colors)
    sorted_trial_indices = np.array(sorted_trial_indices)
    return sorted_trial_indices, sorted_trial_colors

def get_trial_groups_and_colors(beh, last_ephys_bin, tl, bs, ignore_performance=False):
    trial_colors = []
    trial_groups = []

    for trial in beh['trials'][:int(last_ephys_bin/(tl/bs))]:
        if ignore_performance:
            group = (trial['context'][0], trial['type'][0], 'hit')
        else:
            group=(trial['context'][0], 
                trial['type'][0],
                trial['performance'][0])
        c = get_color_for_group(group)
        group=''.join(group)
        trial_colors.append(c)
        trial_groups.append(group)
    trial_colors = np.array(trial_colors)
    trial_groups = np.array(trial_groups)
    return trial_groups, trial_colors


def plot_individual_rate_maps(mouse, day, cluster_ids, label='GC', figpath=''):
    tcs, _, _ , last_ephys_bin, beh, _ = compute_vr_tcs(mouse, day)
    trial_groups, trial_colors = get_trial_groups_and_colors(beh, last_ephys_bin, tl, bs)
    sorted_trial_indices, sorted_trial_colors = get_sorted_trials_and_colors(beh, last_ephys_bin, tl, bs)
 
    for id in cluster_ids:
        tc = tcs[id]
        tc = gaussian_filter(np.nan_to_num(tc).astype(np.float64), sigma=2.5)
        tc = zscore(tc)
        tc = tc[:last_ephys_bin] # only want bins with ephys data in it

        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=rm_figsize, width_ratios=[1,0.05], sharey=True)
        plot_firing_rate_map(ax[0], tc, bs=bs, tl=tl,p=95, sort_indices=None)
        ax[1].axis('off')
        ax[1].scatter(np.ones(len(trial_colors)), 
                    np.arange(0,len(trial_colors)), 
                    c = trial_colors,
                    marker='s')
        ax[0].set_xlabel('Pos (cm)')
        fig.savefig(f'{figpath}/M{mouse}D{day}{label}{id}.pdf', dpi=300, bbox_inches='tight')
        plt.show()

        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=rm_figsize, width_ratios=[1,0.05], sharey=True)
        plot_firing_rate_map(ax[0], tc, bs=bs, tl=tl,p=95, sort_indices=sorted_trial_indices)
        ax[1].axis('off')
        ax[1].scatter(np.ones(len(sorted_trial_colors)), 
                    np.arange(0,len(sorted_trial_colors)), 
                    c = sorted_trial_colors,
                    marker='s')
        ax[0].set_xlabel('Pos (cm)')
        fig.savefig(f'{figpath}/M{mouse}D{day}{label}{id}_sorted.pdf', dpi=300, bbox_inches='tight')
        plt.show()


def get_avg_profile(tc, bs=bs, tl=tl, mask=None):
    bpt = tl/bs
    n_trials = int(len(tc)/(bpt))
    trial_rate_map = []
    for i in range(n_trials):
        trial_rate_map.append(tc[int(i*bpt): int((i+1)*bpt)])
    trial_rate_map = np.array(trial_rate_map)

    if mask is None:
        return np.arange(bs/2, tl+(bs/2), bs), np.nanmean(trial_rate_map, axis=0)
    else:
        return np.arange(bs/2, tl+(bs/2), bs), np.nanmean(trial_rate_map[mask], axis=0)



def get_kmeans_spatial_labels(tc, labels, bs, tl, nperseg = 1600, noverlap = 1400):
    """
    Assign spatial labels using k-means labels allowed applied to the spectrogram.
    tc: 1D numpy array of shape (n_bins)
    labels: 1D numpy array of shape (n_windows) with k-means labels
    """
    bpt = tl/bs
    n_trials = int(len(tc)/(bpt))

    n_bins = len(tc)
    n_windows = len(labels)
    step = nperseg-noverlap  # nperseg - noverlap

    trial_centres = []
    for i in range(n_windows):
        start = i * step
        end = min(start + nperseg, n_bins)
        trial_centres.append(int(((start + end)/2)/bpt))
    trial_centres = np.array(trial_centres)

    kmean_trial_labels = np.full(n_trials, np.nan)
    for i in range(n_trials):
        trial_labels = labels[trial_centres == i+1]
        if len(trial_labels) != 0:

            modal_label = stats.mode(trial_labels, nan_policy='omit').mode
            kmean_trial_labels[i] = modal_label
 
    # Fill NaN values with the nearest non-NaN value
    for i in range(n_trials):
        if np.isnan(kmean_trial_labels[i]):
            # If the label is NaN, we need to find the nearest non-NaN value
            # walk left and right to find the nearest non-NaN value
            left = i - 1
            right = i + 1
            while left >= 0 and np.isnan(kmean_trial_labels[left]):
                left -= 1
            while right < n_trials and np.isnan(kmean_trial_labels[right]):
                right += 1
            if left >= 0 and right < n_trials:
                # Take the nearest non-NaN value
                if (i - left) <= (right - i):
                    kmean_trial_labels[i] = kmean_trial_labels[left]
                else:
                    kmean_trial_labels[i] = kmean_trial_labels[right]
            elif left >= 0:
                kmean_trial_labels[i] = kmean_trial_labels[left]
            elif right < n_trials:
                kmean_trial_labels[i] = kmean_trial_labels[right]

    return kmean_trial_labels

def plot_individual_rate_maps_with_avg_based_on_task_anchoring(mouse, day, cluster_ids, cluster_ids_for_spectrogram=None, label='GC', figpath='', source_path=None):
    if len(cluster_ids)==0:
        return

    if cluster_ids_for_spectrogram is None:
        cluster_ids_for_spectrogram = cluster_ids
    
    tcs, _, _ , last_ephys_bin, beh, _ = compute_vr_tcs(mouse, day, apply_zscore=False) 

    tcs_to_use = {cluster_id: tcs[cluster_id] for cluster_id in cluster_ids_for_spectrogram if cluster_id in tcs}
    results = spectral_analysis(tcs_to_use, tl, bs=bs)
    spectrograms = results[3] 
    S = spectrograms.mean(0)
    max_peaks = np.argmax(S, axis=0)
    labels = np.isin(max_peaks, [12, 28, 44, 60, 76]).astype(int) # these are the peaks that correspond to the task anchoring
    labels = np.isin(max_peaks, [28, 44, 60, 76]).astype(int) # these are the peaks that correspond to the task anchoring

    # translate the labels back to trial_numbers to use in the avging of the rate map

    if 'GC' in label:
        cmap = white_to_hex_cmap('#69201a')
    elif 'NGS' in label:
        cmap = white_to_hex_cmap('#0d2958')
    else:
        cmap = 'binary' 

    for id in cluster_ids:
        tc = tcs[id]
        tc = gaussian_filter(np.nan_to_num(tc).astype(np.float64), sigma=2.5)
        tc = tc[:last_ephys_bin] # only want bins with ephys data in it
        tcz = zscore(tc)
        
        trial_labels = get_kmeans_spatial_labels(tcz, labels, bs=bs, tl=tl) # reuse kmeans function to get the labels based on the task anchoring
    
        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(rm_figsize[0], rm_figsize[1]*1.45), sharex=True, height_ratios=[0.3, 1], width_ratios=[1,0.05], sharey='row')
        plot_firing_rate_map(ax[1,0], tc, bs=bs, tl=tl, p=95, sort_indices=None, cmap=cmap)
        ax[1,1].axis('off')
        ax[0,1].axis('off')
        ax[1,1].scatter(np.ones(len(trial_labels)), 
                    np.arange(0,len(trial_labels)),
                    c=trial_labels,
                    cmap='cool',
                    marker='s',
                    vmin=0, 
                    vmax=1)
        ax[1,0].set_xlabel('Pos (cm)')
        cmap_ta = plt.get_cmap('cool')

        for group in np.unique(trial_labels):
            if len(trial_labels[trial_labels == group])>5:
                x, y = get_avg_profile(tc, bs, tl, mask=trial_labels==group)
                ax[0,0].plot(x,y, color=cmap_ta(group), linewidth=1)
        fig.savefig(f'{figpath}/M{mouse}D{day}{label}{id}_with_avg_task_anchoring.pdf', dpi=300, bbox_inches='tight')
        plt.close()


def plot_individual_rate_maps_with_avg_k_means_grouped_spectrogram(mouse, day, cluster_ids, label='GC', figpath='', source_path=None):
    if len(cluster_ids)==0:
        return
    tcs, _, _ , last_ephys_bin, beh, _ = compute_vr_tcs(mouse, day, apply_zscore=False) 

    tcs_to_use = {cluster_id: tcs[cluster_id] for cluster_id in cluster_ids if cluster_id in tcs}
    results = spectral_analysis(tcs_to_use, tl, bs=bs)
    spectrograms = results[3] 
    fvalid = results[5]

    S = spectrograms.mean(0)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(S.T)
    
    # translate the labels back to trial_numbers to use in the avging of the rate map
 
    for id in cluster_ids:
        tc = tcs[id]
        tc = gaussian_filter(np.nan_to_num(tc).astype(np.float64), sigma=2.5)
        tc = tc[:last_ephys_bin] # only want bins with ephys data in it
        tcz = zscore(tc)
        
        trial_labels = get_kmeans_spatial_labels(tcz, labels, bs=bs, tl=tl)
    
        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(rm_figsize[0], rm_figsize[1]*1.45), sharex=True, height_ratios=[0.3, 1], width_ratios=[1,0.05], sharey='row')
        plot_firing_rate_map(ax[1,0], tc, bs=bs, tl=tl, p=95, sort_indices=None)
        ax[1,1].axis('off')
        ax[0,1].axis('off')
        ax[1,1].scatter(np.ones(len(trial_labels)), 
                    np.arange(0,len(trial_labels)),
                    c=trial_labels,
                    cmap='cool',
                    marker='s')
        ax[1,0].set_xlabel('Pos (cm)')
        cmap = plt.get_cmap('cool')

        for group in np.unique(trial_labels):
            if len(trial_labels[trial_labels == group])>5:
                x, y = get_avg_profile(tc, bs, tl, mask=trial_labels==group)
                ax[0,0].plot(x,y, color=cmap(group), linewidth=1)
        fig.savefig(f'{figpath}/M{mouse}D{day}{label}{id}_with_avg_kmeans_spectrogram.pdf', dpi=300, bbox_inches='tight')
        plt.close()


def plot_open_field_rate_map_optimal_ax(ax, cluster_id, mouse, day, df, source_path=None, optimal_method='median'):
    if source_path is None:
        source_path = DEFAULT_SOURCE_PATH
    session = 'OF1'
    of1_folder = f'{source_path}M{mouse}/D{day}/{session}/'
    shifted_grid_path = of1_folder + "tuning_scores/shifted_grid_score.parquet"
    spatial_path = of1_folder + "tuning_scores/shifted_spatial_information.parquet"
    spikes_path = of1_folder + f"sub-M{mouse}_ses-D{day}_typ-{session}_srt-kilosort4_clusters.npz"
    beh_path = of1_folder + f"sub-M{mouse}_ses-D{day}_typ-{session}_beh.nwb"
    shifted_grid_scores_of1 = pd.read_parquet(shifted_grid_path)
    spatial_information_score_of1 = pd.read_parquet(spatial_path)
    beh_OF = nap.load_file(beh_path)
    clusters_OF = nap.load_file(spikes_path)
    # plot for cluster on ax

    if optimal_method == 'median':
        optimal_lag = df[df.cluster_id==cluster_id].travel.values[0]
    elif optimal_method == 'best_grid_score':
        optimal_lag = df[df.cluster_id==cluster_id].optimal_travel_lag_grid_score.values[0]
    elif optimal_method == 'best_spatial_information':
        optimal_lag = df[df.cluster_id==cluster_id].optimal_travel_lag.values[0]

    grid_score_zero_lag = shifted_grid_scores_of1[(shifted_grid_scores_of1.cluster_id==cluster_id) & (shifted_grid_scores_of1.travel==0)].grid_score.values[0]
    grid_score_lagged = shifted_grid_scores_of1[(shifted_grid_scores_of1.cluster_id==cluster_id) & (shifted_grid_scores_of1.travel==optimal_lag)].grid_score.values[0]
    position = np.stack([beh_OF['P_x'], beh_OF['P_y']], axis=1)
    beh_lag = compute_travel_projected(["P_x", "P_y"], position, position, optimal_lag)
    position_lagged = np.stack([beh_lag['P_x'], beh_lag['P_y']], axis=1)
    tc = nap.compute_2d_tuning_curves(nap.TsGroup([clusters_OF[cluster_id]]), position_lagged, nb_bins=(40,40))[0]
    tc = gaussian_filter_nan(tc[0], sigma=(2.5,2.5))
    #ax.text(0,-8, f'sGS: {np.round(grid_score_zero_lag, decimals=1)}', size=9)
    #ax.text(0,-2, f'oGS: {np.round(grid_score_lagged, decimals=1)}', size=9)

    # clip at 99 percentile for better visualisation
    tc = np.clip(tc, a_min=0, a_max=np.nanpercentile(tc, 99))

    ax.imshow(tc, cmap='viridis')
    ax.axis('off') 

def _fill_nans_from_neighbors(rm):
    """
    For a 2D rate map rm:
      - If a NaN has any non-NaN neighbor in its 3x3 neighborhood,
        fill it with the nearest (Euclidean) non-NaN neighbor in that neighborhood.
      - If *all* neighbors in 3x3 are NaN, leave it as NaN.

    The neighbor search is done on a fixed reference copy of the map,
    and all fills are written into a separate output array, so that
    newly filled values are not used as neighbors for other NaNs
    in the same pass.
    """
    # Reference copy (never modified) and output copy (where we write fills)
    ref = np.array(rm, copy=True)
    out = np.array(rm, copy=True)

    nan_mask = np.isnan(ref)
    if not nan_mask.any():
        return out

    n_rows, n_cols = ref.shape

    # Loop over all NaN pixels using the original mask on the reference map
    for i, j in np.argwhere(nan_mask):
        # 3x3 neighborhood bounds
        r0 = max(0, i - 1)
        r1 = min(n_rows - 1, i + 1)
        c0 = max(0, j - 1)
        c1 = min(n_cols - 1, j + 1)

        neighborhood = ref[r0:r1+1, c0:c1+1]
        # Coordinates of non-NaN neighbors in the neighborhood (local indices)
        local_coords = np.argwhere(~np.isnan(neighborhood))

        # If there are no valid neighbors, leave this pixel as NaN
        if local_coords.size == 0:
            continue

        # Convert local neighbor coords to global image coords
        global_coords = local_coords + np.array([r0, c0])

        # Distances from (i, j) to each valid neighbor
        dists = np.sqrt(
            (global_coords[:, 0] - i) ** 2 +
            (global_coords[:, 1] - j) ** 2
        )
        nearest_idx = np.argmin(dists)
        vi, vj = global_coords[nearest_idx]

        # Fill NaN in the *output* array with the value of the nearest valid neighbor
        out[i, j] = ref[vi, vj]

    return out


def plot_vr_rate_map_trials_ax(ax, cluster_id, mouse, day, df, source_path=None):
    tcs, _, _ , last_ephys_bin, beh, _ = compute_vr_tcs(mouse, day, apply_zscore=False) 
    tc = tcs[cluster_id]
    tc = gaussian_filter(np.nan_to_num(tc).astype(np.float64), sigma=2.5)
    tc = tc[:last_ephys_bin] # only want bins with ephys data in it
    tcz = zscore(tc)
    # plot firing rate map
    plot_firing_rate_map(ax, tc, bs=bs, tl=tl,p=95, sort_indices=None)
    ax.set_xlabel('Pos (cm)')

def plot_vr_rate_map_avg_ax(ax, cluster_id, mouse, day, df, source_path=None):
    tcs, _, _ , last_ephys_bin, beh, _ = compute_vr_tcs(mouse, day, apply_zscore=False) 
    trial_groups, trial_colors = get_trial_groups_and_colors(beh, last_ephys_bin, tl, bs, ignore_performance=True)
    tc = tcs[cluster_id]
    tc = gaussian_filter(np.nan_to_num(tc).astype(np.float64), sigma=2.5)
    tc = tc[:last_ephys_bin] # only want bins with ephys data in it
    tcz = zscore(tc)
    # plot average rate map
    for group in np.unique(trial_groups):
        if len(trial_groups[trial_groups == group])>5:
            x, y = get_avg_profile(tc, bs, tl, mask=trial_groups==group)
            ax.plot(x,y, color=trial_colors[trial_groups==group][0], linewidth=1)
    ax.set_xlim(0,200)


def plot_individual_of_rate_maps_optimal_versus_standard(mouse, day, df, label='GC', figpath='', source_path=None):
    if source_path is None:
        source_path = DEFAULT_SOURCE_PATH
    session = 'OF1'
    of1_folder = f'{source_path}M{mouse}/D{day}/{session}/'
    shifted_grid_path = of1_folder + "tuning_scores/shifted_grid_score.parquet"
    spatial_path = of1_folder + "tuning_scores/shifted_spatial_information.parquet"
    spikes_path = of1_folder + f"sub-M{mouse}_ses-D{day}_typ-{session}_srt-kilosort4_clusters.npz"
    beh_path = of1_folder + f"sub-M{mouse}_ses-D{day}_typ-{session}_beh.nwb"
    shifted_grid_scores_of1 = pd.read_parquet(shifted_grid_path)
    spatial_information_score_of1 = pd.read_parquet(spatial_path)
    beh_OF = nap.load_file(beh_path)
    clusters_OF = nap.load_file(spikes_path)

    for cluster_id in df.cluster_id:
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(5,2), squeeze=False)
        
        # plot the standard rate map
        grid_score_standard = shifted_grid_scores_of1[(shifted_grid_scores_of1.cluster_id==cluster_id) & (shifted_grid_scores_of1.travel==0)].grid_score.values[0]
        position = np.stack([beh_OF['P_x'], beh_OF['P_y']], axis=1)
        beh_lag = compute_travel_projected(["P_x", "P_y"], position, position, 0)
        position_lagged = np.stack([beh_lag['P_x'], beh_lag['P_y']], axis=1)
        tc = nap.compute_2d_tuning_curves(nap.TsGroup([clusters_OF[cluster_id]]), position_lagged, nb_bins=(40,40))[0]
        tc = gaussian_filter_nan(tc[0], sigma=(2.5,2.5))
        ax[0,0].text(0,-2, f'GS: {np.round(grid_score_standard, decimals=1)}', size=10)
        ax[0,0].imshow(tc, cmap='jet')
        
        # plot the lagged rate map
        optimal_lag = df[df.cluster_id==cluster_id].travel.values[0]
        grid_score_lagged = shifted_grid_scores_of1[(shifted_grid_scores_of1.cluster_id==cluster_id) & (shifted_grid_scores_of1.travel==optimal_lag)].grid_score.values[0]
        position = np.stack([beh_OF['P_x'], beh_OF['P_y']], axis=1)
        beh_lag = compute_travel_projected(["P_x", "P_y"], position, position, optimal_lag)
        position_lagged = np.stack([beh_lag['P_x'], beh_lag['P_y']], axis=1)
        tc = nap.compute_2d_tuning_curves(nap.TsGroup([clusters_OF[cluster_id]]), position_lagged, nb_bins=(40,40))[0]
        tc = gaussian_filter_nan(tc[0], sigma=(2.5,2.5))
        ax[0,1].text(0,-2, f'GS: {np.round(grid_score_lagged, decimals=1)}', size=10)
        ax[0,1].imshow(tc, cmap='jet') 

        for axi in ax.flatten():
            axi.axis('off')
        plt.tight_layout()
        plt.savefig(f'{figpath}/M{mouse}D{day}{label}{cluster_id}_OF_rate_maps.pdf', dpi=1000)
        plt.close()   
    return 

def white_to_hex_cmap(hex_color, name='custom_cmap'):
    return LinearSegmentedColormap.from_list(name, ['#FFFFFF', hex_color])

def plot_individual_rate_maps_with_avg_by_trial_type_with_open_field(mouse, day, df, label='GC', figpath='',vr_type='VR',source_path=None):
    if df.empty:
        return
    if source_path is None:
        source_path = DEFAULT_SOURCE_PATH
    if vr_type is 'MCVR':
        tl = mcvr_tl
    else:
        tl = vr_tl
    if label=='GC':
        cmap = white_to_hex_cmap('#69201a')
    elif label=='NGS':
        cmap = white_to_hex_cmap('#0d2958')
    else:
        cmap = 'binary' 

    tcs, _, _ , last_ephys_bin, beh, _ = compute_vr_tcs(mouse, day, apply_zscore=False, vr_type=vr_type, source_path=source_path) 
    trial_groups, trial_colors = get_trial_groups_and_colors(beh, last_ephys_bin, tl, bs, ignore_performance=True)
    sorted_trial_indices, sorted_trial_colors = get_sorted_trials_and_colors(beh, last_ephys_bin, tl, bs, ignore_performance=True)

    for id in df.cluster_id.values:
        tc = tcs[id]
        tc = gaussian_filter(np.nan_to_num(tc).astype(np.float64), sigma=2.5)
        tc = tc[:last_ephys_bin] # only want bins with ephys data in it
        tcz = zscore(tc)

        fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(rm_figsize[0], rm_figsize[1]*1.9), 
                               sharex=False, height_ratios=[0.6, 0.25, 1], width_ratios=[1,0.05])
        # plot unsorted rate map
        plot_firing_rate_map(ax[2,0], tc, bs=bs, tl=tl,p=95, sort_indices=None, cmap=cmap)

        # plot open field rate map
        plot_open_field_rate_map_optimal_ax(ax[0,0], id, mouse, day, df, source_path=None)

        # plot trial colors
        ax[2,1].scatter(np.ones(len(trial_colors)), 
                    np.arange(0,len(trial_colors)), 
                    c = trial_colors,
                    marker='s')
        
        # plot average rate maps
        for group in np.unique(trial_groups):
            if len(trial_groups[trial_groups == group])>5:
                x, y = get_avg_profile(tc, bs, tl, mask=trial_groups==group)
                ax[1,0].plot(x,y, color=trial_colors[trial_groups==group][0], linewidth=1)
        
        ax[2,0].set_xlabel('Pos (cm)')
        ax[2,1].axis('off')
        ax[1,0].set_xticks([])
        ax[1,0].set_xlim([0,200])
        ax[0,1].axis('off')
        ax[1,1].axis('off')
        ax[1,0].yaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))
        fig.savefig(f'{figpath}/M{mouse}D{day}{label}{id}_with_avg_and_of_rate_map.pdf', dpi=300, bbox_inches='tight')
        plt.close()


def plot_individual_rate_maps_with_avg_by_trial_type(mouse, day, cluster_ids, label='GC', figpath='',vr_type='VR',source_path=None):
    if len(cluster_ids)==0:
        return
    if source_path is None:
        source_path = DEFAULT_SOURCE_PATH
    if vr_type == 'MCVR':
        tl = mcvr_tl
    else:
        tl = vr_tl
    if label=='GC':
        cmap = white_to_hex_cmap('#69201a')
    elif label=='NGS':
        cmap = white_to_hex_cmap('#0d2958')
    else:
        cmap = 'binary'

    tcs, _, _ , last_ephys_bin, beh, _ = compute_vr_tcs(mouse, day, apply_zscore=False, vr_type=vr_type, source_path=source_path) 
    trial_groups, trial_colors = get_trial_groups_and_colors(beh, last_ephys_bin, tl, bs, ignore_performance=True)
    sorted_trial_indices, sorted_trial_colors = get_sorted_trials_and_colors(beh, last_ephys_bin, tl, bs, ignore_performance=True)
 
    for id in cluster_ids:
        tc = tcs[id]
        tc = gaussian_filter(np.nan_to_num(tc).astype(np.float64), sigma=2.5)
        tc = tc[:last_ephys_bin] # only want bins with ephys data in it
        tcz = zscore(tc)

        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(rm_figsize[0], rm_figsize[1]*1.45), sharex=True, height_ratios=[0.3, 1], width_ratios=[1,0.05], sharey='row')
        plot_firing_rate_map(ax[1,0], tc, bs=bs, tl=tl,p=95, sort_indices=None, cmap=cmap)
        ax[1,1].axis('off')
        ax[0,1].axis('off')
        ax[1,1].scatter(np.ones(len(trial_colors)), 
                    np.arange(0,len(trial_colors)), 
                    c = trial_colors,
                    marker='s')
        ax[1,0].set_xlabel('Pos (cm)')
        for group in np.unique(trial_groups):
            if len(trial_groups[trial_groups == group])>5:
                x, y = get_avg_profile(tc, bs, tl, mask=trial_groups==group)
                ax[0,0].plot(x,y, color=trial_colors[trial_groups==group][0], linewidth=1)
        fig.savefig(f'{figpath}/M{mouse}D{day}{label}{id}_with_avg.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(rm_figsize[0], rm_figsize[1]*1.45), sharex=True, height_ratios=[0.3, 1], width_ratios=[1,0.05], sharey='row')
        plot_firing_rate_map(ax[1,0], tc, bs=bs, tl=tl,p=95, sort_indices=sorted_trial_indices, cmap=cmap)
        ax[1,1].axis('off')
        ax[0,1].axis('off')
        ax[1,1].scatter(np.ones(len(sorted_trial_colors)), 
                    np.arange(0,len(sorted_trial_colors)), 
                    c = sorted_trial_colors,
                    marker='s')
        ax[1,0].set_xlabel('Pos (cm)')
        for group in np.unique(trial_groups):
            if len(trial_groups[trial_groups == group])>5:
                x, y = get_avg_profile(tc, bs, tl, mask=trial_groups==group)
                ax[0,0].plot(x,y, color=trial_colors[trial_groups==group][0], linewidth=1)
        fig.savefig(f'{figpath}/M{mouse}D{day}{label}{id}_sorted_with_avg.pdf', dpi=300, bbox_inches='tight')
        plt.close() 


def plot_individual_rate_maps_with_avg(mouse, day, cluster_ids, label='GC', figpath=''):
    if len(cluster_ids)==0:
        return

    tcs, _, _ , last_ephys_bin, beh, _ = compute_vr_tcs(mouse, day, apply_zscore=False) 
    trial_groups, trial_colors = get_trial_groups_and_colors(beh, last_ephys_bin, tl, bs)
    sorted_trial_indices, sorted_trial_colors = get_sorted_trials_and_colors(beh, last_ephys_bin, tl, bs)
 
    for id in cluster_ids:
        tc = tcs[id]
        tc = gaussian_filter(np.nan_to_num(tc).astype(np.float64), sigma=2.5)
        tc = tc[:last_ephys_bin] # only want bins with ephys data in it
        tcz = zscore(tc)


        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(rm_figsize[0], rm_figsize[1]*1.45), sharex=True, height_ratios=[0.3, 1], width_ratios=[1,0.05], sharey='row')
        plot_firing_rate_map(ax[1,0], tc, bs=bs, tl=tl,p=95, sort_indices=None)
        ax[1,1].axis('off')
        ax[0,1].axis('off')
        ax[1,1].scatter(np.ones(len(trial_colors)), 
                    np.arange(0,len(trial_colors)), 
                    c = trial_colors,
                    marker='s')
        ax[1,0].set_xlabel('Pos (cm)')
        for group in np.unique(trial_groups):
            if len(trial_groups[trial_groups == group])>5:
                x, y = get_avg_profile(tc, bs, tl, mask=trial_groups==group)
                ax[0,0].plot(x,y, color=trial_colors[trial_groups==group][0], linewidth=1)
        fig.savefig(f'{figpath}/M{mouse}D{day}{label}{id}_with_avg.pdf', dpi=300, bbox_inches='tight')
        plt.close()


        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(rm_figsize[0], rm_figsize[1]*1.45), sharex=True, height_ratios=[0.3, 1], width_ratios=[1,0.05], sharey='row')
        plot_firing_rate_map(ax[1,0], tc, bs=bs, tl=tl,p=95, sort_indices=sorted_trial_indices)
        ax[1,1].axis('off')
        ax[0,1].axis('off')
        ax[1,1].scatter(np.ones(len(sorted_trial_colors)), 
                    np.arange(0,len(sorted_trial_colors)), 
                    c = sorted_trial_colors,
                    marker='s')
        ax[1,0].set_xlabel('Pos (cm)')
        for group in np.unique(trial_groups):
            if len(trial_groups[trial_groups == group])>5:
                x, y = get_avg_profile(tc, bs, tl, mask=trial_groups==group)
                ax[0,0].plot(x,y, color=trial_colors[trial_groups==group][0], linewidth=1)
        fig.savefig(f'{figpath}/M{mouse}D{day}{label}{id}_sorted_with_avg.pdf', dpi=300, bbox_inches='tight')
        plt.close()

def plot_stops_mouse_day(mouse, day, figpath, return_fig=True): 
    _, _, _ , last_ephys_bin, beh,_ = compute_vr_tcs(mouse, day)

    plot_stops(beh, tl=tl, sort=False, return_fig=return_fig, last_ephys_bin=last_ephys_bin,
           savepath=f'{figpath}/M{mouse}D{day}_stops')
    plot_stops(beh, tl=200, sort=True, return_fig=return_fig, last_ephys_bin=last_ephys_bin,
           savepath=f'{figpath}/M{mouse}D{day}_stops_sorted')
    

def plot_vr_rate_maps(mouse, day, cluster_ids, label, figpath):
    print(f'plotting vr rate maps using cluster_ids {cluster_ids}')
    if len(cluster_ids)==0:
        return
    
    tcs,_,_,_,_,_ = compute_vr_tcs(mouse, day)
    ncols = 10
    nrows = int(np.ceil(len(cluster_ids)/ncols))
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 1.4*nrows), squeeze=False)
    counter = 0
    for j in range(nrows):
        for i in range(ncols):
            if counter<len(cluster_ids):
                index = cluster_ids[counter]
                plot_firing_rate_map(ax[j, i], 
                                    zscore(tcs[index]),
                                    bs=bs,
                                    tl=tl,
                                    p=95)
            else:
                ax[j, i].axis('off')
            counter+=1
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])
            ax[j, i].xaxis.set_tick_params(labelbottom=False)
            ax[j, i].yaxis.set_tick_params(labelleft=False)
    plt.tight_layout()
    plt.savefig(f'{figpath}/M{mouse}D{day}_VR_rate_maps_{label}.pdf')
    plt.show()
    #plt.close()    


def plot_spectrogram(mouse, day, cluster_ids, label, figpath=None):
    if len(cluster_ids)==0:
        return
    tcs, _, _ , _, _, _ = compute_vr_tcs(mouse, day)
    tcs_to_use = {cluster_id: tcs[cluster_id] for cluster_id in cluster_ids if cluster_id in tcs}
    results = spectral_analysis(tcs_to_use, tl, bs=bs)
    spectrograms = results[3] 

    plt.figure(figsize=(4,4))
    S = spectrograms.mean(0)
    plt.imshow(S,origin='lower',aspect='auto',vmax=0.25,cmap='magma')
    plt.yticks([0, len(S)/2, len(S)], [0, 1, 2])
    plt.ylabel(f'Frequency (m-1)')
    plt.savefig(f'{figpath}/M{mouse}D{day}_spectrogram_{label}.pdf', dpi=300, bbox_inches='tight')
    plt.close()    


def plot_toroidal_projection(mouse, day, cluster_ids, figpath):
    if len(cluster_ids)==0:
        return
    
    tcs, _, _ , last_ephys_bin, beh,_ = compute_vr_tcs(mouse, day)
    tl=200

    tcs_to_use = {cluster_id: tcs[cluster_id] for cluster_id in cluster_ids if cluster_id in tcs}
    N = len(tcs_to_use)
    zmaps = np.array(list(tcs_to_use.values()))
    results = spectral_analysis(tcs_to_use, tl, bs=bs)
    f_modules =              results[0]
    phi_modules =            results[1]
    grid_cell_idxs_modules = results[2]
    spectrograms =           results[3]
    trial_starts =           results[6]
    grid_cell_idxs = grid_cell_idxs_modules[0]
    phi = phi_modules[0]
    Ng = len(grid_cell_idxs)
    maps = gaussian_filter1d(zmaps[grid_cell_idxs].reshape(Ng, -1), 2, axis=1)
    angles = np.arctan2(np.cos(phi)@maps, np.sin(phi)@maps)

    sorted_trial_indices, sorted_trial_colors = get_sorted_trials_and_colors(beh, last_ephys_bin, tl, bs) 
    trial_groups, trial_colors = get_trial_groups_and_colors(beh, last_ephys_bin, tl, bs)

    angles1 = angles[0].reshape(-1,int(tl/bs))
    angles2 = angles[1].reshape(-1,int(tl/bs))
    angles3 = angles[2].reshape(-1,int(tl/bs))

    for i, angles0 in  enumerate([angles1, angles2, angles3]):
        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(rm_figsize[0], rm_figsize[1]*1.45), sharex=True, height_ratios=[0.3, 1], width_ratios=[1,0.05], sharey='row')

        x = np.arange(1, len(angles0)+1)
        y = np.arange(0, len(angles0[0])*bs, bs)
        X, Y = np.meshgrid(x, y)
        heatmap = ax[1,0].pcolormesh(Y, X, angles0.T, shading='auto', cmap='hsv')
        heatmap.set_rasterized(True)
        ax[1,0].set_xlabel('Pos. (cm)')
        ax[1,1].axis('off')
        ax[0,1].axis('off')
        ax[1,1].scatter(np.ones(len(trial_colors)), 
                    np.arange(0,len(trial_colors)), 
                    c = trial_colors,
                    marker='s',s=1)
        for group in np.unique(trial_groups):
            if len(trial_groups[trial_groups == group])>5:
                x, y = get_avg_profile(angles[i], bs, tl, mask=trial_groups==group)
                ax[0,0].plot(x,y, color=trial_colors[trial_groups==group][0], linewidth=1)
        ax[1,0].set_xlim(0,tl)
        ax[1,0].set_ylim(0,len(angles0))
        ax[1,0].invert_yaxis()
        fig.savefig(f'{figpath}/M{mouse}D{day}A{i}_torus.pdf', dpi=300, bbox_inches='tight')
        plt.close()    


    angles1_sorted = angles[0].reshape(-1,int(tl/bs))[sorted_trial_indices]
    angles2_sorted = angles[1].reshape(-1,int(tl/bs))[sorted_trial_indices]
    angles3_sorted = angles[2].reshape(-1,int(tl/bs))[sorted_trial_indices]

    for i, angles0_sorted in enumerate([angles1_sorted, angles2_sorted, angles3_sorted]):
        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(rm_figsize[0], rm_figsize[1]*1.45), sharex=True, height_ratios=[0.3, 1], width_ratios=[1,0.05], sharey='row')

        x = np.arange(1, len(angles0_sorted)+1)
        y = np.arange(0, len(angles0_sorted[0])*bs, bs)
        X, Y = np.meshgrid(x, y)

        for group in np.unique(trial_groups):
            if len(trial_groups[trial_groups == group])>5:
                x, y = get_avg_profile(angles[i], bs, tl, mask=trial_groups==group)
                ax[0,0].plot(x,y, color=trial_colors[trial_groups==group][0], linewidth=1)

        heatmap = ax[1,0].pcolormesh(Y, X, angles0_sorted.T, shading='auto', cmap='hsv')
        heatmap.set_rasterized(True)
        ax[1,0].set_xlabel('Pos. (cm)')
        ax[1,1].axis('off')
        ax[0,1].axis('off')
        ax[1,1].scatter(np.ones(len(sorted_trial_colors)), 
                    np.arange(0,len(sorted_trial_colors)), 
                    c = sorted_trial_colors,
                    marker='s',s=1)
        ax[1,0].set_xlim(0,tl)
        ax[1,0].set_ylim(0,len(angles0_sorted))
        ax[1,0].invert_yaxis()
        fig.savefig(f'{figpath}/M{mouse}D{day}A{i}_torus_sorted.pdf', dpi=300, bbox_inches='tight')
        plt.close()    


def plot_decoding(mouse, day, cluster_ids, label, figpath):
    if len(cluster_ids)==0:
        return
    
    tcs, tcs_time, autocorrs, last_ephys_bin, beh, clusters = compute_vr_tcs(mouse, day)
    trial_groups, trial_colors = get_trial_groups_and_colors(beh, last_ephys_bin, tl, bs)
    sorted_trial_indices, sorted_trial_colors = get_sorted_trials_and_colors(beh, last_ephys_bin, tl, bs)

    print(len(sorted_trial_indices))

    tns = beh['trial_number']
    dt = beh['travel']-((tns[0]-1)*tl)
    n_bins = int(int(((np.ceil(np.nanmax(dt))//tl)+1)*tl)/bs)
    max_bound = int(((np.ceil(np.nanmax(dt))//tl)+1)*tl)
    min_bound = 0
    dt_bins = np.arange(0,max_bound,bs)

    x_true_dt = dt_bins[:last_ephys_bin]
    true_position = x_true_dt%tl
    trial_numbers = (x_true_dt//tl)+beh['trials']['number'][0]
    tns_to_decode_with = np.array(beh['trials']['number'])
    tns_to_decode_with = tns_to_decode_with[tns_to_decode_with<=np.nanmax(trial_numbers)]
    trial_types = np.array(beh['trials'][:int(last_ephys_bin/(tl/bs))]['type'])

    tns_to_decode = np.array(beh['trials']['number']) # decode all trials to visualise
    tns_to_train = np.array(beh['trials']['number'][(np.isin(beh['trials']['type'], np.array(['b','nb']))) &
                                                    (np.isin(beh['trials']['performance'], np.array(['hit'])))]) 
    tns_to_decode = tns_to_decode[tns_to_decode<=np.nanmax(trial_numbers)] # handles last ephys trials
    tns_to_train = tns_to_train[tns_to_train<=np.nanmax(trial_numbers)] # handles last ephys trials

    tcs = {cluster_id: tcs[cluster_id] for cluster_id in cluster_ids if cluster_id in tcs}

    predictions, errors = cross_validate_decoder(tcs, true_position, trial_numbers, tns_to_decode, tns_to_train, tl, bs, train=0.9, n=10, verbose=False)
    avg_predictions = circular_nanmean(predictions, tl, axis=2)

    sorted_predictions = predictions[sorted_trial_indices]
    sorted_errors = errors[sorted_trial_indices]

    avg_sorted_predictions = circular_nanmean(sorted_predictions, tl, axis=2)
    avg_sorted_errors = circular_nanmean(sorted_errors, tl, axis=2)
    
    # non sorted decoder
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(rm_figsize), width_ratios=[1,0.05], sharey=True)
    x = np.arange(1, len(avg_predictions)+1)
    y = np.arange(0, len(avg_predictions[0])*bs, bs)
    X, Y = np.meshgrid(x, y)
    heatmap = ax[0].pcolormesh(Y, X, avg_predictions.T, shading='auto', cmap='hsv')
    heatmap.set_rasterized(True)
    ax[0].set_xlabel('Pos. (cm)')
    ax[1].axis('off')
    ax[1].scatter(np.ones(len(trial_colors)), 
                    np.arange(0,len(trial_colors)), 
                    c = trial_colors,
                    marker='s',s=1)
    ax[0].set_xlim(0,tl)
    ax[0].set_ylim(0,len(avg_predictions))
    ax[0].invert_yaxis()
    fig.savefig(f'{figpath}/M{mouse}D{day}_Decoder_{label}.pdf', dpi=300, bbox_inches='tight')
    plt.close()    

    # sorted decoder
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(rm_figsize), width_ratios=[1,0.05], sharey=True)
    x = np.arange(1, len(avg_predictions)+1)
    y = np.arange(0, len(avg_predictions[0])*bs, bs)
    X, Y = np.meshgrid(x, y)
    heatmap = ax[0].pcolormesh(Y, X, avg_predictions[sorted_trial_indices].T, shading='auto', cmap='hsv')
    heatmap.set_rasterized(True)
    ax[0].set_xlabel('Pos. (cm)')
    ax[1].axis('off')
    ax[1].scatter(np.ones(len(sorted_trial_colors)), 
                    np.arange(0,len(sorted_trial_colors)), 
                    c = sorted_trial_colors,
                    marker='s',s=1)
    ax[0].set_xlim(0,tl)
    ax[0].set_ylim(0,len(avg_predictions))
    ax[0].invert_yaxis()
    fig.savefig(f'{figpath}/M{mouse}D{day}_Decoder_sorted_{label}.pdf', dpi=300, bbox_inches='tight')
    plt.close()    


    # sorted decoder with avg
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(rm_figsize[0], 1.45*rm_figsize[1]), 
                           height_ratios=[0.3, 1], width_ratios=[1,0.05], sharex=True, sharey='row')
    x = np.arange(1, len(avg_predictions)+1)
    y = np.arange(0, len(avg_predictions[0])*bs, bs)
    X, Y = np.meshgrid(x, y)
    heatmap = ax[1,0].pcolormesh(Y, X, avg_predictions[sorted_trial_indices].T, shading='auto', cmap='hsv')
    heatmap.set_rasterized(True)

    ax[0,0].plot(y,y, color='black', linestyle='dashed')
    ax[0,0].set_ylim(0,tl)
    ax[0,0].plot(y, circular_nanmean(avg_sorted_predictions[:len(trial_types[trial_types=='b'])], tl=tl, axis=0), color='tab:blue', linewidth=1)
    ax[0,0].plot(y, circular_nanmean(avg_sorted_predictions[len(trial_types[trial_types=='b']):], tl=tl, axis=0), color='tab:orange', linewidth=1)
    ax[1,0].set_xlabel('Pos. (cm)')
    ax[1,1].axis('off')
    ax[0,1].axis('off')
    ax[1,1].scatter(np.ones(len(sorted_trial_colors)), 
                    np.arange(0,len(sorted_trial_colors)), 
                    c = sorted_trial_colors,
                    marker='s',s=1)
    ax[1,0].set_xlim(0,tl)
    ax[1,0].set_ylim(0,len(avg_predictions))
    ax[1,0].invert_yaxis()
    fig.savefig(f'{figpath}/M{mouse}D{day}_Decoder_sorted_with_avg_{label}.pdf', dpi=300, bbox_inches='tight')
    plt.close()    

    
    norm = TwoSlopeNorm(vmin=-35, vcenter=0, vmax=35)
    # collated plot for assaying over training sets
    fig, ax = plt.subplots(
        2, 6, layout='constrained', figsize=(8.5*rm_figsize[0], 1.45*rm_figsize[1]), sharey=False, sharex=True, height_ratios=[0.3,1]
    ) 
    for i, train_set in enumerate(zip([['b', 'nb'], ['b'], ['nb']])):
        tns_to_decode = np.array(beh['trials']['number']) # decode all trials to visualise
        tns_to_train = np.array(beh['trials']['number'][np.isin(beh['trials']['type'], np.array(train_set)) &
                                                        np.isin(beh['trials']['performance'], np.array('hit'))]) 
        tns_to_decode = tns_to_decode[tns_to_decode<=np.nanmax(trial_numbers)] # handles last ephys trials
        tns_to_train = tns_to_train[tns_to_train<=np.nanmax(trial_numbers)] # handles last ephys trials

        predictions, errors = cross_validate_decoder(tcs, true_position, trial_numbers, tns_to_decode, tns_to_train,tl,bs,train=0.9, n=50)
        sorted_predictions = predictions[np.argsort(trial_types)]
        sorted_errors = errors[np.argsort(trial_types)]

        avg_sorted_predictions = circular_nanmean(sorted_predictions, tl, axis=2)
        avg_sorted_errors = np.nanmean(sorted_errors, axis=2)
        sem_sorted_errors = stats.sem(sorted_errors, axis=2)
        
        x = np.arange(1, len(avg_sorted_predictions)+1)
        y = np.arange(0, len(avg_sorted_predictions[0])*bs, bs)
        X, Y = np.meshgrid(x, y)

        #ax[0,i*2].set_title(f'train:{train_set}')
        ax[0,i*2].plot(np.arange(bs/2,(tl+bs/2),bs),np.arange(bs/2,(tl+bs/2),bs), color='black', linestyle='dashed')
        ax[0,i*2].plot(np.arange(bs/2,(tl+bs/2),bs), circular_nanmean(avg_sorted_predictions[:len(trial_types[trial_types=='b'])], tl=tl, axis=0), color='tab:blue')
        ax[0,i*2].plot(np.arange(bs/2,(tl+bs/2),bs), circular_nanmean(avg_sorted_predictions[len(trial_types[trial_types=='b']):], tl=tl, axis=0), color='tab:orange')
        heatmap1 = ax[1,i*2].pcolormesh(Y, X, avg_sorted_predictions.T, shading='auto', cmap='hsv')
        heatmap1.set_rasterized(True)
        ax[1,i*2].axhline(y=len(trial_types[trial_types=='b']), color='black')
        b_error = np.nanmean(avg_sorted_errors[:len(trial_types[trial_types=='b'])], axis=0)
        nb_error = np.nanmean(avg_sorted_errors[len(trial_types[trial_types=='b']):], axis=0)
        ax[0,(i*2)+1].plot(np.arange(bs/2,(tl+bs/2),bs), b_error, color='tab:blue')
        ax[0,(i*2)+1].plot(np.arange(bs/2,(tl+bs/2),bs), nb_error, color='tab:orange')

        heatmap2 = ax[1,(i*2)+1].pcolormesh(Y, X, avg_sorted_errors.T, shading='auto', cmap='Purples')
        heatmap2.set_rasterized(True)
        ax[1,(i*2)+1].axhline(y=len(trial_types[trial_types=='b']), color='black')
    for i in range(6):
        ax[1,i].invert_yaxis()
        ax[1,i].set_xlabel(f'Pos (cm)')
        ax[0,i].set_xlim(0,tl)
        ax[1,i].set_xlim(0,tl)
        if i != 0:
            ax[1,i].set_yticklabels([])
    fig.savefig(f'{figpath}/M{mouse}D{day}_different_train_sets_{label}.pdf', dpi=300, bbox_inches='tight')
    plt.close()    



def plot_projected_stops(mouse, day, cluster_ids, label, figpath):
    if len(cluster_ids)==0:
        return
    
    tcs, tcs_time, autocorrs, last_ephys_bin, beh, clusters = compute_vr_tcs(mouse, day)
    trial_groups, trial_colors = get_trial_groups_and_colors(beh, last_ephys_bin, tl, bs)
    sorted_trial_indices, sorted_trial_colors = get_sorted_trials_and_colors(beh, last_ephys_bin, tl, bs)

    tns = beh['trial_number']
    dt = beh['travel']-((tns[0]-1)*tl)
    n_bins = int(int(((np.ceil(np.nanmax(dt))//tl)+1)*tl)/bs)
    max_bound = int(((np.ceil(np.nanmax(dt))//tl)+1)*tl)
    min_bound = 0
    dt_bins = np.arange(0,max_bound,bs)
    x_true_dt = dt_bins[:last_ephys_bin]
    trial_numbers = (x_true_dt//tl)+beh['trials']['number'][0]

    last_ephys_time_bin = clusters[clusters.index[0]].count(bin_size=time_bs, time_units = 'ms').index[-1]

    # time binned variables for later
    ep = nap.IntervalSet(start=0, end=last_ephys_time_bin, time_units = 's')
    speed_in_time = beh['S'].bin_average(bin_size=time_bs, time_units = 'ms', ep=ep)
    dt_in_time = beh['travel'].bin_average(bin_size=time_bs, time_units = 'ms', ep=ep)-((tns[0]-1)*tl)
    time_vals = np.arange(0, len(speed_in_time)*(time_bs/1000),time_bs/1000) # secs
    pos_in_time = dt_in_time%tl
    trial_number_in_time = (dt_in_time//tl)+tns[0]

    # decoding in time
    tcs_time = {cluster_id: tcs_time[cluster_id] for cluster_id in cluster_ids if cluster_id in tcs_time}

    speed_in_time = np.array(speed_in_time)
    pos_in_time = np.array(pos_in_time) 
    trial_number_in_time = np.array(trial_number_in_time)
    dt_in_time = np.array(dt_in_time)
    tns_to_decode_with = np.array(beh['trials']['number'])
    tns_to_decode_with = tns_to_decode_with[tns_to_decode_with<=np.nanmax(trial_numbers)]
    trial_types = np.array(beh['trials']['type'])

    tns_to_decode = np.array(beh['trials']['number']) # decode all trials to visualise
    tns_to_train = np.array(beh['trials']['number'][np.isin(beh['trials']['type'], np.array(['b','nb']))]) 
    tns_to_decode = tns_to_decode[tns_to_decode<=np.nanmax(trial_numbers)] # handles last ephys trials
    tns_to_train = tns_to_train[tns_to_train<=np.nanmax(trial_numbers)] # handles last ephys trials

    predictions_in_time, errors_in_time = cross_validate_decoder_time(tcs_time, 
                                                    true_position=pos_in_time, 
                                                    trial_numbers=trial_number_in_time, 
                                                    tns_to_decode=tns_to_decode, 
                                                    tns_to_train=tns_to_train, 
                                                    tl=tl, bs=bs, train=0.9, n=10, verbose=False)

    avg_predictions_in_time = [np.mean(np.stack(preds_n), axis=0) for preds_n in predictions_in_time]
    avg_predictions_in_time = np.concatenate(avg_predictions_in_time).ravel()

    # create new stop mask
    interpf = interp1d(time_vals, avg_predictions_in_time, kind='nearest', fill_value=np.nan, bounds_error=False)
    avg_predictions_in_time_interp = interpf(dt.index)
    projected_stops_mask = (avg_predictions_in_time_interp>90) & (avg_predictions_in_time_interp<110)

    plot_stops(beh, savepath=f'{figpath}/M{mouse}D{day}_stops'+label,tl=tl,sort=False,return_fig=False,stop_mask=projected_stops_mask, last_ephys_bin=last_ephys_bin)
    plot_stops(beh, savepath=f'{figpath}/M{mouse}D{day}_stops_sorted'+label,tl=tl,sort=True,return_fig=False,stop_mask=projected_stops_mask, last_ephys_bin=last_ephys_bin)
    return




def compare_decodings(mouse, day, cluster_ids_1, cluster_ids_2, label1='', label2='', figpath=''):
    print(f'for {label1}, there are {len(cluster_ids_1)} cells being used for decoding')
    print(f'for {label2}, there are {len(cluster_ids_2)} cells being used for decoding')
    tcs, tcs_time, autocorrs, last_ephys_bin, beh, clusters = compute_vr_tcs(mouse, day)
    trial_groups, trial_colors = get_trial_groups_and_colors(beh, last_ephys_bin, tl, bs)
    sorted_trial_indices, sorted_trial_colors = get_sorted_trials_and_colors(beh, last_ephys_bin, tl, bs)

    tns = beh['trial_number']
    dt = beh['travel']-((tns[0]-1)*tl)
    n_bins = int(int(((np.ceil(np.nanmax(dt))//tl)+1)*tl)/bs)
    max_bound = int(((np.ceil(np.nanmax(dt))//tl)+1)*tl)
    min_bound = 0
    dt_bins = np.arange(0,max_bound,bs)
 
    x_true_dt = dt_bins[:last_ephys_bin]
    true_position = x_true_dt%tl
    trial_numbers = (x_true_dt//tl)+beh['trials']['number'][0]
    tns_to_decode_with = np.array(beh['trials']['number'])
    tns_to_decode_with = tns_to_decode_with[tns_to_decode_with<=np.nanmax(trial_numbers)]
    trial_types = np.array(beh['trials']['type'])

    tns_to_decode = np.array(beh['trials']['number']) # decode all trials to visualise
    tns_to_train = np.array(beh['trials']['number'][np.isin(beh['trials']['type'], np.array(['b','nb']))]) 
    tns_to_decode = tns_to_decode[tns_to_decode<=np.nanmax(trial_numbers)] # handles last ephys trials
    tns_to_train = tns_to_train[tns_to_train<=np.nanmax(trial_numbers)] # handles last ephys trials
    trial_types = trial_types[:len(tns_to_decode)]

    tcs_1 = {cluster_id: tcs[cluster_id] for cluster_id in cluster_ids_1 if cluster_id in tcs}
    predictions_1, errors = cross_validate_decoder(tcs_1, true_position, trial_numbers, tns_to_decode, tns_to_train, tl, bs, train=0.9, n=10, verbose=False)
    avg_predictions_1 = circular_nanmean(predictions_1, tl, axis=2)

    tcs_2 = {cluster_id: tcs[cluster_id] for cluster_id in cluster_ids_2 if cluster_id in tcs}
    predictions_2, errors = cross_validate_decoder(tcs_2, true_position, trial_numbers, tns_to_decode, tns_to_train, tl, bs, train=0.9, n=10, verbose=False)
    avg_predictions_2 = circular_nanmean(predictions_2, tl, axis=2)

    delta = avg_predictions_1-avg_predictions_2

    norm = TwoSlopeNorm(vmin=-35, vcenter=0, vmax=35)
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(0.8, 2), width_ratios=[1,0.05], sharey=True)
    x = np.arange(1, len(avg_predictions_1)+1)
    y = np.arange(0, len(avg_predictions_1[0])*bs, bs)
    X, Y = np.meshgrid(x, y)
    heatmap = ax[0].pcolormesh(Y, X, delta.T, shading='auto', cmap='bwr', norm=norm)
    heatmap.set_rasterized(True)
    ax[0].set_xlabel('Pos. (cm)')
    ax[1].axis('off')
    ax[1].scatter(np.ones(len(trial_colors)), 
                  np.arange(0,len(trial_colors)), 
                  c = trial_colors,
                  marker='s')
    ax[0].set_xlim(0,tl)
    ax[0].set_ylim(0,len(avg_predictions_1))
    ax[0].invert_yaxis()
    fig.savefig(f'{figpath}/compare_decoders_{label1}_{label2}_M{mouse}D{day}.pdf', dpi=300, bbox_inches='tight')


    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(0.8, 2), width_ratios=[1,0.05], sharey=True)
    x = np.arange(1, len(avg_predictions_1)+1)
    y = np.arange(0, len(avg_predictions_1[0])*bs, bs)
    X, Y = np.meshgrid(x, y)
    heatmap = ax[0].pcolormesh(Y, X, delta[sorted_trial_indices].T, shading='auto', cmap='bwr', norm=norm)
    heatmap.set_rasterized(True)
    ax[0].set_xlabel('Pos. (cm)')
    ax[1].axis('off')
    ax[1].scatter(np.ones(len(sorted_trial_colors)), 
                  np.arange(0,len(sorted_trial_colors)), 
                  c = sorted_trial_colors,
                  marker='s')
    ax[0].set_xlim(0,tl)
    ax[0].set_ylim(0,len(avg_predictions_1))
    ax[0].invert_yaxis()
    fig.savefig(f'{figpath}/compare_decoders_{label1}_{label2}_M{mouse}D{day}_sorted.pdf', dpi=300, bbox_inches='tight')

    avg_b_delta = np.nanmean(delta[sorted_trial_indices][:len(trial_types[trial_types=='b'])], axis=0)
    avg_nb_delta = np.nanmean(delta[sorted_trial_indices][len(trial_types[trial_types=='b']):], axis=0)
    sem_b_delta = stats.sem(delta[sorted_trial_indices][:len(trial_types[trial_types=='b'])], axis=0, nan_policy='omit')
    sem_nb_delta = stats.sem(delta[sorted_trial_indices][len(trial_types[trial_types=='b']):], axis=0, nan_policy='omit')

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(7, 2), width_ratios=[1,1], sharey=True)
    y = np.arange(0, len(avg_b_delta)*bs, bs)
    ax[0].set_xlabel('Pos. (cm)')
    ax[1].set_xlabel('Pos. (cm)')
    ax[0].set_ylabel('delta (cm)')
    ax[0].set_xlim(0,tl)
    ax[0].plot(y, avg_b_delta, color='tab:blue')
    ax[0].plot(y, avg_nb_delta, color='tab:orange')
    ax[0].fill_between(y, avg_b_delta+sem_b_delta, avg_b_delta-sem_b_delta, color='tab:blue', alpha=0.3, linewidth=0, edgecolor='none')
    ax[0].fill_between(y, avg_nb_delta+sem_nb_delta, avg_nb_delta-sem_nb_delta, color='tab:orange', alpha=0.3, linewidth=0, edgecolor='none')
    ax[1].plot(y, avg_b_delta-avg_nb_delta, color='black')
    fig.savefig(f'{figpath}/compare_decoders_{label1}_{label2}_M{mouse}D{day}_diff.pdf', dpi=300, bbox_inches='tight')


    # compare decoding accuracies
    unique_trial_groups = np.unique(trial_groups)
    print(unique_trial_groups)
    print(trial_groups.shape)
    print(trial_groups[0])
    
    decoding_accuracy = {}
    for group_label in unique_trial_groups:
        group_mask = trial_groups==group_label
        group_decoding_1 = avg_predictions_1[group_mask]
        group_decoding_2 = avg_predictions_2[group_mask]

        true_locations = np.tile(np.arange(0,tl,bs), len(group_decoding_1)).reshape((len(group_decoding_1), int(tl/bs)))
        errors_1 = true_locations - group_decoding_1
        errors_2 = true_locations - group_decoding_2
        errors_1[errors_1>(tl*0.75)] = tl-errors_1[errors_1>(tl*0.75)]
        errors_1[errors_1<(-tl*0.75)] = tl+errors_1[errors_1<(-tl*0.75)]
        errors_2[errors_2>(tl*0.75)] = tl-errors_2[errors_2>(tl*0.75)]
        errors_2[errors_2<(-tl*0.75)] = tl+errors_2[errors_2<(-tl*0.75)]
        errors_1 = np.abs(errors_1)
        errors_2 = np.abs(errors_2)
        trial_errors_1 = np.nanmean(errors_1,axis=0)
        trial_errors_2 = np.nanmean(errors_2,axis=0)
        decoding_accuracy[label1+group_label] = trial_errors_1
        decoding_accuracy[label2+group_label] = trial_errors_2

    # Create a violin plot
    plt.figure(figsize=(5, 2))
    plt.violinplot([decoding_accuracy[key] for key in decoding_accuracy.keys()], showmeans=True,showmedians=True)
    plt.xticks(np.arange(len(decoding_accuracy.keys())), decoding_accuracy.keys(), rotation=30)
    plt.xlabel('Trial Groups')
    plt.ylabel('Mean Decoding Error (cm)')
    plt.tight_layout()
    plt.show()


    decoding_accuracy = {}
    differences = {}
    for tt_label in ['b', 'nb']:
        tt_mask = trial_types==tt_label
        print(f'np.unque(tt_mask) {np.unique(trial_types)}')
        print(f'shape(tt_mask) {np.shape(tt_mask)}')
        print(f'sum(tt_mask) {np.sum(tt_mask)}')

        decoding_1 = avg_predictions_1[tt_mask]
        decoding_2 = avg_predictions_2[tt_mask]
        diff = decoding_2-decoding_1
        diff[diff>(tl*0.5)] = tl-diff[diff>(tl*0.5)]
        diff[diff<(-tl*0.5)] = tl+diff[diff<(-tl*0.5)]

        true_locations = np.tile(np.arange(0,tl,bs), len(decoding_1)).reshape((len(decoding_1), int(tl/bs)))
        print(f'shape(true_locations) {np.shape(true_locations)}')
        print(f'shape(decoding_1) {np.shape(decoding_1)}')

        errors_1 = true_locations - decoding_1
        errors_2 = true_locations - decoding_2
        errors_1[errors_1>(tl*0.5)] = tl-errors_1[errors_1>(tl*0.5)]
        errors_1[errors_1<(-tl*0.5)] = tl+errors_1[errors_1<(-tl*0.5)]
        errors_2[errors_2>(tl*0.5)] = tl-errors_2[errors_2>(tl*0.5)]
        errors_2[errors_2<(-tl*0.5)] = tl+errors_2[errors_2<(-tl*0.5)]
        errors_1 = np.abs(errors_1)
        errors_2 = np.abs(errors_2)

        trial_errors_1 = np.nanmean(errors_1,axis=0)
        trial_errors_2 = np.nanmean(errors_2,axis=0)
        decoding_accuracy[label1+'_'+tt_label] = trial_errors_1
        decoding_accuracy[label2+'_'+tt_label] = trial_errors_2
        differences[tt_label] = np.nanmean(diff, axis=0)

    # Create a violin plot
    plt.figure(figsize=(2, 2))
    plt.violinplot([decoding_accuracy[key] for key in decoding_accuracy.keys()], showmeans=True,showmedians=True)
    #plt.xticks(np.arange(1,1+len(decoding_accuracy.keys())), decoding_accuracy.keys(), rotation=30)
    plt.xlabel('Trial Groups')
    plt.ylabel('Mean Decoding Error (cm)')
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(2, 2))
    plt.violinplot([differences[key] for key in differences.keys()], showmeans=True,showmedians=True)
    #plt.xticks(np.arange(1, 1+len(differences.keys())), differences.keys(), rotation=30)
    plt.xlabel('Trial Groups')
    plt.ylabel('Diff in Decoding (cm)')
    plt.tight_layout()
    plt.show()


def extract_border(image, color):
    mask = np.all(image == color, axis=-1)

    rows, cols = mask.shape
    border_mask = np.copy(mask)
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if mask[i, j] == 1:
                neighbors = np.array([
                    mask[i-1, j-1], mask[i-1, j], mask[i-1, j+1],
                    mask[i, j-1],                   mask[i, j+1],
                    mask[i+1, j-1], mask[i+1, j], mask[i+1, j+1]
                ])
                if np.all(neighbors):
                    border_mask[i, j] = 0

    border_points = np.column_stack(np.where(border_mask == 1))
    border_points = np.column_stack(np.where(border_mask == 1))

    return border_points



def get_task_anchored_labels_in_time(mouse, day, cluster_ids_for_spectrogram):
    tcs, tcs_time, autocorrs, last_ephys_bin, beh, clusters = compute_vr_tcs(mouse, day, apply_zscore=False) # recompute tcs without z-scoring to use in the spectrogram
    tcs_to_use = {cluster_id: tcs[cluster_id] for cluster_id in cluster_ids_for_spectrogram if cluster_id in tcs}
    results = spectral_analysis(tcs_to_use, tl, bs=bs)
    spectrograms = results[3] 
    S = spectrograms.mean(0)
    max_peaks = np.argmax(S, axis=0)
    labels = np.isin(max_peaks, [12, 28, 44, 60, 76]).astype(int) # these are the peaks that correspond to the task anchoring
    trial_labels = get_kmeans_spatial_labels(tcs[cluster_ids_for_spectrogram[0]], labels, bs=bs, tl=tl) # reuse kmeans function to get the labels based on the task anchoring

    last_ephys_time_bin = clusters[clusters.index[0]].count(bin_size=time_bs, time_units = 'ms').index[-1]

    # time binned variables for later
    ep = nap.IntervalSet(start=0, end=last_ephys_time_bin, time_units = 's')
    speed_in_time = beh['S'].bin_average(bin_size=time_bs, time_units = 'ms', ep=ep)
    dt_in_time = beh['travel'].bin_average(bin_size=time_bs, time_units = 'ms', ep=ep)-((beh['trial_number'][0]-1)*tl)
    time_vals = np.arange(0, len(speed_in_time)*(time_bs/1000),time_bs/1000) # secs
    pos_in_time = dt_in_time%tl
    trial_number_in_time = (dt_in_time//tl)+beh['trial_number'][0]

    trial_number_in_time = pd.Series(np.array(trial_number_in_time)).fillna(method='ffill').fillna(method='bfill').to_numpy()
    trial_labels_trial_numbers = np.arange(trial_number_in_time[0], len(trial_labels)+trial_number_in_time[0], 1)
    trial_labels_in_time = np.zeros(len(trial_number_in_time), dtype=int)
    assert len(trial_labels) == len(trial_labels_trial_numbers), f'len(trial_labels)={len(trial_labels)} != len(trial_labels_trial_numbers)={len(trial_labels_trial_numbers)}'

    for i, tn in enumerate(trial_number_in_time.astype(int)):
        trial_labels_in_time[i] = trial_labels[np.where(trial_labels_trial_numbers == tn)[0][0]].astype(int)
    
    return trial_labels_in_time


from astropy.convolution import convolve, Gaussian1DKernel
def cross_correlation_with_jitter(
    cell1_spikes, cell2_spikes, bin_size=1, max_lag_ms=100, jitter_window_ms=5, n_jitters=1000, duration_s=2000, gauss_sigma_bins=1
):
    """
    Compute cross-correlation between two spike trains and assess significance by jittering cell2_spikes.
    Applies Gaussian smoothing to binned spike trains.
    Returns:
        lags, original_cc, jittered_ccs (n_jitters x len(lags)), pvals (per lag),
        optimal_lag (ms), optimal_corr (value at optimal lag)
    """
    n_bins = int(duration_s * (1000/bin_size))
    # Bin both spike trains
    binned1, _ = np.histogram(cell1_spikes, bins=n_bins, range=(0, duration_s))
    binned2, _ = np.histogram(cell2_spikes, bins=n_bins, range=(0, duration_s))
    # Gaussian smoothing
    gauss_kernel = Gaussian1DKernel(gauss_sigma_bins)
    binned1 = convolve(binned1, gauss_kernel)
    binned2 = convolve(binned2, gauss_kernel)
    # Compute original cross-correlation
    max_lag_bins = int(max_lag_ms // bin_size)
    lags = np.arange(-max_lag_bins, max_lag_bins + 1)
    original_cc = correlate(binned1, binned2, mode='full')
    center = len(original_cc) // 2
    original_cc = original_cc[center - max_lag_bins : center + max_lag_bins + 1]
    # Jitter cell2_spikes and compute surrogate cross-correlations
    jittered_ccs = []
    for _ in range(n_jitters):
        jittered_spikes = cell2_spikes + np.random.uniform(-jitter_window_ms, jitter_window_ms, size=cell2_spikes.shape)
        jittered_spikes = np.clip(jittered_spikes, 0, duration_s)
        binned2_jit, _ = np.histogram(jittered_spikes, bins=n_bins, range=(0, duration_s))
        binned2_jit = convolve(binned2_jit, gauss_kernel)
        cc = correlate(binned1, binned2_jit, mode='full')
        cc = cc[center - max_lag_bins : center + max_lag_bins + 1]
        jittered_ccs.append(cc)
    jittered_ccs = np.array(jittered_ccs)
    pvals = np.mean(jittered_ccs >= original_cc, axis=0)
    optimal_idx = np.argmax(original_cc)
    optimal_lag = lags[optimal_idx] * bin_size
    optimal_corr = original_cc[optimal_idx]


    return lags * bin_size, original_cc, jittered_ccs, pvals, optimal_lag, optimal_corr


def cross_correlation_with_jitter(
    cell1_spikes, cell2_spikes, bin_size=1, max_lag_ms=100, jitter_window_ms=5, 
    n_jitters=1000, duration_s=2000, gauss_sigma_bins=1, plot=True
):
    """
    Compute cross-correlation between two spike trains and assess significance by jittering cell2_spikes.
    Applies Gaussian smoothing to binned spike trains.
    Returns:
        lags, original_cc, jittered_ccs (n_jitters x len(lags)), pvals (per lag),
        optimal_lag (ms), optimal_corr (value at optimal lag), noise_correlation
    """

    def normalized_cross_correlation(x, y, max_lag_bins=max_lag_ms // bin_size):
        x = (x - np.mean(x)) / (np.std(x) + 1e-10)
        y = (y - np.mean(y)) / (np.std(y) + 1e-10)
        cc = correlate(x, y, mode='full')
        center = len(cc) // 2
        cc = cc[center - max_lag_bins : center + max_lag_bins + 1]
        cc = cc / len(x)  # Normalize by number of samples
        return cc

    n_bins = int(duration_s * (1000/bin_size))
    # Bin both spike trains
    binned1, _ = np.histogram(cell1_spikes, bins=n_bins, range=(0, duration_s))
    binned2, _ = np.histogram(cell2_spikes, bins=n_bins, range=(0, duration_s))
    # Gaussian smoothing
    gauss_kernel = Gaussian1DKernel(gauss_sigma_bins)
    binned1 = convolve(binned1, gauss_kernel)
    binned2 = convolve(binned2, gauss_kernel)
    # Compute original cross-correlation
    max_lag_bins = int(max_lag_ms // bin_size)
    lags = np.arange(-max_lag_bins, max_lag_bins + 1)
    original_cc = normalized_cross_correlation(binned1, binned2)
    center = len(original_cc) // 2
    original_cc = original_cc[center - max_lag_bins : center + max_lag_bins + 1]

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 1.7))
        plt.plot(lags * bin_size, original_cc, label='Original CC', color='blue')
        plt.title('Original Cross-Correlation')
        plt.xlabel('Lag (ms)')
        plt.ylabel('Correlation')
        plt.axhline(0, color='black', linestyle='--')
        plt.legend()
        plt.show()

    # Jitter cell2_spikes and compute surrogate cross-correlations
    jittered_ccs = []
    jittered_max_corrs = []
    for i in range(n_jitters):
        if i % 10 == 0:
            print(f'Jitter iteration {i+1}/{n_jitters}')
        jittered_spikes = cell2_spikes + np.random.uniform(-jitter_window_ms, jitter_window_ms, size=cell2_spikes.shape)
        jittered_spikes = np.clip(jittered_spikes, 0, duration_s)
        binned2_jit, _ = np.histogram(jittered_spikes, bins=n_bins, range=(0, duration_s))
        binned2_jit = convolve(binned2_jit, gauss_kernel)
        cc = normalized_cross_correlation(binned1, binned2_jit)
        cc = cc[center - max_lag_bins : center + max_lag_bins + 1]
        jittered_ccs.append(cc)
        jittered_max_corrs.append(np.max(cc))
    jittered_ccs = np.array(jittered_ccs)

    if plot:
        # plot 4 jittered cross-correlations in a subplot nrows=4
        fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(8, 5), squeeze=False,sharex=True)
        ax[0, 0].set_title(f'Jittered Cross-Correlations')
        for j in [0,1,2,3]:
            ax[j, 0].plot(lags * bin_size, jittered_ccs[j], color='red', alpha=1)
            ax[j, 0].set_xlabel('Lag (ms)')
            ax[j, 0].set_ylabel('Correlation')
            ax[j, 0].axhline(0, color='black', linestyle='--')
        plt.tight_layout()
        plt.show()

    jittered_max_corrs = np.array(jittered_max_corrs)
    pvals = np.mean(jittered_ccs >= original_cc, axis=0)
    optimal_idx = np.argmax(original_cc)
    optimal_lag = lags[optimal_idx] * bin_size
    optimal_corr = original_cc[optimal_idx]
    # Noise correlation: original max - median jittered max
    noise_correlation = optimal_corr - np.median(jittered_max_corrs)

    if plot:
        plt.figure(figsize=(3,3))
        plt.hist(jittered_max_corrs, bins=30, alpha=0.5, color='gray', label='Jittered Max Correlations')
        plt.axvline(np.median(jittered_max_corrs), color='red', linestyle='--', label='Median Jittered Max Correlation')
        plt.axvline(optimal_corr, color='blue', linestyle='--', label='Original Max Correlation')
        plt.title('Distribution of Jittered Max Correlations')
        plt.xlabel('Max Correlation')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    return lags * bin_size, original_cc, jittered_ccs, pvals, optimal_lag, optimal_corr, noise_correlation


def load_cluster_locations(clusters, cells):
    for column in ['coord_SCs_x', 
                   'coord_SCs_y', 
                   'coord_SCs_z', 
                   'coord_probe_x', 
                   'coord_probe_y',
                   'brain_region']:
        vals = []
        for id in cells.cluster_id:
            vals.append(clusters[column][id])
        cells[column] = vals
    return cells


def extract_border(image, color, only_border=True):
    mask = image == color

    rows, cols = mask.shape
    border_mask = np.copy(mask)
    
    if only_border:
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if mask[i, j] == 1:
                    neighbors = np.array([
                        mask[i-1, j-1], mask[i-1, j], mask[i-1, j+1],
                        mask[i, j-1],                   mask[i, j+1],
                        mask[i+1, j-1], mask[i+1, j], mask[i+1, j+1]
                    ])
                    if np.all(neighbors):
                        border_mask[i, j] = 0
     
    points = np.column_stack(np.where(border_mask == 1))
    return points


def get_annotation_colors_2D(annotations):
    # set colors based on annotations
    annotation_colors = np.zeros((annotations.shape[0], annotations.shape[1]), dtype=object)
    # Set colors to black where annotation contains 'VIS'
    for y in range(annotation_colors.shape[0]):
        for x in range(annotation_colors.shape[1]):
            if 'VIS' in str(annotations[y, x]):
                annotation_colors[y, x] = 'black'
            elif 'ENT' in str(annotations[y, x]):
                annotation_colors[y, x] = 'silver'
            elif 'RSP' in str(annotations[y, x]):
                annotation_colors[y, x] = 'grey'
            elif 'SUB' in str(annotations[y, x]):
                annotation_colors[y, x] = 'gainsboro'
            elif 'PAR' in str(annotations[y, x]):
                annotation_colors[y, x] = 'dimgrey'
            elif 'PRE' in str(annotations[y, x]):
                annotation_colors[y, x] = 'dimgrey'
            elif 'POST' in str(annotations[y, x]):
                annotation_colors[y, x] = 'lightslategrey'
            elif 'HPF' in str(annotations[y, x]):
                annotation_colors[y, x] = 'lightslategrey'
            elif 'ECT' in str(annotations[y, x]):
                annotation_colors[y, x] = 'lightslategrey'
            elif 'APr' in str(annotations[y, x]):
                annotation_colors[y, x] = 'darkgrey'
            else:
                annotation_colors[y, x] = 'white'
    return annotation_colors



def reconstruct_shank_id(clusters_df, mouse, colname='unit_location_x'):
    # M21 was implanted in reverse: remap probe_x values so shank 0 is at the
    # left (low x) and shank 3 at the right (high x), matching all other mice.
    # The 8 unique probe_x values and their flipped counterparts:
    _M21_X_REMAP = {
        0.: 782., 32.: 750.,
        250.: 532., 282.: 500.,
        500.: 282., 532.: 250.,
        750.: 32.,  782.: 0.,
    }

    if mouse == 21:
        unique_before = sorted(clusters_df[colname].dropna().unique())
        print(f"M21 unique {colname} values (before remap): {unique_before}")
        clusters_df = clusters_df.copy()
        clusters_df[colname] = clusters_df[colname].apply(
            lambda v: _M21_X_REMAP.get(float(v), v)
        )
        unique_after = sorted(clusters_df[colname].dropna().unique())
        print(f"M21 unique {colname} values (after remap):  {unique_after}")

    shank_ids = []
    for index, cluster in clusters_df.iterrows():
        x_pos = cluster[colname]
        if x_pos <= 150:
            shank_id = 0
        elif x_pos <= 400:
            shank_id = 1
        elif x_pos <= 650:
            shank_id = 2
        else:
            shank_id = 3
        shank_ids.append(shank_id)
    clusters_df['shank_id'] = shank_ids
    return clusters_df
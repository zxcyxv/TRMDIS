"""
Feature Engineering Script (K=10 Events)
축구 이벤트 데이터에서 마지막 10개 이벤트를 와이드 포맷으로 변환

총 피처 수: 165개
- A. 전체 10개 이벤트: 11개 × 10 = 110개 (is_last 포함)
- B. 첫 9개 이벤트 (마스킹): 6개 × 9 = 54개
- C. 에피소드 레벨: 1개

제외된 피처: team_id_enc, is_home, period_id (패스 위치 예측에 무관)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Constants
K = 8  # Number of last events to use
FIELD_X = 105
FIELD_Y = 68
GOAL_X = 105
GOAL_Y = 34  # Goal center

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / 'open_track1'


def load_train_data():
    """Load training data"""
    return pd.read_csv(DATA_DIR / 'train.csv')


def load_test_data():
    """Load test data - combine all episode CSVs"""
    test_index = pd.read_csv(DATA_DIR / 'test.csv')

    dfs = []
    for _, row in test_index.iterrows():
        path = DATA_DIR / row['path'].replace('./', '')
        df = pd.read_csv(path)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True), test_index


def create_event_features(df):
    """Create features for each event"""
    df = df.copy()

    # Convert is_home to int (handles 'True'/'False' strings)
    df['is_home'] = df['is_home'].map({'True': 1, 'False': 0, True: 1, False: 0}).fillna(0).astype(int)

    # Time delta (seconds between events)
    df['dt'] = df.groupby('game_episode')['time_seconds'].diff().fillna(0)

    # Position deltas
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']

    # Distance and speed
    df['dist'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['speed'] = np.where(df['dt'] > 0, df['dist'] / df['dt'], 0)

    # X zone (7 zones: 0=defense, 6=attack)
    df['x_zone'] = (df['start_x'] / (FIELD_X / 7)).astype(int).clip(0, 6)

    # Y lane (3 lanes: 0=left, 1=center, 2=right)
    df['lane'] = pd.cut(df['start_y'], bins=[0, FIELD_Y/3, 2*FIELD_Y/3, FIELD_Y],
                        labels=[0, 1, 2], include_lowest=True).astype(int)

    # Distance and angle to goal
    df['dist_to_goal'] = np.sqrt((GOAL_X - df['start_x'])**2 + (GOAL_Y - df['start_y'])**2)
    df['angle_to_goal'] = np.degrees(np.arctan2(GOAL_Y - df['start_y'], GOAL_X - df['start_x']))

    # Normalized event index within episode
    df['ep_idx_norm'] = df.groupby('game_episode')['action_id'].transform(
        lambda x: (x - x.min()) / max(x.max() - x.min(), 1)
    )

    return df


def encode_categorical(df, encoders=None):
    """Encode categorical variables"""
    df = df.copy()

    if encoders is None:
        encoders = {}

    # Type encoding
    if 'type_name' not in encoders:
        encoders['type_name'] = LabelEncoder()
        df['type_id'] = encoders['type_name'].fit_transform(df['type_name'].fillna('Unknown'))
    else:
        df['type_id'] = df['type_name'].fillna('Unknown').apply(
            lambda x: encoders['type_name'].transform([x])[0] if x in encoders['type_name'].classes_ else -1
        )

    # Result encoding
    if 'result_name' not in encoders:
        encoders['result_name'] = LabelEncoder()
        df['res_id'] = encoders['result_name'].fit_transform(df['result_name'].fillna('Unknown'))
    else:
        df['res_id'] = df['result_name'].fillna('Unknown').apply(
            lambda x: encoders['result_name'].transform([x])[0] if x in encoders['result_name'].classes_ else -1
        )

    # Team encoding
    if 'team_id' not in encoders:
        encoders['team_id'] = LabelEncoder()
        df['team_id_enc'] = encoders['team_id'].fit_transform(df['team_id'].astype(str))
    else:
        df['team_id_enc'] = df['team_id'].astype(str).apply(
            lambda x: encoders['team_id'].transform([x])[0] if x in encoders['team_id'].classes_ else -1
        )

    return df, encoders


def create_wide_features(df, k=K):
    """Convert to wide format with last K events (vectorized)"""

    # Features present in all K events
    all_event_features = [
        'start_x', 'start_y', 'dt', 'ep_idx_norm',
        'x_zone', 'lane', 'dist_to_goal', 'angle_to_goal',
        'type_id', 'res_id', 'is_home'
    ]

    # Features masked for last event (target leakage prevention)
    masked_features = ['end_x', 'end_y', 'dx', 'dy', 'dist', 'speed']

    # Sort and get last K events per episode
    df = df.sort_values(['game_episode', 'action_id'])
    df_last_k = df.groupby('game_episode').tail(k).copy()

    # Assign position index (0 to k-1) within each episode's last k events
    df_last_k['pos'] = df_last_k.groupby('game_episode').cumcount()

    # Get episode count to adjust positions for episodes with < k events
    ep_counts = df_last_k.groupby('game_episode').size()
    df_last_k['ep_count'] = df_last_k['game_episode'].map(ep_counts)
    df_last_k['pos'] = df_last_k['pos'] + (k - df_last_k['ep_count'])

    # Pivot each feature
    result = df_last_k[['game_episode']].drop_duplicates().set_index('game_episode')

    # All event features for all K positions
    for feat in all_event_features:
        pivot = df_last_k.pivot(index='game_episode', columns='pos', values=feat)
        pivot.columns = [f'{feat}_{i}' for i in pivot.columns]
        result = result.join(pivot)

    # Masked features (only for positions 0 to k-2)
    for feat in masked_features:
        pivot = df_last_k[df_last_k['pos'] < k - 1].pivot(
            index='game_episode', columns='pos', values=feat
        )
        pivot.columns = [f'{feat}_{i}' for i in pivot.columns]
        result = result.join(pivot)

    # Episode-level features from last event
    last_events = df_last_k[df_last_k['pos'] == k - 1][
        ['game_episode', 'res_id', 'result_name', 'end_x', 'end_y']
    ].set_index('game_episode')
    last_events.columns = ['last_result_encoded', 'last_result_name', 'target_end_x', 'target_end_y']
    result = result.join(last_events)

    return result.reset_index()


def get_feature_columns():
    """Get list of feature columns (excluding ID and target)"""
    features = []

    # All event features (K events)
    all_event_features = [
        'start_x', 'start_y', 'dt', 'ep_idx_norm',
        'x_zone', 'lane', 'dist_to_goal', 'angle_to_goal',
        'type_id', 'res_id', 'is_home'
    ]

    for i in range(K):
        for feat in all_event_features:
            features.append(f'{feat}_{i}')

    # Masked features (K-1 events)
    masked_features = ['end_x', 'end_y', 'dx', 'dy', 'dist', 'speed']
    for i in range(K - 1):
        for feat in masked_features:
            features.append(f'{feat}_{i}')

    # Episode-level
    features.append('last_result_encoded')

    return features


def process_train_data():
    """Process training data and return features with targets"""
    print("Loading training data...")
    df = load_train_data()
    print(f"Loaded {len(df)} events, {df['game_episode'].nunique()} episodes")

    print("Creating event features...")
    df = create_event_features(df)

    print("Encoding categorical variables...")
    df, encoders = encode_categorical(df)

    print("Creating wide format features (K=10)...")
    wide_df = create_wide_features(df, k=K)
    print(f"Created {len(wide_df)} episode features")

    return wide_df, encoders


def process_test_data(encoders):
    """Process test data using fitted encoders"""
    print("Loading test data...")
    df, test_index = load_test_data()
    print(f"Loaded {len(df)} events, {df['game_episode'].nunique()} episodes")

    print("Creating event features...")
    df = create_event_features(df)

    print("Encoding categorical variables...")
    df, _ = encode_categorical(df, encoders)

    print("Creating wide format features (K=10)...")
    wide_df = create_wide_features(df, k=K)
    print(f"Created {len(wide_df)} episode features")

    return wide_df


def split_by_result(df):
    """Split data by last pass result (Successful vs Unsuccessful)"""
    successful = df[df['last_result_name'] == 'Successful'].copy()
    unsuccessful = df[df['last_result_name'] != 'Successful'].copy()

    print(f"Successful passes: {len(successful)} ({len(successful)/len(df)*100:.1f}%)")
    print(f"Unsuccessful passes: {len(unsuccessful)} ({len(unsuccessful)/len(df)*100:.1f}%)")

    return successful, unsuccessful


def main():
    """Main execution"""
    # Process training data
    train_wide, encoders = process_train_data()

    # Split by result
    print("\n--- Splitting by last pass result ---")
    train_success, train_fail = split_by_result(train_wide)

    # Save processed data
    print("\nSaving processed data...")
    train_wide.to_csv(DATA_DIR / 'train_features_k8.csv', index=False)
    train_success.to_csv(DATA_DIR / 'train_features_k8_success.csv', index=False)
    train_fail.to_csv(DATA_DIR / 'train_features_k8_fail.csv', index=False)

    # Process test data
    print("\n--- Processing test data ---")
    test_wide = process_test_data(encoders)
    test_wide.to_csv(DATA_DIR / 'test_features_k8.csv', index=False)

    # Print feature summary
    feature_cols = get_feature_columns()
    print(f"\n=== Feature Summary ===")
    print(f"Total features: {len(feature_cols)}")
    print(f"- All event features (11 × {K}): {11 * K}")
    print(f"- Masked features (6 × {K-1}): {6 * (K-1)}")
    print(f"- Episode-level features: 1")

    # Stats
    print(f"\n=== Data Stats ===")
    print(f"Train total: {len(train_wide)}")
    print(f"Train successful: {len(train_success)}")
    print(f"Train unsuccessful: {len(train_fail)}")
    print(f"Test total: {len(test_wide)}")

    return train_wide, train_success, train_fail, test_wide, encoders


if __name__ == '__main__':
    train_wide, train_success, train_fail, test_wide, encoders = main()

# FIFA Player Transfer Market Value Predictor
# run with: python3 app.py
# then open http://127.0.0.1:5000 in Safari

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import base64
import io
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

# loading dataset
print("\nloading dataset...")
df_raw = pd.read_csv('data.csv', encoding='latin1', low_memory=False)
print("dataset loaded, shape:", df_raw.shape)

def clean_currency(value):
    if pd.isnull(value):
        return 0
    value = str(value).encode('ascii', 'ignore').decode('ascii')
    value = value.replace('€', '').replace(',', '').strip()
    if 'M' in value:
        return float(value.replace('M', '')) * 1000000
    elif 'K' in value:
        return float(value.replace('K', '')) * 1000
    else:
        try:
            return float(value) if value else 0
        except:
            return 0

df_raw['Value'] = df_raw['Value'].apply(clean_currency)
df_raw['Wage']  = df_raw['Wage'].apply(clean_currency)
print("currency cleaned")

# features
features = [
    'Age', 'Overall', 'Potential', 'Wage',
    'International Reputation', 'Weak Foot', 'Skill Moves', 'Position',
    'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing',
    'Dribbling', 'BallControl', 'Acceleration', 'SprintSpeed',
    'Reactions', 'ShotPower', 'Stamina', 'Strength', 'Vision'
]
target = 'Value'

df = df_raw[df_raw['Value'] > 0].copy()
df_model = df[features + [target]].dropna()
print("model data shape:", df_model.shape)

le = LabelEncoder()
df_model = df_model.copy()
df_model['Position'] = le.fit_transform(df_model['Position'].astype(str))
positions_list = list(le.classes_)

X = df_model[features]
y = df_model[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# training models
print("\ntraining models...")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
print("linear regression done")

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("random forest done")

gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
print("gradient boosting done")

pred_lr = lr.predict(X_test_scaled)
pred_rf = rf.predict(X_test)
pred_gb = gb.predict(X_test)

scores = {
    'lr': round(r2_score(y_test, pred_lr) * 100, 2),
    'rf': round(r2_score(y_test, pred_rf) * 100, 2),
    'gb': round(r2_score(y_test, pred_gb) * 100, 2),
}
mae_scores = {
    'lr': round(mean_absolute_error(y_test, pred_lr) / 1e6, 2),
    'rf': round(mean_absolute_error(y_test, pred_rf) / 1e6, 2),
    'gb': round(mean_absolute_error(y_test, pred_gb) / 1e6, 2),
}
print("scores:", scores)

# safe helpers for NaN values
def safe_int(val, default):
    try:
        v = float(val)
        return default if np.isnan(v) else int(v)
    except:
        return default

def safe_float(val, default):
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except:
        return default

# famous players list
famous_players_names = [
    'L. Messi', 'Cristiano Ronaldo', 'Neymar Jr', 'K. De Bruyne',
    'E. Hazard', 'L. Modri?', 'M. Salah', 'S. Mané', 'K. Mbappé',
    'H. Kane', 'R. Lewandowski', 'T. Kroos', 'Sergio Ramos',
    'M. ter Stegen', 'Alisson', 'V. van Dijk', 'A. Griezmann',
    'P. Dybala', 'G. Bale', 'Isco'
]

famous_players_data = {}
for name in famous_players_names:
    # try exact match first then partial
    row = df_raw[df_raw['Name'] == name]
    if row.empty:
        row = df_raw[df_raw['Name'].str.contains(name.split('.')[-1].strip(), na=False)]
    if not row.empty:
        row = row.iloc[0]
        pos = str(row.get('Position', 'ST')).strip()
        if pos not in le.classes_:
            pos = 'ST'
        famous_players_data[name] = {
            'age':          safe_int(row.get('Age'), 25),
            'overall':      safe_int(row.get('Overall'), 85),
            'potential':    safe_int(row.get('Potential'), 85),
            'wage':         safe_float(row.get('Wage'), 100000),
            'reputation':   safe_int(row.get('International Reputation'), 3),
            'weakfoot':     safe_int(row.get('Weak Foot'), 3),
            'skillmoves':   safe_int(row.get('Skill Moves'), 3),
            'position':     pos,
            'crossing':     safe_int(row.get('Crossing'), 70),
            'finishing':    safe_int(row.get('Finishing'), 70),
            'heading':      safe_int(row.get('HeadingAccuracy'), 70),
            'passing':      safe_int(row.get('ShortPassing'), 70),
            'dribbling':    safe_int(row.get('Dribbling'), 70),
            'ballcontrol':  safe_int(row.get('BallControl'), 70),
            'acceleration': safe_int(row.get('Acceleration'), 70),
            'sprintspeed':  safe_int(row.get('SprintSpeed'), 70),
            'reactions':    safe_int(row.get('Reactions'), 70),
            'shotpower':    safe_int(row.get('ShotPower'), 70),
            'stamina':      safe_int(row.get('Stamina'), 70),
            'strength':     safe_int(row.get('Strength'), 70),
            'vision':       safe_int(row.get('Vision'), 70),
            'actual_value': safe_float(row.get('Value'), 0),
            'club':         str(row.get('Club', 'Unknown')),
            'nationality':  str(row.get('Nationality', 'Unknown')),
        }
        print(f"loaded: {name} ({famous_players_data[name]['club']})")
    else:
        print(f"NOT FOUND: {name}")

famous_list = [{'name': k, **v} for k, v in famous_players_data.items()]
print(f"\nloaded {len(famous_list)} famous players")
print("open http://127.0.0.1:5000 in Safari")


# convert figure to base64
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return b64


# generate all 8 graphs for a player
def generate_graphs(inp, predicted_value, player_name="Player"):
    graphs = []
    pred_m    = predicted_value / 1e6
    overall   = float(inp['Overall'])
    age       = float(inp['Age'])
    dribbling = float(inp['Dribbling'])
    sprintspd = float(inp['SprintSpeed'])
    finishing = float(inp['Finishing'])
    passing   = float(inp['ShortPassing'])
    reactions = float(inp['Reactions'])
    vision    = float(inp['Vision'])
    stamina   = float(inp['Stamina'])
    strength  = float(inp['Strength'])

    df_ref = df[['Name','Overall','Age','Value','Wage',
                 'Dribbling','SprintSpeed','Finishing',
                 'ShortPassing','Reactions','Vision',
                 'Stamina','Strength']].dropna().copy()

    plt.rcParams.update({
        'text.color': 'white', 'axes.labelcolor': '#888',
        'xtick.color': '#666', 'ytick.color': '#666'
    })

    # graph 1 - overall vs value scatter
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.scatter(df_ref['Overall'], df_ref['Value']/1e6, alpha=0.2, color='#3a86ff', s=12, label='All Players')
    ax.scatter([overall], [pred_m], color='#00ff87', s=220, zorder=6, marker='*', label=player_name)
    ax.annotate(f' {player_name} €{pred_m:.1f}M', xy=(overall, pred_m),
                color='#00ff87', fontsize=9, xytext=(overall+0.8, pred_m+1.5))
    ax.set_title('Overall Rating vs Market Value', color='white', fontsize=12, pad=10)
    ax.set_xlabel('Overall Rating')
    ax.set_ylabel('Market Value (€M)')
    ax.spines[:].set_color('#222')
    ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
    graphs.append(fig_to_b64(fig))

    # graph 2 - radar chart
    stat_vals   = [dribbling, sprintspd, finishing, passing, reactions, vision]
    stat_labels = ['Dribbling','Sprint\nSpeed','Finishing','Passing','Reactions','Vision']
    N      = len(stat_labels)
    angles = [n/N*2*np.pi for n in range(N)] + [0]
    vnorm  = [v/99 for v in stat_vals] + [stat_vals[0]/99]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#111827')
    ax.plot(angles, vnorm, color='#00ff87', linewidth=2.5)
    ax.fill(angles, vnorm, color='#00ff87', alpha=0.18)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(stat_labels, color='#ccc', fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25','50','75','99'], color='#555', fontsize=7)
    ax.spines['polar'].set_color('#333')
    ax.grid(color='#222', linewidth=0.6)
    ax.set_title(f'{player_name} — Skill Radar', color='white', fontsize=12, pad=18)
    graphs.append(fig_to_b64(fig))

    # graph 3 - similar rated players comparison
    similar = df_ref[
        (df_ref['Overall'] >= overall - 3) &
        (df_ref['Overall'] <= overall + 3)
    ].sort_values('Value', ascending=False).head(12).copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    names_l  = list(similar['Name']) + [f'★ {player_name}']
    vals_l   = list(similar['Value']/1e6) + [pred_m]
    cols     = ['#3a86ff'] * len(similar) + ['#00ff87']
    ax.barh(names_l, vals_l, color=cols, alpha=0.8, height=0.6)
    ax.set_title(f'Value vs Similar Rated Players (Overall ~{int(overall)})', color='white', fontsize=12, pad=10)
    ax.set_xlabel('Market Value (€M)')
    ax.spines[:].set_color('#222')
    graphs.append(fig_to_b64(fig))

    # graph 4 - age vs value
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.scatter(df_ref['Age'], df_ref['Value']/1e6, alpha=0.2, color='#ff6b6b', s=12, label='All Players')
    ax.scatter([age], [pred_m], color='#00ff87', s=220, zorder=6, marker='*', label=player_name)
    ax.set_title('Age vs Market Value', color='white', fontsize=12, pad=10)
    ax.set_xlabel('Age')
    ax.set_ylabel('Market Value (€M)')
    ax.spines[:].set_color('#222')
    ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
    graphs.append(fig_to_b64(fig))

    # graph 5 - all 3 model predictions
    input_df_full = pd.DataFrame(inp, index=[0])[features]
    input_scaled  = scaler.transform(input_df_full)
    p_lr = lr.predict(input_scaled)[0] / 1e6
    p_rf = rf.predict(input_df_full)[0] / 1e6
    p_gb = gb.predict(input_df_full)[0] / 1e6
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    bars = ax.bar(['Linear Regression','Random Forest','Gradient Boosting'],
                  [p_lr, p_rf, p_gb],
                  color=['#3a86ff','#f39c12','#00ff87'], alpha=0.85, width=0.5)
    for bar, val in zip(bars, [p_lr, p_rf, p_gb]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f'€{val:.1f}M', ha='center', color='white', fontsize=10)
    ax.set_title(f'All 3 Model Predictions — {player_name}', color='white', fontsize=12, pad=10)
    ax.set_ylabel('Predicted Value (€M)')
    ax.spines[:].set_color('#222')
    graphs.append(fig_to_b64(fig))

    # graph 6 - feature importance
    feat_imp = pd.DataFrame({
        'Feature': features,
        'Importance': gb.feature_importances_
    }).sort_values('Importance', ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis', ax=ax)
    ax.set_title('Feature Importance — What Drives Player Value', color='white', fontsize=12, pad=10)
    ax.spines[:].set_color('#222')
    graphs.append(fig_to_b64(fig))

    # graph 7 - player stats vs position average
    pos_enc   = float(inp['Position'])
    pos_rows  = df_model[df_model['Position'] == pos_enc]
    stat_cols = ['Dribbling','SprintSpeed','Finishing','ShortPassing','Reactions','Vision','Stamina','Strength']
    pos_avg   = [pos_rows[c].mean() if c in pos_rows.columns else 70 for c in stat_cols]
    p_vals    = [dribbling, sprintspd, finishing, passing, reactions, vision, stamina, strength]
    x = np.arange(len(stat_cols))
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.bar(x-0.2, pos_avg, width=0.38, color='#3a86ff', alpha=0.75, label='Position Avg')
    ax.bar(x+0.2, p_vals,  width=0.38, color='#00ff87', alpha=0.85, label=player_name)
    ax.set_xticks(x)
    ax.set_xticklabels(stat_cols, rotation=15, ha='right', fontsize=9)
    ax.set_title(f'{player_name} vs Position Average', color='white', fontsize=12, pad=10)
    ax.set_ylabel('Stat Value')
    ax.spines[:].set_color('#222')
    ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
    graphs.append(fig_to_b64(fig))

    # graph 8 - value distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    vals50 = df_ref[df_ref['Value'] < 50e6]['Value'] / 1e6
    ax.hist(vals50, bins=50, color='#3a86ff', edgecolor='#222', alpha=0.7, label='All Players')
    if pred_m <= 50:
        ax.axvline(pred_m, color='#00ff87', linewidth=2.5, linestyle='--',
                   label=f'{player_name}: €{pred_m:.1f}M')
        ax.text(pred_m+0.3, ax.get_ylim()[1]*0.85, f'€{pred_m:.1f}M', color='#00ff87', fontsize=10)
    ax.set_title('Market Value Distribution — Where Does This Player Sit?', color='white', fontsize=12, pad=10)
    ax.set_xlabel('Market Value (€M)')
    ax.set_ylabel('Number of Players')
    ax.spines[:].set_color('#222')
    ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
    graphs.append(fig_to_b64(fig))

    return graphs


@app.route('/')
def index():
    return render_template('index.html',
                           positions=positions_list,
                           scores=scores,
                           mae_scores=mae_scores,
                           famous_players=famous_list)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data          = request.json
        position_text = data.get('position', '').strip().upper()
        player_name   = data.get('player_name', 'Player')

        if position_text not in le.classes_:
            return jsonify({'error': f'Invalid position. Valid: {", ".join(le.classes_)}'}), 400

        inp = {
            'Age':                      float(data['age']),
            'Overall':                  float(data['overall']),
            'Potential':                float(data['potential']),
            'Wage':                     float(data['wage']),
            'International Reputation': float(data['reputation']),
            'Weak Foot':                float(data['weakfoot']),
            'Skill Moves':              float(data['skillmoves']),
            'Position':                 float(le.transform([position_text])[0]),
            'Crossing':                 float(data['crossing']),
            'Finishing':                float(data['finishing']),
            'HeadingAccuracy':          float(data['heading']),
            'ShortPassing':             float(data['passing']),
            'Dribbling':                float(data['dribbling']),
            'BallControl':              float(data['ballcontrol']),
            'Acceleration':             float(data['acceleration']),
            'SprintSpeed':              float(data['sprintspeed']),
            'Reactions':                float(data['reactions']),
            'ShotPower':                float(data['shotpower']),
            'Stamina':                  float(data['stamina']),
            'Strength':                 float(data['strength']),
            'Vision':                   float(data['vision'])
        }

        input_df  = pd.DataFrame(inp, index=[0])[features]
        predicted = gb.predict(input_df)[0]
        value_m   = predicted / 1e6
        graphs    = generate_graphs(inp, predicted, player_name)

        return jsonify({
            'display': f'€{value_m:.1f}M' if value_m >= 1 else f'€{predicted:,.0f}',
            'value_m': round(value_m, 1),
            'graphs':  graphs
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, port=5000)
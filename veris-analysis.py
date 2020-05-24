# Analysis of VERIS data taken from
# https://github.com/87bdharr/VERIS-Data-Analysis

import pandas as pd
import matplotlib.pyplot as plt
from verispy import VERIS
import numpy as np
import seaborn as sns
import plotly.express as px

########################################################################
# Load data
data_dir = "../VCDB/data/json/validated"
v = VERIS(json_dir=data_dir)

veris_df = v.json_to_df(verbose=True)
veris_df.shape

########################################################################
# Show a plot of the frequency of actor types
actor = v.enum_summary(veris_df, 'actor')
actor = actor[['enum', 'x']]
actor.columns = ['Actor Type', 'Count']

actor_freqs = v.enum_summary(veris_df, 'actor')
actor_freqs = actor_freqs[['enum', 'freq']]
actor_freqs['Percentage'] = actor_freqs['freq'].apply(lambda x: x*100)
'{0:.2f}'.format(actor_freqs['Percentage'][0])

fig1 = actor.set_index('Actor Type').T.plot(
    kind='bar', stacked=True, figsize=(7, 5),
    alpha=0.7, label=None, grid=False, width=0.3)
plt.annotate('{0:.2f}'.format(actor_freqs['Percentage'][0]), xy=[0, 2000])
plt.annotate('{0:.2f}'.format(actor_freqs['Percentage'][1]), xy=[0, 6000])
plt.annotate('{0:.2f}'.format(actor_freqs['Percentage'][2]), xy=[0, 8300])
plt.show()

########################################################################
# Show a frequency plot of internal and external actor types
internal_actor = v.enum_summary(veris_df, 'actor.internal.variety')
internal_actor.drop(internal_actor.index[15], inplace=True)
internal_actor = internal_actor[['enum', 'x']]
internal_actor.columns = ['Internal Actor Type', 'Count']
internal_actor = internal_actor.sort_values(by='Count', ascending=False)
internal_actor.head()

external_actor = v.enum_summary(veris_df, 'actor.external.variety')
external_actor.drop(external_actor.index[13], inplace=True)
external_actor = external_actor[['enum', 'x']]
external_actor.columns = ['External Actor Type', 'Count']
external_actor = external_actor.sort_values(by='Count', ascending=False)
external_actor.drop(external_actor.index[12], inplace=True)
external_actor.head()

internal_actor.set_index('Internal Actor Type').T

df4 = actor.set_index('Actor Type').T
df5 = internal_actor.set_index('Internal Actor Type').T
df6 = external_actor.set_index('External Actor Type').T

########################################################################
# Plot Internal Actors
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

df4.plot(kind='bar', stacked=True, ax=ax1, label='Count of Actor Types')
ax1.legend(bbox_to_anchor=(1, 1), fancybox=True, framealpha=1, shadow=True, borderpad=1)
ax1.set_title('Incidents by Actor Type', fontweight='bold', fontsize='15')

df5.plot(kind='bar', stacked=True, ax=ax2)
ax2.legend(bbox_to_anchor=(1, 1), fancybox=True, framealpha=1, shadow=True, borderpad=1)
ax2.set_title('Breakdown of Internal Actors by Type', fontweight='bold', fontsize='15')
fig2.tight_layout()

# Plot External Actors
fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(13, 6))

df4.plot(kind='bar', stacked=True, ax=ax3, label='Count of Actor Types')
ax3.legend(bbox_to_anchor=(1, 1), fancybox=True, framealpha=1, shadow=True, borderpad=1)
ax3.set_title('Incidents by Actor Type', fontweight='bold', fontsize='15')

df6.plot(kind='bar', stacked=True, ax=ax4)
ax4.legend(bbox_to_anchor=(1, 1), fancybox=True, framealpha=1, shadow=True, borderpad=1)
ax4.set_title('Breakdown of External Actors by Type', fontweight='bold', fontsize='15')
fig3.tight_layout()

########################################################################
# Plot frequency of Actions used
action = v.enum_summary(veris_df, 'action')

df_action = action[['enum', 'x']]
df_action.columns = ['Action', 'Count']
df_action['Count'].apply(pd.to_numeric, errors='coerce')
df_action['Count'].sum()
df_action

action_freqs = v.enum_summary(veris_df, 'action')
action_freqs = action_freqs[['enum', 'freq']]
action_freqs['Percentage'] = action_freqs['freq'].apply(lambda x: x*100)
action_freqs

sns.set()
palette = sns.color_palette("Paired")
sns.set_palette('Paired')
sns.set_style('whitegrid')
df_action.set_index('Action').T.plot(
    kind='barh', stacked=True, figsize=(12, 4),
    alpha=0.7, label=None)
x_ticks = np.arange(0, 11000, 1000)
plt.xticks(x_ticks)
plt.legend(
    bbox_to_anchor=(1, 1), loc=2, fancybox=True,
    framealpha=1, shadow=True, borderpad=1)
plt.annotate('{0:.2f}'.format(action_freqs['Percentage'][0]), xy=[1100, 0])
plt.annotate('{0:.2f}'.format(action_freqs['Percentage'][1]), xy=[3400, 0])
plt.annotate('{0:.2f}'.format(action_freqs['Percentage'][2]), xy=[5400, 0])
plt.annotate('{0:.2f}'.format(action_freqs['Percentage'][3]), xy=[7100, 0])
plt.annotate('{0:.2f}'.format(action_freqs['Percentage'][4]), xy=[8300, 0])
plt.annotate('{0:.2f}'.format(action_freqs['Percentage'][5]), xy=[8900, 0])
plt.show()

########################################################################
# Plot the frequency of an action over time
df = px.data
df

hacking = v.enum_summary(veris_df, 'action.hacking.variety')
hacking = hacking.drop(hacking[hacking.x < 5].index)
hacking

hacking.sort_values(by='x', ascending=True).plot(x='enum', y='x', kind='barh')

internal = v.enum_summary(veris_df, 'actor.internal.variety')
internal.head()

timeline_info = veris_df[[
    'timeline.incident.day',
    'timeline.incident.month',
    'timeline.incident.year']]
timeline_info = timeline_info.dropna(axis='rows')

timeline_info['timeline.incident.day'] = timeline_info[
        'timeline.incident.day'].astype(int)
timeline_info['timeline.incident.month'] = timeline_info[
    'timeline.incident.month'].astype(int)
timeline_info['timeline.incident.year'] = timeline_info[
    'timeline.incident.year'].astype(int)

timeline_info['timeline.incident.day'] = timeline_info[
    'timeline.incident.day'].astype(str)
timeline_info['timeline.incident.month'] = timeline_info[
    'timeline.incident.month'].astype(str)
timeline_info['timeline.incident.year'] = timeline_info[
    'timeline.incident.year'].astype(str)

timeline_info['combined date'] = \
    timeline_info['timeline.incident.year'] + '-' + \
    timeline_info['timeline.incident.month'] + '-' + \
    timeline_info['timeline.incident.day']

timeline_info.head()

incorrect_dates = timeline_info[
    timeline_info['combined date'] == '2018-2-29'].index
timeline_info.drop(incorrect_dates, inplace=True)

timeline_info['combined date'] = pd.to_datetime(timeline_info['combined date'])
timeline_info = timeline_info.drop(
    ['timeline.incident.day',
     'timeline.incident.month',
     'timeline.incident.year'], axis=1)

timeline_info.dtypes
timeline_info.describe()

########################################################################
# Plot a time line of hacking
df2 = timeline_info.join(veris_df)
df2 = df2.set_index('combined date')
Hacking = df2['2006':'2019']
Hacking = Hacking['action.Hacking']
Hacking.resample('M').sum().plot(kind='line')

# Work out the types of hacking and plot a stacked barchart over time
Hacking_Variety = df2.filter(like='action.hacking.variety')
Hacking_Variety = Hacking_Variety['2006':'2019']

column_list = Hacking_Variety \
    . loc[:, Hacking_Variety.sum() > 4] \
    .sum() \
    .sort_values(ascending=False) \
    .index.to_list()

HV = Hacking_Variety.loc[:, column_list].resample('Y').sum()
HV.rename(columns={
    'combined date': 'Combined Date',
    'action.hacking.variety.Unknown': 'Unkown',
    'action.hacking.variety.DoS': 'DoS',
    'action.hacking.variety.Exploit vuln': 'Exploit Vuln',
    'action.hacking.variety.Use of stolen creds': 'Stolen Creds',
    'action.hacking.variety.SQLi': 'SQLi',
    'action.hacking.variety.Other': 'Other',
    'action.hacking.variety.Brute force': 'Brute Force',
    'action.hacking.variety.Use of backdoor or C2': 'Backdoor or C2',
    'action.hacking.variety.Abuse of functionality': 'Abuse of functionality'},
          inplace=True)
HV.head()

ax1 = HV.iloc[:, 1:9].plot(kind='bar', stacked=True, figsize=(10, 5))
ax1.set_xticklabels([
    '2006', '2007', '2008', '2009', '2010', '2011', '2012',
    '2013', '2014', '2015', '2016', '2017', '2018', '2019'])
ax1.set_ylabel('Count of Hacking Events',
               fontsize=14, labelpad=20, fontweight='bold')
ax1.set_xlabel('Year', fontsize=14, labelpad=20, fontweight='bold')
plt.xticks(rotation=0)
plt.grid(False)
plt.tight_layout()

########################################################################
# Plot size of organisation affected
size = v.enum_summary(veris_df, 'victim.employee_count')
size

mapping = {'Small': 'Small',
           '1 to 10': 'Small',
           '11 to 100': 'Small',
           '101 to 1000': 'Small',
           '1001 to 10000': 'Medium',
           '10001 to 25000': 'Medium',
           '25001 to 50000': 'Large',
           '50001 to 100000': 'Large',
           'Large': 'Large',
           'Over 100000': 'Large',
           'Unknown': 'Unknown'
}

size['Category'] = size.enum.map(mapping)

size[['enum', 'x', 'Category']].groupby(
    'Category').sum().sort_values(
        by='x', ascending=False).plot(kind='bar')

########################################################################
# Plot patterns


def get_pattern(df):
    """ Generates the DBIR "patterns," with liberal inspiration from the
    getpatternlist.R: https://gist.github.com/jayjacobs/a145cb87551f551fc719

    Parameters
    ----------
    df: pd DataFrame with most VERIS encodings already built (verispy package).

    Returns
    -------
    pd DataFrame with the patterns.
    Does not return as part of original VERIS DF.
    """

    skimmer = df['action.physical.variety.Skimmer'] | \
        (df['action.physical.variety.Tampering'] &
         df['attribute.confidentiality.data.variety.Payment'])

    espionage = df['actor.external.motive.Espionage'] | \
        df['actor.external.variety.State-affiliated']

    pos = df['asset.assets.variety.S - POS controller'] | \
        df['asset.assets.variety.U - POS terminal']

    dos = df['action.hacking.variety.DoS']

    webapp = df['action.hacking.vector.Web application']
    webapp = webapp & ~(webapp & dos)

    misuse = df['action.Misuse']

    vfilter = skimmer | espionage | pos | dos | webapp | misuse

    mal_tmp = df['action.Malware'] & \
        ~df['action.malware.vector.Direct install']
    malware = mal_tmp & ~vfilter

    theftloss = df['action.error.variety.Loss'] | \
        df['action.physical.variety.Theft']

    vfilter = vfilter | malware | theftloss

    errors = df['action.Error'] & ~vfilter

    vfilter = vfilter | errors

    other = ~vfilter

    patterns = pd.DataFrame({
        'Point of Sale': pos,
        'Web Applications': webapp,
        'Privilege Misuse': misuse,
        'Lost and Stolen Assets': theftloss,
        'Miscellaneous Errors': errors,
        'Crimeware': malware,
        'Payment Card Skimmers': skimmer,
        'Denial of Service': dos,
        'Cyber-Espionage': espionage,
        'Everything Else': other
        })

    # reduce names to single label (first one encountered)
    patterns_copy = patterns.copy()

    for col in patterns_copy.columns:
        patterns_copy[col] = patterns_copy[col].apply(
            lambda x: col if x else '')

    patterns_copy['pattern'] = patterns_copy.apply(
        lambda x: ','.join(x), axis=1)

    def get_first(pats):
        pats = [pat for pat in pats.split(',') if len(pat) > 0]
        return pats[0]

    patterns_copy['pattern'] = patterns_copy['pattern'].apply(
        lambda x: get_first(x))

    # add 'pattern.' to the column names
    patterns.rename(columns={col: ''.join(('pattern.', col))
                             for col in patterns.columns}, inplace=True)

    patterns['pattern'] = patterns_copy['pattern']

    return patterns


# Plot the patterns
Data = get_pattern(veris_df)
pattern = Data['pattern']
pattern.value_counts().plot(kind='pie')


########################################################################
# Something new


def get_actor_df(actor_variety):
    df_actors = v.enum_summary(
        veris_df, actor_variety, by='actor')
    df_actors.drop(['n', 'freq'], axis=1, inplace=True)
    df_actors.columns = ['Actor Origin', 'Actor Type', 'Count']

    return df_actors


def remove_actor(df_actors, actor):
    df = df_actors[df_actors['Actor Origin'] == actor]
    df_actors.drop(df.index, inplace=True)

    return df_actors


# Get internal actors
df_actors_internal = get_actor_df('actor.internal.variety')

# Remove external, partner and unknown actors
df_actors_internal = remove_actor(df_actors_internal, 'actor.External')
df_actors_internal = remove_actor(df_actors_internal, 'actor.Partner')
df_actors_internal = remove_actor(df_actors_internal, 'actor.Unknown')

df_actors_internal['Actor Origin'] = 'Internal'
df_actors_internal

# Get external actors
df_actors_external = get_actor_df('actor.external.variety')

# Remove internal, partner and unknown actors
df_actors_external = remove_actor(df_actors_external, 'actor.Internal')
df_actors_external = remove_actor(df_actors_external, 'actor.Partner')
df_actors_external = remove_actor(df_actors_external, 'actor.Unknown')

df_actors_external['Actor Origin'] = 'External'
df_actors_external

df_actors_combined = pd.concat([df_actors_internal, df_actors_external])
df_actors_combined

# Further work here that require plotly chart_studio

########################################################################
# Action by Actor
df_action_actor = v.enum_summary(veris_df, 'action', by='actor')
df_action_actor.drop(['n', 'freq'], axis=1, inplace=True)
df_action_actor.columns = ['Actor Origin', 'Action Type', 'Count']
df_Unknown_3 = df_action_actor[df_action_actor[
    'Actor Origin'] == 'actor.Unknown']
df_action_actor.drop(df_Unknown_3.index, inplace=True)


map_origin = {'actor.External': 'External',
              'actor.Internal': 'Internal',
              'actor.Partner': 'Partner'}

df_action_actor['Actor Origin'] = df_action_actor[
    'Actor Origin'].map(map_origin)
df_action_actor

# Further work here that require plotly chart_studio

########################################################################
# Timeline to Discovery Heatmap

compromise_time = v.enum_summary(veris_df, 'timeline.compromise.unit')
compromise_time.drop(['n', 'freq'], axis=1, inplace=True)
compromise_time.sort_values(by='x', ascending=False)
compromise_time.insert(0, 'Timeline Event', 'Compromise')

compromise_NA = compromise_time[compromise_time['enum'] == 'NA']
compromise_Unknown = compromise_time[compromise_time['enum'] == 'Unknown']
compromise_time.drop(compromise_NA.index, inplace=True)
compromise_time.drop(compromise_Unknown.index, inplace=True)

mapping_time = {'Seconds': 1,
                'Minutes': 2,
                'Hours': 3,
                'Days': 4,
                'Weeks': 5,
                'Months': 6,
                'Years': 7,
                'Never': 8}

compromise_time['Time Unit'] = compromise_time.enum.map(mapping_time)
compromise_time.drop(columns=['enum'], inplace=True)
compromise_time.columns = ['Timeline Event', 'Count', 'Time Unit']

columns_titles = ["Timeline Event", "Time Unit", "Count"]
compromise_time = compromise_time.reindex(columns=columns_titles)

compromise_time


def get_timeline_df(x, event):

    df = v.enum_summary(veris_df, x)
    df.drop(['n', 'freq'], axis=1, inplace=True)
    df.sort_values(by='x', ascending=False)
    df.insert(0, 'Timeline Event', event)
    df_NA = df[df['enum'] == 'NA']
    df_Unknown = df[df['enum'] == 'Unknown']
    df.drop(df_NA.index, inplace=True)
    df.drop(df_Unknown.index, inplace=True)
    mapping_time = {'Seconds': 1,
                    'Minutes': 2,
                    'Hours': 3,
                    'Days': 4,
                    'Weeks': 5,
                    'Months': 6,
                    'Years': 7,
                    'Never': 8}
    df['Time Unit'] = df.enum.map(mapping_time)
    df.drop(columns=['enum'], inplace=True)
    df.columns = ['Timeline Event', 'Count', 'Time Unit']
    columns_titles = ["Timeline Event", "Time Unit", "Count"]
    df = df.reindex(columns=columns_titles)

    return df


discovery_time = get_timeline_df(
    'timeline.discovery.unit', 'Discovery')
exfiltration_time = get_timeline_df(
    'timeline.exfiltration.unit', 'Exfiltration')
containment_time = get_timeline_df(
    'timeline.containment.unit', 'Containment')

timeline_df = pd.concat([
    compromise_time, discovery_time, exfiltration_time, containment_time])
timeline_df

timeline_matrix = timeline_df.pivot('Time Unit', 'Timeline Event', 'Count')
timeline_matrix.columns
columns_matrix_titles = [
    "Compromise", "Exfiltration", "Discovery", "Containment"]
timeline_matrix = timeline_matrix.reindex(columns=columns_matrix_titles)
timeline_matrix.sort_index(ascending=False, inplace=True)
timeline_matrix

# Plot matrix
fig_heatmap = plt.figure(figsize=(10, 8))
r = sns.heatmap(timeline_matrix, cmap='BuPu',
                cbar_kws={'label': 'Count'}, linewidths=.05)
plt.xlabel('Timeline Stage', labelpad=20, fontsize=13)
plt.yticks([7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 0.5],
           ['Seconds', 'Minutes', 'Hours', 'Days',
            'Weeks', 'Months', 'Years', 'Never'],
           rotation=0)
plt.ylabel('Time from initial attack (units)', labelpad=20, fontsize=13)
r.set_title(
    'Heatmap of Timeline Events: \n Organisations'
    + ' typically discover a cyber attack long after'
    + 'the damage has been done'
)

########################################################################
# Exfiltration time

exfiltration_time = v.enum_summary(veris_df, 'timeline.exfiltration.unit')
exfiltration_time.sort_values(by='x', ascending=False)

discovery_time = v.enum_summary(veris_df, 'timeline.discovery.unit')
discovery_time.sort_values(by='x', ascending=False)

containment_time = v.enum_summary(veris_df, 'timeline.containment.unit')
containment_time.sort_values(by='x', ascending=False)

df_actors_internal = v.enum_summary(veris_df, 'actor.internal.variety', by='actor')
df_actors_internal

df_actors_partner = v.enum_summary(veris_df, 'actor')
df_actors_partner

df_actors_developers = v.enum_summary(veris_df, 'action',
                                      by='actor.internal.variety')
df_actors_developers = df_actors_developers[df_actors_developers['by'] ==
                                            'actor.internal.variety.Developer']
df_actors_developers.plot(kind='bar', x='enum', y='x',
                          legend=False, figsize=(8, 6))
plt.xticks(rotation=25)
plt.ylabel('Count')

#%%
import sumy
import nltk
import pickle
import pandas as pd
import seaborn as sns
from rouge import Rouge 
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from summarizer import Summarizer # BERT
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

#%%
def append_summaries(sentences):
    summary = ''
    for sentence in sentences:
        summary += str(sentence)

    return summary
    
# %%
df = pd.read_pickle('cnn_dataset_10k.pkl')
df.dropna(inplace=True)
df.drop(2119, inplace=True) # for some reason BERT fails to summarize this line
# df = df[df['summary'].str.split().apply(len) > 60]
df = df[df['summary'].str.split().apply(len) > 50]
# df = df.head(30)
# df = df.head(3)
df = df.head(1000)


# %%
# Summarize text.
df['summary_LexRank'] = ''
df['summary_TextRank'] = ''
df['summary_Luhn'] = '' 
df['summary_LSA'] = ''
df['summary_BERT'] = ''

lex_summarizer = LexRankSummarizer()
text_summarizer = TextRankSummarizer()
luhn_summarizer = LuhnSummarizer()
lsa_summarizer = LsaSummarizer()
bert_summarizer = Summarizer()
rouge = Rouge()

for i, r in df.iterrows():

    parser = PlaintextParser.from_string(df['text'].loc[i], Tokenizer("english"))
    sentence_amount = 2
    
    sentences = text_summarizer(parser.document, sentence_amount - 1)
    df['summary_TextRank'].loc[i] = append_summaries(sentences)

    sentences = lex_summarizer(parser.document, sentence_amount - 1) 
    df['summary_LexRank'].loc[i] = append_summaries(sentences)

    sentences = luhn_summarizer(parser.document, sentence_amount - 1) 
    df['summary_Luhn'].loc[i] = append_summaries(sentences)

    sentences = lsa_summarizer(parser.document, sentence_amount) 
    df['summary_LSA'].loc[i] = append_summaries(sentences)

    sentences = bert_summarizer(df['text'].loc[i], max_length=170)
    df['summary_BERT'].loc[i] = ''.join(sentences)

df = df[df['summary_BERT'].astype(bool)]  # drop empty summaries (happens in about 1/100 summaries)


#%%
print('LSA ground truth')
print(df['summary'].str.split().apply(len))
print(df['summary'].str.split().apply(len).mean())

print('BERT')
print(df['summary_BERT'].str.split().apply(len))
print(df['summary_BERT'].str.split().apply(len).mean())

print('LSA')
print(df['summary_LSA'].str.split().apply(len))
print(df['summary_LSA'].str.split().apply(len).mean())

# %%
sns.reset_orig
# import seaborn as sns

#%%
# Count words in every summary.
df['summary_groundTruth_wordCount'] = df['summary'].str.split().apply(len)
df['summary_LexRank_wordCount'] = df['summary_LexRank'].str.split().apply(len)
df['summary_TextRank_wordCount'] = df['summary_TextRank'].str.split().apply(len)
df['summary_Luhn_wordCount'] = df['summary_Luhn'].str.split().apply(len)
df['summary_LSA_wordCount'] = df['summary_LSA'].str.split().apply(len)
df['summary_BERT_wordCount'] = df['summary_BERT'].str.split().apply(len)

# Create boxplots of word count.
df2 = pd.DataFrame(data=df, columns = ['summary_groundTruth_wordCount','summary_Luhn_wordCount','summary_LSA_wordCount','summary_TextRank_wordCount', 'summary_LexRank_wordCount','summary_BERT_wordCount'])
df2 = df2.rename(columns={'summary_groundTruth_wordCount': 'ground truth', 'summary_Luhn_wordCount': 'Luhn','summary_LSA_wordCount': 'LSA', 'summary_TextRank_wordCount':'TextRank', 'summary_LexRank_wordCount': 'LexRank',   'summary_BERT_wordCount': 'BERT'})
fig = sns.boxplot(x="variable", y="value", data=pd.melt(df2))

plt.ylabel('word count')
plt.xlabel('summary')
# plt.savefig('word_count_1k.png')
plt.show(fig)

# df
#%%
# Calculate the various ROUGE scores per algorithm.
df['LexRank_ROUGE-1'] = ''
df['LexRank_ROUGE-2'] = ''
df['LexRank_ROUGE-l'] = ''

df['TextRank_ROUGE-1'] = ''
df['TextRank_ROUGE-2'] = ''
df['TextRank_ROUGE-l'] = ''

df['Luhn_ROUGE-1'] = ''
df['Luhn_ROUGE-2'] = ''
df['Luhn_ROUGE-l'] = ''

df['LSA_ROUGE-1'] = ''
df['LSA_ROUGE-2'] = ''
df['LSA_ROUGE-l'] = ''

df['BERT_ROUGE-1'] = ''
df['BERT_ROUGE-2'] = ''
df['BERT_ROUGE-l'] = ''

for i, r in df.iterrows():
    scores = rouge.get_scores(df['summary_LexRank'].loc[i], df['summary'].loc[i])[0]
    df['LexRank_ROUGE-1'].loc[i] = scores['rouge-1']['f']
    df['LexRank_ROUGE-2'].loc[i] = scores['rouge-2']['f']
    df['LexRank_ROUGE-l'].loc[i] = scores['rouge-l']['f']

    scores = rouge.get_scores(df['summary_TextRank'].loc[i], df['summary'].loc[i])[0]
    df['TextRank_ROUGE-1'].loc[i] = scores['rouge-1']['f']
    df['TextRank_ROUGE-2'].loc[i] = scores['rouge-2']['f']
    df['TextRank_ROUGE-l'].loc[i] = scores['rouge-l']['f']

    scores = rouge.get_scores(df['summary_Luhn'].loc[i], df['summary'].loc[i])[0]
    df['Luhn_ROUGE-1'].loc[i] = scores['rouge-1']['f']
    df['Luhn_ROUGE-2'].loc[i] = scores['rouge-2']['f']
    df['Luhn_ROUGE-l'].loc[i] = scores['rouge-l']['f']

    scores = rouge.get_scores(df['summary_LSA'].loc[i], df['summary'].loc[i])[0]
    df['LSA_ROUGE-1'].loc[i] = scores['rouge-1']['f']
    df['LSA_ROUGE-2'].loc[i] = scores['rouge-2']['f']
    df['LSA_ROUGE-l'].loc[i] = scores['rouge-l']['f']

    scores = rouge.get_scores(df['summary_BERT'].loc[i], df['summary'].loc[i])[0]
    df['BERT_ROUGE-1'].loc[i] = scores['rouge-1']['f']
    df['BERT_ROUGE-2'].loc[i] = scores['rouge-2']['f']
    df['BERT_ROUGE-l'].loc[i] = scores['rouge-l']['f']


#%%
# Calculate the mean of ROUGE-1, ROUGE-2, and ROUGE-L per algorithm.
LexRank_ROUGE_mean = (df['LexRank_ROUGE-1'].mean() + df['LexRank_ROUGE-2'].mean() + df['LexRank_ROUGE-l'].mean())/3
TextRank_ROUGE_mean = (df['TextRank_ROUGE-1'].mean() + df['TextRank_ROUGE-2'].mean() + df['TextRank_ROUGE-l'].mean())/3
Luhn_ROUGE_mean = (df['Luhn_ROUGE-1'].mean() + df['Luhn_ROUGE-2'].mean() + df['Luhn_ROUGE-l'].mean())/3
LSA_ROUGE_mean = (df['LSA_ROUGE-1'].mean() + df['LSA_ROUGE-2'].mean() + df['LSA_ROUGE-l'].mean())/3
BERT_ROUGE_mean = (df['BERT_ROUGE-1'].mean() + df['BERT_ROUGE-2'].mean() + df['BERT_ROUGE-l'].mean())/3

#%%
# Construct our features based on which text contain which word.
def get_words_in_text(text):
    all_words = []
    for (words, sentiment) in text:
        all_words.extend(words)
    return all_words
    
def extract_features(document):
    n_gram = 3
    ngram_vocab = nltk.ngrams(document, n_gram)
    features = dict([(ng, True) for ng in ngram_vocab])
    return features

#%%
# Calculate the conversationlaity per algorithm
f = open('ConversationalClassifier.pickle', 'rb')
cc = pickle.load(f) 
f.close()

df['cc_prob_LexRank'] = ''
df['cc_prob_TextRank'] = ''
df['cc_prob_Luhn'] = ''
df['cc_prob_LSA'] = ''
df['cc_prob_BERT'] = ''

for i, r in df.iterrows():
    df['cc_prob_LexRank'].loc[i] = cc.prob_classify(extract_features(df['summary_LexRank'].loc[i].split())).prob(1)
    df['cc_prob_TextRank'].loc[i] = cc.prob_classify(extract_features(df['summary_TextRank'].loc[i].split())).prob(1)
    df['cc_prob_Luhn'].loc[i] = cc.prob_classify(extract_features(df['summary_Luhn'].loc[i].split())).prob(1)
    df['cc_prob_LSA'].loc[i] = cc.prob_classify(extract_features(df['summary_LSA'].loc[i].split())).prob(1)
    df['cc_prob_BERT'].loc[i] = cc.prob_classify(extract_features(df['summary_BERT'].loc[i].split())).prob(1)

#%%
print('cc_prob_LexRank: ', df['cc_prob_LexRank'].mean())
print('cc_prob_TextRank: ', df['cc_prob_TextRank'].mean())
print('cc_prob_Luhn: ', df['cc_prob_Luhn'].mean())
print('cc_prob_LSA: ', df['cc_prob_LSA'].mean())
print('cc_prob_BERT: ', df['cc_prob_BERT'].mean())

#%% 
# plot a graph of summarization algorithm performance
df_results = pd.DataFrame({'cc_prob': [df['cc_prob_LexRank'].mean(), df['cc_prob_TextRank'].mean(), df['cc_prob_Luhn'].mean(), df['cc_prob_LSA'].mean(), df['cc_prob_BERT'].mean()],
                           'ROUGE_mean':[LexRank_ROUGE_mean, LexRank_ROUGE_mean, Luhn_ROUGE_mean, LSA_ROUGE_mean, BERT_ROUGE_mean], 
                           'algorithm': ['LexRank', 'TextRank', 'Luhn', 'LSA', 'BERT']
                            })

ax = sns.lmplot('cc_prob', # Horizontal axis
           'ROUGE_mean', # Vertical axis
           data=df_results, # Data source
           fit_reg=False, # No regression line
           size = 5,
           aspect = 2) # size and dimension

# plt.title('Summarization algorithms')
# Set x-axis label
plt.xlabel('Conversationality')
# Set y-axis label
plt.ylabel('ROUGE')

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.0005, point['y'], str(point['val']))

label_point(df_results.cc_prob, df_results.ROUGE_mean, df_results.algorithm, plt.gca()) 
# plt.savefig('results_1k.png')

#%%
df
# df.to_csv('results_1000.csv')


# %%
# df = pd.read_pickle('cnn_dataset_10k.pkl')

# %%
# df = pd.read_csv('results_1k.csv', index_col=0)

# %%
df['Luhn_ROUGE_mean'] = ''
df['BERT_ROUGE_mean'] = ''
df['diff'] = ''

for i, r in df.iterrows():
    df['Luhn_ROUGE_mean'].loc[i] = (df['Luhn_ROUGE-1'].loc[i] + df['Luhn_ROUGE-2'].loc[i] + df['Luhn_ROUGE-l'].loc[i])/3
    df['BERT_ROUGE_mean'].loc[i] = (df['BERT_ROUGE-1'].loc[i] + df['BERT_ROUGE-2'].loc[i] + df['BERT_ROUGE-l'].loc[i])/3

for i, r in df.iterrows():
    df['diff'].loc[i] = abs(df['Luhn_ROUGE_mean'].loc[i] - df['BERT_ROUGE_mean'].loc[i])

diff = df.groupby('diff', sort = True).max()
diff.to_csv('diff.csv')
# %%

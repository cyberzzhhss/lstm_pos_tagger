from __future__ import division
from collections import defaultdict
import math
import sys
import string

UNK_WORDS = [
    'unk',
    'unk_noun',
    'unk_verb',
    'unk_digit',
    'unk_adv',
    'unk_cap_noun',
    'unk_punct',
    'unk_adj',
    ]
ADJ_SUFFIX = [
    'zy',
    'dy',
    'ese',
    'less',
    'ish',
    'ian',
    'able',
    'esque',
    'i',
    'lly',
    'ible',
    'ous',
    'ly',
    'ive',
    'sy',
    'ful',
    'al',
    'ic',
    'ical',
    ]
VERB_SUFFIX = [
    'ify',
    'fy', 
    'ate', 
    'ize', 
    'ise'
    ]
ADV_SUFFIX = [
    'ally',
    'wards',
    'wise',
    'ily',
    'ly',
    'ward',
    ]
NOUN_SUFFIX = [
    'hood',
    'ness',
    'cy',
    'ery',
    'ation',
    'ism',
    'ance',
    'or',
    'scape',
    'ure',
    'ship',
    'ist',
    'er',
    'dom',
    'ry',
    'action',
    'acy',
    'ion',
    'al',
    'age',
    'ity',
    'ty',
    'ee',
    'ence',
    'ment',
    'ling',
    ]

PUNCT = set(string.punctuation)


def assign_unk(word):
    if any(word.endswith(s) for s in ADJ_SUFFIX):
        return 'unk_adj'
    if any(word.endswith(s) for s in VERB_SUFFIX):
        return 'unk_verb'
    if any(word.endswith(s) for s in ADV_SUFFIX):
        return 'unk_adv'
    if any(word.endswith(s) for s in NOUN_SUFFIX):
        return 'unk_noun'
    if any(char in PUNCT for char in word):
        return 'unk_punct'
    if any(char.isupper() for char in word) or ('-' in word and word[0].isupper()):
        return 'unk_cap_noun'
    if any(char.isdigit() for char in word):
        return 'unk_digit'
    return 'unk'


word_count = defaultdict(int)
tag_count = defaultdict(int)

transition = defaultdict(lambda : defaultdict(int))
emission = defaultdict(lambda : defaultdict(int))

with open(sys.argv[1], 'r') as train_data:
    for line in train_data:
        if line.split():
            (word, tag) = line.split()
            word_count[word] += 1

with open(sys.argv[1], 'r') as train_data:

    vocab = [key for (key, value) in word_count.items() if value > 1]
    vocab.extend(UNK_WORDS)
    vocab.append('<n>')
    vocab = sorted(vocab)
    vocab_list = vocab
    vocab_set = set(vocab_list)

    prev = '<s>'
    for line in train_data:
        if not line.split():
            word = '<n>'
            tag = '<s>'
        else:
            (word, tag) = line.split()
            word_count[word] += 1
            if word not in vocab_set:
                word = assign_unk(word)
        tag_count[tag] += 1
        transition[prev][tag] += 1
        emission[tag][word] += 1
        prev = tag

tags = sorted(tag_count.keys())
num_tags = len(tags)

ALPHA = 0.001

trans_p = [num_tags * [0] for i in range(num_tags)]
for i in range(num_tags):
    for j in range(num_tags):
        prev = tags[i]
        tag = tags[j]
        count = 0
        if prev in transition and tag in transition[prev]:
            count = transition[prev][tag]
        trans_p[i][j] = (count + ALPHA) / (num_tags * ALPHA + tag_count[prev])

vocab_length = len(vocab_list)
emi_p = [vocab_length * [0] for i in range(num_tags)]

for i in range(num_tags):
    for j in range(vocab_length):
        tag = tags[i]
        word = vocab_list[j]
        count = 0
        if word in emission[tag]:
            count = emission[tag][word]
        emi_p[i][j] = (count + ALPHA) / (vocab_length * ALPHA + tag_count[tag])

train_data.close()

seq = []
processed_seq = []
with open(sys.argv[2], 'r') as test_data:
    for line in test_data:
        word = line.strip()
        seq.append(word)
        if not line.split():
            processed_seq.append('<n>')
        else:
            if word not in vocab_set:
                word = assign_unk(word)
            processed_seq.append(word)
test_data.close()

(obs_space, states, obs, trans_p, emi_p) = (vocab_list, tags, processed_seq, trans_p, emi_p)

states_len = len(states)
obs_i = {}

for (i, observation) in enumerate(obs_space):
    obs_i[observation] = i

start_tag_i = states.index('<s>')
first_obs_i = obs_i[obs[0]]
seq_len = len(obs)
viterbi = [[0] * seq_len for i in range(states_len)]
backptr = [[None] * seq_len for i in range(states_len)]


for s in range(states_len):
    backptr[s][0] = 0
    if trans_p[start_tag_i][s] == 0:
        viterbi[s][0] = -sys.maxsize
    else:
        viterbi[s][0] = math.log(emi_p[s][first_obs_i]) + math.log(trans_p[start_tag_i][s])

for t in range(1, seq_len):
    obs_t_i = obs_i[obs[t]]
    for cur_s in range(states_len):
        (max_prob, max_path) = (-sys.maxsize, None)
        for prev_s in range(states_len):
            prob = math.log(emi_p[cur_s][obs_t_i]) + math.log(trans_p[prev_s][cur_s]) + viterbi[prev_s][t - 1]
            if prob > max_prob:
                (max_path, max_prob) = (prev_s, prob)
            backptr[cur_s][t] = max_path
            viterbi[cur_s][t] = max_prob

max_prob = viterbi[0][seq_len - 1]
state_i_seq = seq_len * [None]
result = seq_len * [None] 

for state_i in range(1, states_len):
    if viterbi[state_i][seq_len - 1] > max_prob:
        max_prob = viterbi[state_i][seq_len - 1]
    state_i_seq[seq_len - 1] = state_i

result[seq_len - 1] = states[state_i_seq[seq_len - 1]]
for t in range(seq_len - 1, 0, -1):
    state_i_seq[t - 1] = backptr[state_i_seq[t]][t]
    result[t - 1] = states[state_i_seq[t - 1]]

name = sys.argv[2] + "_hmm_output"
with open(name, 'w') as out:
    for (word, tag) in zip(seq, result):
        if not word:
            out.write('\n')
        else:
            out.write('{0}\t{1}\n'.format(word, tag))
out.close()

import sys
import torch
from build_lstm_model import pos_tagger, word_encoder_fn, char_encoder_fn
import numpy as np
device = torch.device("cpu")

def tag_sentence(test_file, model_file, out_file):
    WORD_EMBEDDING_DIM = 100
    CHAR_EMBEDDING_DIM = 50
    CHAR_CNN_FILTER_SIZE = 3
    HIDDEN_DIM = 128
    N_LAYERS = 2
    DROPOUT = 0
    word_dict, tag_dict, char_dict, model_state_dict = torch.load(model_file)
    ix_to_tag = {}
    for tag in tag_dict:
        ix_to_tag[tag_dict[tag]] = tag

    CHAR_INPUT_DIM = len(char_dict) + 1
    CHAR_PAD_IDX = len(char_dict)
    WORD_INPUT_DIM = len(word_dict) + 1
    WORD_PAD_IDX = len(word_dict)
    OUTPUT_DIM = len(tag_dict) + 1
    word_encoder = word_encoder_fn(vocab_size=WORD_INPUT_DIM, embedding_dim=WORD_EMBEDDING_DIM, pad_idx=WORD_PAD_IDX)
    char_encoder = char_encoder_fn(vocab_size=CHAR_INPUT_DIM, embedding_dim=CHAR_EMBEDDING_DIM,
                               filter_size=CHAR_CNN_FILTER_SIZE, hidden_dim=WORD_EMBEDDING_DIM, pad_idx=CHAR_PAD_IDX)
    model = pos_tagger(char_encoder=char_encoder, word_encoder=word_encoder, output_dim=OUTPUT_DIM,
                      hidden_dim=HIDDEN_DIM,
                      n_layers=N_LAYERS, dropout=DROPOUT)
    model.load_state_dict(model_state_dict)
    model.to(device)

    f = open(test_file,'r')
    f2 = open(out_file,'w')
    line = f.readline().strip('\n')
    untagged_words = []
    while line:
        if line.strip('\n') == '':
            max_word_len = max([len(word) for word in untagged_words])
            words = [word_dict[word] if word in word_dict else word_dict['<unk>'] for word in untagged_words]
            chars = []
            for word in untagged_words:
                # chars_in_word = [char_dict[char] for char in word]
                chars_in_word = []
                for char in word:
                    if char not in char_dict:
                        chars_in_word.append(0)
                    else:
                        chars_in_word.append(char_dict[char])
                padding_len = max_word_len - len(chars_in_word)
                for _ in range(padding_len):
                    chars_in_word.append(CHAR_PAD_IDX)
                chars.append(chars_in_word)
            words_tensor = torch.tensor(words, dtype=torch.long).view(-1, 1).to(device)
            chars_tensor = torch.tensor(chars, dtype=torch.long).view(1, -1, max_word_len).to(device)
            with torch.no_grad():
                predictions = model(words_tensor, chars_tensor)
                predictions = predictions.view(-1, predictions.shape[-1]).cpu()
            assigned_tag_ix = np.argmax(predictions, axis=1).numpy()   
            tagged_words = ['%s\t%s\n' % (untagged_words[i], ix_to_tag[assigned_tag_ix[i]]) for i in range(len(assigned_tag_ix))]
            for tagged_word in tagged_words:
                f2.write(tagged_word)
            f2.write('\n')
            line = f.readline()
            untagged_words = []
            continue
        line = line.strip('\n')
        untagged_words.append(line)
        line = f.readline()
    f.close()
    f2.close()
    print("Finished tagging.")

if __name__ == "__main__":
    model_file = sys.argv[1]
    test_file = sys.argv[2]
    out_file = sys.argv[2]+'_lstm_output'
    tag_sentence(test_file, model_file, out_file)
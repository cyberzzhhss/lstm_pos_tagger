
HMM

	Run bigram HMM tagger:

		python3 hmm_tagger.py <train_data> <test_data>

		It will output test_data_hmm_output


LSTM

	Dependencies:
	
		numpy, pytorch

	Run LSTM tagger:

		(1) build model:
			
			python3 build_lstm_model.py <train_data>

		It will output train_data_model

		(2) use model to tag:
			
			python3 lstm_tagger.py <train_data_model> <test_data>

		It will output test_data_lstm_output

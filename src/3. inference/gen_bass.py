import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for GPU inference
from glob import glob
import pickle as pickle
from gen_utils import bass_trans_ev_model_tf, generate_bass_ev_trans_tf, create_onehot_enc
import numpy as np

thr_measures = 16
thr_max_tokens = 800
thr_min_tokens = 50
dec_seq_length = 797


'''load Encoders pickle for onehotencoders'''

#encoders pickle is created during pre-processing
encoders_trans = './aux_files/bass_encoders_cp.pickle'

    
with open(encoders_trans, 'rb') as handle:
    TransEncoders = pickle.load(handle)
#[Encoder_RG, Decoder_Bass]

'''Load Inference Transformer. You may download pre-trained model based 
on the paper. See instructions in ReadME.md'''
trans_bass_hb = bass_trans_ev_model_tf(TransEncoders, dec_seq_length)


'''Set Temperature'''
temperature = 0.9

'''Load MIDI files with Guitar (1st) and Bass (2nd). See examples in midi_in folder'''
'''max 16 bars'''
#input folder (put txt token files of rg only here)
inp_path = glob('./tokens_in/*.txt')
#output folder
out_path = './tokens_out/'

# Inference on the test set

test_path = r"..\..\data\processed\test_set_streams_16_8_800_50.pickle"

with open(test_path, 'rb') as handle:
    testSet = pickle.load(handle)

enc_input_test = np.int64(np.stack(testSet['Encoder_Input'])) #encoder input
output_test = []

for Enc_Input in enc_input_test:
    # call generation functions
    bass_HB = generate_bass_ev_trans_tf(trans_bass_hb, TransEncoders, temperature, Enc_Input, dec_seq_length=dec_seq_length)      
    # save token files to be passed to the tokens2gp5 algorithm
    output_test.append(bass_HB)
    
# output_test is a list of lists of tokens for the 11 817 test set sequences
            
  




 
    


    

    
 
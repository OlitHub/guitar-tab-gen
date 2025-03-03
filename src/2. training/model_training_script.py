import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import pickle
import numpy as np
import time
from aux_train_tf import HybridTransformer, create_masks

physical_devices = tf.config.list_physical_devices('GPU')

train_path = "train_set_streams.pickle"
test_path = "test_set_streams.pickle"

with open(train_path, 'rb') as handle:
    trainSet = pickle.load(handle)

with open(test_path, 'rb') as handle:
    testSet = pickle.load(handle)
    
#train 
enc_input_train = np.int64(np.stack(trainSet['Encoder_Input'])) #encoder input
dec_input_train = np.int64(np.stack(trainSet['Decoder_Input'])) #decoder onset stream
dec_output_train = np.int64(np.stack(trainSet['Decoder_Output']))
#validation
enc_input_val = np.int64(np.stack(testSet['Encoder_Input']))
dec_input_val = np.int64(np.stack(testSet['Decoder_Input']))
dec_output_val = np.int64(np.stack(testSet['Decoder_Output']))


#prepare datasets
BUFFER_SIZE = len(enc_input_train)
BUFFER_SIZE_EVAL = len(enc_input_val)
BATCH_SIZE = 32 #set batch size
steps_per_epoch = BUFFER_SIZE//BATCH_SIZE
steps_per_epoch_eval = BUFFER_SIZE_EVAL//BATCH_SIZE

#create training and evaluation tf dataset
dataset = tf.data.Dataset.from_tensor_slices((enc_input_train, 
                                              dec_input_train, dec_output_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

dataset_eval = tf.data.Dataset.from_tensor_slices((enc_input_val,
                                                   dec_input_val, dec_output_val)).shuffle(BUFFER_SIZE_EVAL)
dataset_eval = dataset_eval.batch(BATCH_SIZE, drop_remainder=True)


#set transformer hyper parameters
num_layers = 4  #attention layers
#Embeddings
d_model_enc = 240 #Encoder Embedding (64 + 16 + 32 + 64 + 64)

d_model_dec = 192 #Decoder Embedding (96 + 96)

units = 1024 #for Dense Layers and BLSTM Encoder
num_heads = 8 #
dropout_rate = 0.3

#vocab sizes
enc_vocab = 1754
dec_vocab = 1083

#sequence lengths
enc_seq_length = 800
dec_seq_length = 793

#for relative attention half or full window
rel_dec_seq = dec_seq_length


model = HybridTransformer(num_layers=num_layers, d_model_enc=d_model_enc,
                          d_model_dec=d_model_dec, num_heads=num_heads,
                          dff=units, input_vocab=enc_vocab+1, target_vocab=dec_vocab+1, 
                          pe_target=dec_seq_length, 
                          mode_choice='relative', #change to multihead for vanilla attention mechanism
                          max_rel_pos_tar=rel_dec_seq, rate=dropout_rate)


#Set Optimizers and Loss Function
optimizer = tf.keras.optimizers.Adam(0.0005, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction='none')


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

#Set TF Metrics

train_loss_list = []
train_accuracy_list = []
val_loss_list = []
val_accuracy_list = []


#Set Checkpoints
checkpoint_path = './checkpoints/'

ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Latest checkpoint restored!!')

# Set input signatures

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64)
]

val_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64)
]


'''Training and Validation functions'''
@tf.function(input_signature=train_step_signature)
def train_step(inp, tar_inp, tar_real):

  _, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

  with tf.GradientTape() as tape:
    preds, _ = model(	inp,
						tar_inp,
						True,
						combined_mask,
						dec_padding_mask)
    
    loss = loss_function(tar_real, preds)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
  acc = accuracy_function(tar_real, preds)

  train_loss_list.append(loss.numpy())
  train_accuracy_list.append(acc.numpy())

  
  
  
@tf.function(input_signature=val_step_signature)
def val_step(inp, tar_inp, tar_real):


  _, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

  preds, _ = model(	inp,
					tar_inp,
					False, #change?
					combined_mask,
					dec_padding_mask)
  
  loss = loss_function(tar_real, preds)
  
  acc = accuracy_function(tar_real, preds)

  val_loss_list.append(loss.numpy())
  val_accuracy_list.append(acc.numpy())

  
"""START TRAINING"""
epochs = 2
patience = 0
curr_loss = 99.99    
for epoch in range(epochs):
  start = time.time()

  train_loss_list = []
  train_accuracy_list = []

  
  print(f'Epoch {epoch + 1}')
  print('----')
  for (batch, (inp, tar_inp, tar_real)) in enumerate(dataset.take(steps_per_epoch)):
    train_step(inp, tar_inp, tar_real)


    if batch % 50 == 0:
      print(f'Batch {batch}')
      print(f'Onset Loss {np.mean(train_loss_list):.4f} -- Onset Accuracy {np.mean(train_accuracy_list):.4f}')


  print('----')
  print(f'Onset Loss {np.mean(train_loss_list):.4f} -- Onset Accuracy {np.mean(train_accuracy_list):.4f}')

  
  
  print('Evaluating...')

  val_loss_list = []
  val_accuracy_list = []

  
  for (batch, (inp, tar_inp, tar_real)) in enumerate(dataset_eval.take(steps_per_epoch_eval)):
    val_step(inp, tar_inp, tar_real)
  
  print('----')
  print(f'Validation Onset Loss {np.mean(val_loss_list):.4f} -- Onset Accuracy {np.mean(val_accuracy_list):.4f}')
  
  
  val_loss = np.round(val_loss.result().numpy(), decimals = 5) #change weights
  print('Overall weighted Validation Loss: ', val_loss)
  
  '''EARLY STOP MECHANISM'''
  if curr_loss > val_loss:
    #save checkpoint
    print('Checkpoint saved.')
    patience = 0
    save_path = ckpt_manager.save()
    curr_loss = val_loss
    
  else:
      print('No validation loss improvement.')
      patience += 1
      
  print(f'Time taken for this epoch: {time.time() - start:.2f} secs\n')    
  print('*******************************')
      
  if patience > 5:
      print('Terminating the training.')
      break


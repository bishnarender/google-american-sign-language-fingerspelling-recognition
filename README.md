## google-american-sign-language-fingerspelling-recognition
## score at 1st position is achieved.
![asl-submission](https://github.com/bishnarender/google-american-sign-language-fingerspelling-recognition/assets/49610834/f6d237fd-96d6-4b46-8d8f-ad34d4be7237)

### Start 
-----
For better understanding of project, read the files in the following order:
1. preprocessing_1.ipynb
2. preprocessing_2_supplement.ipynb
3. all_in_one.ipynb
4. asl-submission-2023-09-07.ipynb

For each video frame (present in parquet file), there are rows of frames associated with a sequence_id/phrase. These rows have x,y and z values for each of the 543 landmarks/keypoints. Out of these we have chosen “Face, Pose, Right hand and Left hand” landmarks in a total 130 landmarks. So, each parquet file is resized to 3-dimension i.e., (frames, columns_per_frame/landmarks, 3) or (384,130,3). The 3 present in the last dimension have the value of x,y, and z for respective column/landmark. We have chosen 384 sequences for a particular pharse/sign/sequence_id i.e., 384 sequences of expressions of landmarks/keypoints generates a phrase.

Data is splitted into 4 folds by signer/participant_id. 

Mask differentiates original and padded sequences.

The encoder is a significantly improved version of Squeezeformer, where the feature extraction was adapted to handle mediapipe landmarks instead of speech signals.

Prior to data augmentations, the data was normalized with std/mean and nans were zero filled.

### Part I
-----
![asl](https://github.com/bishnarender/google-american-sign-language-fingerspelling-recognition/assets/49610834/def5c7be-d56e-497d-ac87-2d9711d1e8f8)

![asl_1](https://github.com/bishnarender/google-american-sign-language-fingerspelling-recognition/assets/49610834/94e4b105-a366-448f-a752-3eeff5c9475f)

![asl_2](https://github.com/bishnarender/google-american-sign-language-fingerspelling-recognition/assets/49610834/45277807-da6e-404b-b3cf-4245814a2fd1)

384 is the length of our input sequence. 144 is the embedding associated with each element of the input sequence. The same embedding is also used to create a "query, key and value" vector for an element in the input sequence. Since the "vector" length of the "query, key and value" vector is somewhat smaller than the embedding size (144). So, we have chosen this as 36. This created 4 self-attention heads.

Score vector of each element (in input sequence) is computed as [1,36]@[36,384] which gives [1,384]. Where [1,36] represents query vector of target element,
[384,36] represents key vectors of elements and [1,384] represents the score vector of the target element. The scores are divided by the square root of dimension/length (i.e., 36) of key vectors. Results are passed through softmax operation. 

Normalized score vector of each element (in input sequence) is then multiplied as [1,384]@[384,36] which gives [1,36]. Where [1,384] represents the score vector of the target element, [384,36] represents value vectors of elements and [1,36] represents the final result of the target element. Here, the theme is to dot product normalized score vector with the respective "index vector" one by one to get the resultant vector. Further, the first index vector is obtained by picking the first index values from each value vector.

Absolute position embeddings (e.g., sine and cosine positional embeddings) provide the model with information about the order of tokens in a sequence, but they do not consider the relationships between tokens or their relative distances. Absolute position embeddings do not directly capture relative positions to each word.
Instead, they provide a static reference to each token's position in the sequence. The problem with this approach is that it limits the maximum length of the sequence that the model can process, and it also ignores the relative distances between tokens. With additive positional embedding at input, the attention/score matrices have much lower rank, limiting the representative power. 

But using per-head position embeddings and then adding position information to "score/attention matrix" directly results in allowing for higher rank attention. Reference paper: "A Simple and Effective Positional Encoding for Transformers" by Pu-Chin Chen, Henry Tsai and Srinadh Bhojanapalli.

<b>Positional embedding is basically a learned/learnable positional encoding.</b>

Absolute position embeddings are computed as:<br>
<b>PE_positive(position, 2i) = sin(position * e^(-2i * log(10000) / d_model))</b><br>
<b>PE_positive(position, 2i+1) = cos(position * e^(-2i * log(10000) / d_model))</b><br>
<b>PE_negative(position, 2i) = sin(-1 * position * e^(-2i * log(10000) / d_model))</b><br>
<b>PE_negative(position, 2i+1) = cos(-1 * position * e^(-2i * log(10000) / d_model))</b><br>
Both positive and negative absolute embeddings are computed for each position in order to represent a "rotation matrix" for each position.
https://www.youtube.com/watch?v=C6rV8BsrrCc

​These "absolute position embeddings" are transformed to per-head position embeddings such that the embedding size is the same as the embedding size of query vector of head.

<b>Rotary position embedding:</b> rotate the affine-transformed word embedding vector by amount of angle multiples of its position index. Affine-transformed word embedding vector is a new vector through a linear transformation followed by an addition of a bias vector.
![rotation_matrix](https://github.com/bishnarender/google-american-sign-language-fingerspelling-recognition/assets/49610834/1d3e47a8-7269-4547-a0b5-c8ba8f43e55e)

The positions associated with each head are 36 and size of embedding to these positions depends upon sequence length. ​The per-head position embeddings are transformed to "rotary position embeddings" of elements after having dot product with query vectors of elements i.e., [x1,x2,x3,x4]@[sinθ1,cosθ1,sinθ2,cosθ2] to get [x1sinθ1 + x2cosθ1+ x3sinθ2 + x4cosθ2]. Where [x1,x2,x3,x4] is a "query vector" for element x, ​[sin,cos,sin,cos] is one of all position embeddings of a head and [x1sinθ1 + x2cosθ1+ x3sinθ2 + x4cosθ2] is the first index value of the rotary position vector of x. Similarly after having other index values for the "rotary position vector" of x, the vector becomes [..., -x1sinθ1 + x2cosθ1 - x3sinθ2 + x4cosθ2, ...]. Thus, per-head position embeddings rotate a "query vector" in its embedding space. Finally, we have transformed one-word position from "query vector space" to "embedding space of head positions". Hence, a neural network can learn to understand relative word positions. If a model can identify the relative positions of words by rotations, it should be able to detect any relative positions. 
![relative](https://github.com/bishnarender/google-american-sign-language-fingerspelling-recognition/assets/49610834/e464b6e3-b354-42a9-85fa-a258732824ad)

Rotary Position Embedding/Encoding (RoPE) comes with flexibility of being expand to any sequence lengths. Rotary embeddings speed up training ~2X and tf-lite inference approx ~3X allowing larger models to be used.

u_bias_param is the bias parameter and has the bias to each self-attention head. Each head bias has embedding size 36. ​Further the same head bias is added to all "query vectors" associated with each head. These "u biased" "query vectors" are injected to dot product with key vectors.

v_bias_param is the bias parameter and has the bias to each self-attention head. Each head bias has embedding size 36. ​Further the same head bias is added to all "query vectors" associated with each head. These "v biased" "query vectors" are injected to dot product with position vectors.

The reason for the too negative score for "padded positions" is because once you take the softmax of the scores, the too negative values get zeroed out. This essentially tells the model to put no focus on "padded positions".

### Part II
-----
![1_asl](https://github.com/bishnarender/google-american-sign-language-fingerspelling-recognition/assets/49610834/151f08c5-756b-436d-90f7-210067ac6285)

![1_asl_1](https://github.com/bishnarender/google-american-sign-language-fingerspelling-recognition/assets/49610834/1081ae47-7fca-429f-8b5c-258d09457b44)
Final loss which is used for gradient calculation is calculated as:
<code>
loss = (1- 0.02) * ((1-0.4)*loss_1 + 0.4 * loss_2) + 0.02 * loss_3
</code>

In this part, again "rotary position embeddings" are created but in a different way using cached sin and cos ("transformers/models/llama/modeling_llama.py").

Rotary embeddings ​are cached once and fe​d ​into each layer/module with the input data so they are not duplicated in each layer. This resulted in 20% less parameters in the model.

Further, this part results in "relative position embeddings" after having dot-product between query and key vectors. Because, here we have "rotary position embeddings" for key vectors also in addition to query vectors.

Additionally to the output of the encoder, a single linear layer is added to take the features of the first time-step (of the sequence) to predict a "confidence score", which helps to identify garbage data and can be used in post-processing. As a target/label for this we used normalized levensthein distance clipped to [0,1] of OOF predictions of a best model (from the first part). Further here in this part, val_score_pp represents levensthein distance after post-process (i.e., with dummy phrases to positions having "confidence score" less than 0.15).

A "causal decoder" mainly uses "encoder cross attentions" for the sequence's beginning and "previous characters and cross attentions" for the end. To improve the model's accuracy, we use a separate causal decoder on the reversed sequence as an auxiliary loss, making the model rely more on the "encoder cross attention" for the label's end (i.e., in reverse order the previous end tokens become the first). 

#### augmentation
-----
In addition to certain universal augmentation methods, some special methods are performed which I am discussing here.

50% of time reverse the value of x over time-axis and then flip the order of keypoints/landmarks over time-axis.
<code>
#- data.shape => torch.Size([216, 130, 3])
data[:,:,0] = -data[:,:,0]
data = data[:,flip_array]
</code>

50% of time outer_cutmix: concat randomly cutted portion of two sequences of same participant.
<code>
new_phrase = phrase[cut_off_phrase:] + phrase2[:cut_off_phrase2]
#- data.shape => torch.Size([156, 130, 3])
new_data = torch.cat([data[cut_off_data:], data2[:cut_off_data2]])
</code>

75% of time mask finger landmarks i.e., assign 0.0 value to "x,y and z" of finger landmarks.
<code>
#- class FingersDrop
hand_indices = [[t for t,l in enumerate(landmarks) if i in l] for i in 'x_left_ x_right_'.split()]
hand_indices = np.array(hand_indices)
#- hand_indices.shape => (2, 21)
#- n_fingers => (2, 6)       

#- reshaping after picking finger indices from hand.
self.finger_indices = np.reshape(hand_indices[:,1:], (-1, 4))
#- self.finger_indices.shape => (10, 4)

drop_indices = self.finger_indices[fidx].flatten()
#- drop_indices.shape => (48,)
#- self.mask_value => 0.0
x_new[:, drop_indices] =  torch.tensor(self.mask_value)
</code>

#### Final
-----
The final result is an ensemble of 2 models (trained over different seeds) from part 2.



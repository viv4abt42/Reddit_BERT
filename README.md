# Similarity calculation using Reddit user profile data

## Project goals
Calculate a similarity score between two Reddit users by analyzing their posts and comments.
The similarity score of different pairs of users can be compared and reflects how they are similar.

## Project input parameters
- client_id, client_secret, user_agent. A Reddit app credentials. See: 
    - [Where create app](https://www.reddit.com/prefs/apps)
    - [How to use these credentials](https://praw.readthedocs.io/en/stable/getting_started/quick_start.html#read-only-reddit-instances)
- text_fetch_depth. How many new comments and post will be fetched. This parameter influences RAM and time consumption
- max_tokenizer_length. This parameter influences RAM and time consumption. See:
    - [Encode_plus method parameters](https://huggingface.co/docs/transformers/v4.20.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode_plus)

## Project classes
- RedditUser: class for fetching user texts
- Reddit: class for working with Reddit. Delegates fetching to RedditUser
- TextToVec: a class for converting a text to a vector
- Application: a class for starting the project. Also contains some utility methods.


## Project workflow
# Building user vector representation
- For each user:
    - Downloads last `text_fetch_depth` comments and posts of the users using [`PRAW`](https://praw.readthedocs.io/en/stable/)
python library
    - Transforms these texts to vectors using [BERT Tokenizer and BERT pre-trained encoder](https://huggingface.co/bert-base-uncased)
    - Averaging user text vectors to build a vector characterizing this user
    
![](pics/Schema%20of%20code.png)

# Overall workflow

- Project asks a user to enter two user names
- Using described above algorithm we get two user representation vectors
- Calculates [cosine similarity score](https://en.wikipedia.org/wiki/Cosine_similarity)

![](pics/Overall%20schema.png)

## How text-to-vector conversion works
This project uses a pre-trained BERT model. BERT consists of two parts: encoder and decoder.
It was trained on [masked language modeling and next sentence prediction tasks](https://huggingface.co/bert-base-uncased).
The encoder has hidden output states. 
We can say that averaging these vectors represents a vector representation of a whole text. 
So this representation can be used as a feature vector of this text.

![](pics/berts_encoder_as_converter.png)
See: [Measuring Text Similarity Using BERT](https://www.analyticsvidhya.com/blog/2021/05/measuring-text-similarity-using-bert/)
## Results

# Using [pooler_output](https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions)
Here was used already existing in BERT some kind of aggregation of all hidden states of all input tokens: BertPooler.
Results are strange. Often the similarity score is in a range [0.9, 1.0]. 
It's weird and needs more time to investigate.

# Using [last_hidden_state](https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions)
The result is better. The similarity score is in the range [0.5, 1.0]. Here we used an averaging of all hidden states of BERT encoder.
Each hidden state represents an inner representation of a token in an input sequence of words.

# Could be tried old good [Word2vec](https://en.wikipedia.org/wiki/Word2vec) as a word encoder
We could map each word in a text to a vector using Word2vec. Then all these vectors could be aggregated by averaging.
This method contains an obvious flaw: it doesn't consider a text as a sequence of words with inner structure.
But it will be fast and still might output a meaningful result.

## What to improve
- Fine-tuning of BERT model using Reddit texts
- Preprocessing of Reddit texts (delete links, smiley faces etc.)

## Links
1. [Original article](https://arxiv.org/abs/1810.04805)
2. [Some youtube video about BERT](https://www.youtube.com/watch?v=iDulhoQ2pro&ab_channel=YannicKilcher)
3. [A series of articles explaining BERT](https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452)


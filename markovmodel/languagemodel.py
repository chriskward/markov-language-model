import numpy as np
import pandas as pd

class MarkovModel():
	""" A character based Markov Language Model

	MarkovModel(text,n=5,alpha=0.001)

	text: training text str()
	n: character n-gram length
	alpha: smoothing parameter

	methods:
	
	.generate( int:no_of_chars ) -> str
	generates a sequence of characters using the models probability mass function

	.predict( str: text_sample ) -> float
	retunrs the log probability of a text_sample


	internals:
	
	The model estimates a probability distribution over the set
	of all characters occuring in the training text, given the
	preceeding n characters.

	e.g P(Xn+1 | Xn , Xn-1, Xn-2 , ...)

	e.g P( O | HELL )  for n=4
	"""



	def __init__(self,data,n=5,alpha=0.001):
		"""
		model = MarkovModel(text,n=5,alpha=0.001)

		text: training text str()
		n: character n-gram length
		alpha: smoothing parameter
		"""

		self.alpha = alpha
		self.n = n

		# n is the context (sliding) window size, e.g n=3 text='Hello' --> 'Hel' 'ell' 'llo'
		# alpha is the probability mass function smoothing parameter

		if not isinstance(data,str) : raise TypeError('Training data must be provided as type: str')

		context,next_char = self._stride_split(data, self.n)

		# split the text string into an array of context strings (character n-grams) and an array with
		# next characters corresponding to each character n-gram
		# eg n=2 text='Hello'
		#
		# context = [Hel' 'ell' 'llo']
		# next_char = ['l', 'o']
		#
		# In the text sample, the ngram in context[0] is
		# followed by the character in next_char[0]
		#
		# note: we need to discard the final entry of context.
		# eg, context[-1] = 'llo' as this  ngram represents
		# the end of our training sample and is not followed by another letter

		pmf = pd.DataFrame( index=np.unique(context), columns=np.unique(next_char), data=self.alpha )
			
		# dataframe with row for each unique context string occuring in training sample
		# and column for each unique next_char letter occuring in training sample
		#
		# pmf is populated with 0<alpha<1 as an initial count.

		self._pmf = self._estimate_probs(pmf,context,next_char)

		# estimate probabilities
		# 
		# self._pmf is an estimated conditional probability mass function over
		# all characters that appear in the training text.
		#
		# each row represents a context n-gram; eg ['Hel']
		# each column in that row represents the probability that each
		# character occurs after the n-gram ['Hel']

	def _stride_split(self, text, n):

		# to avoid the need to create character n-grams by iterating
		# through the training text string, we create a new view on
		# the ndarray containing the text by adjusting window size and stride
		# (much faster but requires numpy>=1.25)

		text = np.array( text,dtype=np.str_ )
		buffer = np.frombuffer( text, dtype='U1' )
		context = np.lib.stride_tricks.sliding_window_view( buffer, (n,)).view('U'+str(n)).flatten()
		context = context[:-1]

		# return an ndarray of ngrams (context) and an ndarray of
		# suceeding characters for each ngram.

		return context, buffer[n:]


	def _estimate_probs(self, pmf, context, next_char):

		# for each unique n-gram occuring in the training text
		# obtain the index position of each occurance in the
		# training text.

		for x in pmf.index:
			mask = np.argwhere(context==x)

		# use these index positions to index next_char
		# for each different character that comes after
		# the ngram 'x', add one to the corresponding
		# column of the dataframe in the row 'x'

			for y in next_char[mask]:
				pmf.loc[x,y] += 1

		# we now have a dataframe of counts. For each
		# ngram (row), the columns indicate the number of
		# times each corresponding succeeds this nggram
		# normalise each row to a probability distribution

		norm_constant = pmf.sum(axis=1)
		return pmf.divide(norm_constant, axis=0)


	def generate(self,length=100):
		"""
		.generate( int:no_of_chars ) -> str
		generates a sequence of characters using the models probability mass function

		This method initially selects a n-gram uniformly at random then;

		1) samples from the n-grams corresponding mass function (ie the corresponding row of self._pmf) by the inverstion method
		2) appends this character to the output string
		3) selects the trailing n characters of the output string as the next n-gram
		4) repeats from (1) till a string of the desired length has been sampled
		"""

		# randomly select a row of self._pmf
		# we are randomly choosing an n-gram to start with

		text_out = np.random.choice(self._pmf.index)

		while len(text_out) < length:

			# select the trailing n characters to be the next ngram
		
			context = text_out[-self.n:]

			# if this ngram is not in the pmf (which could
			# happen as we are randomly sampling characters)
			# then we sample a new ngram from the pmf and 
			# append this to the output string.

			if context not in self._pmf.index:
				context = np.random.choice(self._pmf.index)
				text_out += ' '+context

			# select the row of pmf corresponding to 'context'
			# this row the conditional pmf over the character set
			# given 'context'

			# cumulatively sum this row, now we have the
			# cumulative distribution function

			cdf = np.cumsum( self._pmf.loc[context] )

			# sample from this by inversion and append
			# selected character to the output string

			cdf -= np.random.uniform()
			cdf[cdf<0] = np.nan
			text_out += cdf.index[ cdf.argmin() ]

		return text_out


	def predict(self,text):
		"""
		.predict( str: text_sample ) -> float
		retunrs the log probability of a text_sample

		how likely is text_sample given the models probability mass function
		(the log probability that text_sample was written by the same person as the models training text) 
		"""

		context, next_char = self._stride_split(text,self.n)
		prob_seq = []

		for x,y in zip(context,next_char):

			# for each context ngram and next_char, if it 
			# occurs in self._pmf, append the probability
			# to prob_seq

			if x in self._pmf.index and y in self._pmf.columns:
				prob_seq.append( self._pmf.loc[x,y] )

			# its possible that the text we are trying to 
			# assess the probability of containts n-grams and / or
			# characters out model has not seen.
			#
			# to avoid pandas indexing exceptions, if any unknown
			# ngrams or characters are encountered, we ignore

		return np.log(prob_seq).sum()

		# return the sum of the log-prob for each character in 'text'




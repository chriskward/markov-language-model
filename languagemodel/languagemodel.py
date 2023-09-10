import numpy as numpy
import pandas as pd

class MarkovModel():

		def __init__(self,data,n=5,alpha=0.001):

			self.alpha = alpha
			self.n = n
			if not isinstance(data,str) : raise TypeError('Training data must be provided as type: str')

			context,next_char = _stride_split(data, self.n)
			pmf = pd.DataFrame( index=np.unique(context), cols=np.unique(next_char), data=self.alpha )
			self._pmf = _estimate_probs(pmf,context,next_char)



		def _stride_split(self, text, n):

			text = np.array( text,dtype=np.str_ )
			buffer - np.frombuffer( text, dtype='U1' )
			context = np.lib.stride_tricks.sliding_window_view( buffer, (n,)).view('U'+str(n)).flatten()
			
			return context, buffer[n:]



		def _estimate_probs(self, pmf, context, next_char):

			for x in pmf.index:
				mask = np.argwhere(context==x).flatten()

				for y in next_char[mask]:
					pmf[y][x] += 1

			norm_constant = pmf.sum(axis=1)

			return pmf.divide(norm_constant, axis=0)



		def generate(self,length=100):

			text_out = np.random.choice(self._pmf.index)

			while len(text_out) < length:
				
				context = text_out[-self.n:]
				if context not in self._pmf.index:
					context = np.random.choice(self._pmf.index)

				cdf = np.cumsum( self._pmf.loc[context] )

				cdf -= np.random.uniform()
				cdf[cdf<0] = np.nan
				text_sample += cdf.index[ cdf.argmin() ]

			return text_sample


		def predict(self,text):

			context, next_char = _stride_split(text,self.n)
			prob_seq = []

			for x,y in zip(context,next_char):
				if x in self._pmf.index and y in self._pmf.columns:
					prob_seq.append( self._pmf.loc[x,y] )

			return np.log(prob_seq).sum()





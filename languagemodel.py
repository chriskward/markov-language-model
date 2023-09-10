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


		


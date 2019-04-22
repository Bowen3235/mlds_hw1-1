import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms 
import torch.utils.data as data
import numpy as np
import random as rd
from torch.autograd import Variable
import matplotlib.pyplot as plt

def Tar_Func( x ):
	y = np.sin ( 5*np.pi*x ) / (5*np.pi*x)
	#y = np.abs(3*x**2 - 0.5)
	return y

class Net(nn.Module):
	def __init__(self, input_size, hidden_size, output_size ):
		super( Net , self).__init__()
		self.rnn = nn.RNN( input_size = input_size , hidden_size = hidden_size , batch_first = True )
		self.linear = nn.Linear( hidden_size , output_size )
		self.act = nn.Tanh()

	def forward(self,x):

		pred, hidden = self.rnn( x , None )
		pred = self.act( self.linear(pred) ).view( pred.data.shape[0],-1,1 )

		return pred

def train( model, optimizer, epoch, data ,target , times  ):
	model.train()
	criterion = nn.MSELoss(  )
	for batch_idx in range( times ):
		hidden = None
		data = Variable( torch.Tensor(data.reshape(data.shape[0],-1,1)) ,requires_grad=True) 
		target = Variable( torch.Tensor(target.reshape(target.shape[0],-1,1)) ,requires_grad=True)
		#
		output = model( data )
		#print( output - target )

		lost = criterion( output , target )

		optimizer.zero_grad()
		lost.backward()
		optimizer.step()
		#learning_rate = 0.01

		'''for f in net.parameters():
			f.data.sub_(f.grad.data * learning_rate)
		'''
		print( 'Train Epoch {} , batch_num :{}, Loss: {:.6f}'.format( epoch , batch_idx , lost.data.item() )  )

def test( model , num , epoch , data , target ):
	
	net.eval()

	test_loss = 0
	correct = 0

	loss = nn.MSELoss(  )

	data = Variable( torch.Tensor(data.reshape(data.shape[0],-1,1)) ,requires_grad=True) 
	target = Variable( torch.Tensor(target.reshape(target.shape[0],-1,1)) ,requires_grad=True)
	output = model( data )
	test_loss += abs(loss( target ,output ).data.item())
	pred = output.data

	grah_x = []
	grah_y = []
	grah_p = []
		#print( data )
	#print( pred )
		#print( target )
	for i in range(len(pred)):
		for j in range(len(pred[i])):
			if np.abs( pred[i][j] - target.data[i][j] ) < 0.01:
				correct += 1
		#correct += pred.eq(target.data).sum()
	total = len( pred ) * len( pred[0] )
	pred = np.squeeze(np.array(pred))
	#print(target)

	test_loss /= total
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, total,
		100. * correct / total))
	if num == (epoch - 1):

		plt.plot( np.squeeze(data.detach().numpy()) , np.squeeze(target.detach().numpy())  , label = 'Target'  )
		plt.plot( np.squeeze(data.detach().numpy()) , np.array( pred ) , label = 'training'  )
		plt.legend()
		plt.show()



if __name__ == '__main__':
	batch = 100
	epoch = 10
	data_num = 8000

	x_train = []
	y_train = []
	x_test = []
	y_test = []
	for i in range(data_num):
		k = 1/1000 * rd.randint(1,999)
		if i % batch == 0:
			x_train.append([])
			y_train.append([])
		x_train[int( i/batch )].append( [k] )
		y_train[int( i/batch )].append( [Tar_Func( k )] ) 
		if i > data_num-500 :
			k = (i - data_num + 500 )/500
			x_test.append( k )
			y_test.append( Tar_Func( k ) )
	x_test = np.array(x_test)
	y_test = np.array(y_test)
	x_train = np.array(x_train[ : len(x_train) - 50  ]) 
	y_train = np.array(y_train[ : len(y_train) - 50  ])
	
	#print(x_train.shape)
	#print(y_train.shape)

	'''
	loader = data.DataLoader(
		dataset = data.TensorDataset( x_train , y_train ),
		batch_size = batch,
		shuffle = True)

	test_loader = data.DataLoader(
		dataset = data.TensorDataset( x_test.double() , y_test.double() ),
		batch_size = batch,
		shuffle = True)
	'''
	#net = Net( input_size = 1 , hidden_size = 10 , output_size = 1 )
	net = torch.load( 'modelsin_rnn.pt' )
	#print(net)

	#print( y_train )

	#print( list( net.parameters() )[0].size() )

	#print( list(net.parameters()))

	optimizer = optim.Adam( net.parameters() , lr = 1e-3 )

	totalp = 0
	for item in list(net.parameters()):
		totalp += len( item )
	print( totalp )

	times = int(data_num/batch)

	
	for i in range( epoch  ):
		train( net , optimizer , i+1 , x_train , y_train ,times )
		test( net , i , epoch , x_test , y_test )
	

	torch.save( net , 'modelsin_rnn.pt' )
	'''
	net.eval()
	pre = net(x_test).detach().numpy()
	#print( pre )
	plt.plot( np.array(x_test) , np.array(y_test) , label = 'Target' )
	plt.plot( np.array(x_test) , pre , label = 'training' )
	plt.legend()
	plt.show()
	'''






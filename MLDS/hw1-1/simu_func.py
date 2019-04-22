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

class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        assert x.shape[-1] % self._pool_size == 0, \
            'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[-1], self._pool_size)
        m, i = x.view(*x.shape[:-1], x.shape[-1] // self._pool_size, self._pool_size).max(-1)
        return m10
def Tar_Func( x ):
	y = np.sin ( 5*np.pi*x )
	#y = np.abs(3*x**2 - 0.5)
	#y=x
	return y

class Net(nn.Module):
	def __init__(self):
		super( Net , self).__init__()

		self.bnin = nn.BatchNorm1d( 1 , momentum = 0.5 )
		nn.init.normal_( self.bnin.weight , mean = 10 ,std = .1 )
		nn.init.constant_( self.bnin.bias , 3 )

		self.input = nn.Linear( 1 , 10 )
		self.ri = nn.LeakyReLU()
		nn.init.normal_( self.input.weight , mean = 10 ,std = .1 )
		nn.init.constant_( self.input.bias , 3 )
		#self.drop1 = nn.Dropout( 0.7 )
		self.r1 = nn.LeakyReLU()

		self.bn1 = nn.BatchNorm1d( 10 , momentum = 0.5 )
		nn.init.normal_( self.bn1.weight , mean = 10 ,std = .1 )
		nn.init.constant_( self.bn1.bias , 3 )

		self.fc1 = nn.Linear(10,10)
		nn.init.normal_( self.fc1.weight , mean = 10 ,std = .1 )
		nn.init.constant_( self.fc1.bias , 3 )


		#self.drop2 = nn.Dropout( 0.7 )
		self.r2 = nn.LeakyReLU()
		self.bn2 = nn. BatchNorm1d( 10 , momentum = 0.5 )
		nn.init.normal_( self.bn2.weight , mean = 10 ,std = .1 )
		nn.init.constant_( self.bn2.bias , 3 )

		self.fc2 = nn.Linear(10,10)
		nn.init.normal_( self.fc2.weight , mean = 10 ,std = .1 )
		nn.init.constant_( self.fc2.bias , 3 )

		#self.drop3 = nn.Dropout( 0.7 )
		self.r3 = nn.LeakyReLU()
		self.bn3 = nn.BatchNorm1d( 10 , momentum = 0.5 )
		nn.init.normal_( self.bn3.weight , mean = 10 ,std = .1 )
	#nn.init.constant_( self.bn3.bias , 3 )

		self.fc3 = nn.Linear(10,10)
		nn.init.normal_( self.fc3.weight , mean = 10 ,std = .1 )
		nn.init.constant_( self.fc3.bias , 3 )
		#self.drop4 = nn.Dropout( 0.7 )
		self.r4 = nn.LeakyReLU()
		self.bn4 = nn. BatchNorm1d( 10 , momentum = 0.5 )
		nn.init.normal_( self.bn4.weight , mean = 10 ,std = .1 )
		nn.init.constant_( self.bn4.bias , 3 )

		self.fc4 = nn.Linear(10,10)
		nn.init.normal_( self.fc4.weight , mean = 10 ,std = .1 )
		nn.init.constant_( self.fc4.bias , 3 )

		#self.drop5 = nn.Dropout( 0.7 )
		self.r5 = nn.LeakyReLU()
		self.bn5 = nn.BatchNorm1d( 10 , momentum = 0.5 )
		nn.init.normal_( self.bn5.weight , mean = 5 ,std = .1 )
		nn.init.constant_( self.bn5.bias , 3 )

		self.fc5 = nn.Linear(10,5)
		nn.init.normal_( self.fc3.weight , mean = 5 ,std = .1 )
		nn.init.constant_( self.fc5.bias , 3 )

		#self.drop6 = nn.Dropout( 0.7 )
		self.r6 = nn.LeakyReLU()
		self.bn6 = nn.BatchNorm1d( 5 , momentum = 0.5 )
		nn.init.normal_( self.bn6.weight , mean = 5 ,std = .1 )
		nn.init.constant_( self.bn6.bias , 3 )

		self.fc6 = nn.Linear(5,1)
		#nn.init.normal_( self.fc6.weight , mean = 0 ,std = .1 )
		#nn.init.constant_( self.fc6.bias , 3 )





		#self.input.weight.data.fill_(0)
		#self.fc1.weight.data.fill_(0)
		#self.fc2.weight.data.fill_(0)
		#self.fc3.weight.data.fill_(0)
	def forward(self,x):

		x = self.ri(self.input( self.bnin(x.float()) ) )
		#x = self.drop1( x )
		x = self.r1(self.fc1(self.bn1(x)))
		#x = self.drop2( x )
		x = self.r2(self.fc2(self.bn2(x)))
		#x = self.drop3( x )
		x = self.r3(self.fc3(self.bn3(x)))
		#x = self.drop4( x )
		x = self.r4(self.fc4(self.bn4(x)))
		#x = self.drop5( x )
		x = self.r5(self.fc5(self.bn5(x)))
		#x = self.drop6( x )
		result = self.r6(self.fc6(self.bn6(x)))
		return result

def train( model, optimizer, epoch, train_loader  ):
	model.train()
	criterion = nn.MSELoss( size_average = False )
	for batch_idx, ( data, target ) in enumerate( train_loader ):
		data, target = Variable( data ,requires_grad=True) , Variable( target ,requires_grad=True).float()
		#print( data )
		optimizer.zero_grad()
		output = model( data )
		#print( output )
		#print( list(model.parameters())[2].grad )
		lost = criterion( output , target )

		
		lost.backward()
		#print( model.input.grad_fn )
		optimizer.step()
		#learning_rate = 0.01

		'''for f in net.parameters():
			f.data.sub_(f.grad.data * learning_rate)
		'''
		print( 'Train Epoch {} , batch_num :{}, Loss: {:.6f}'.format( epoch , batch_idx , lost.data.item() )  )

def test( model , test_loader ):
	
	net.eval()

	test_loss = 0
	correct = 0

	loss = nn.MSELoss(  )

	for data,target in test_loader:
		output = model( data )
		target = target.float()
		test_loss += abs(loss( target ,output ).data.item())
		pred = output.data
		#print( data )
		#print( pred )
		#print( target )
		for i in range(len(pred)):
			if np.abs( pred[i] - target.data[i] ) < 0.01:
				correct += 1
		#correct += pred.eq(target.data).sum()

	test_loss /= len(test_loader)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
	x_train = []
	y_train = []
	x_test = []
	y_test = []
	for i in range(150000):
		k = 1/1000 * rd.randint(1,999)
		if i < (150000-500):
			x_train.append( [k] )
			y_train.append( [Tar_Func( k )] )
			#print ( k , Tar_Func(k))
		else:
			k = (i-150000+500) * 1/500
			x_test.append( [k] )
			y_test.append( [Tar_Func( k )] )
	x_train = torch.from_numpy( np.array(x_train) )
	y_train = torch.from_numpy( np.array(y_train) )
	x_test = torch.from_numpy( np.array(x_test) )
	y_test = torch.from_numpy( np.array(y_test) )
	#print(x_train.shape)

	batch = 100
	epoch = 100

	loader = data.DataLoader(
		dataset = data.TensorDataset( x_train , y_train ),
		batch_size = batch,
		shuffle = True)

	test_loader = data.DataLoader(
		dataset = data.TensorDataset( x_test.double() , y_test.double() ),
		batch_size = batch,
		shuffle = True)

	net = Net( )
	#net = torch.load( 'modelsimple.pt' )
	print(net)
	#print( y_train )

	#print( list( net.parameters() )[0].size() )

	#print( list(net.parameters()))

	optimizer = optim.Adam( net.parameters()  , lr = 1e-4 )

	
	for i in range( epoch  ):
		train( net , optimizer , i+1 , loader )
		test( net , test_loader )
	

	net.eval()
	pre = net(x_test).detach().numpy()
	#print( pre )
	plt.plot( np.array(x_test) , np.array(y_test) , label = 'Target' )
	plt.plot( np.array(x_test) , pre , label = 'training' )
	plt.legend()
	plt.show()
	torch.save( net , 'modelsimple.pt' )






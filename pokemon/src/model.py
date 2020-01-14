import torch
import torch.nn as nn
import torch.nn.functional as F


DEBUG = False


class myconv( nn.Module ):
	"""
	only need to input in_dim and out_dim with channels,auto handle kernel
	stride default to be 1
	normalize default to be true
	dropout default to be 0
	"""
	def __init__( self, in_ch, out_ch, in_dim, out_dim, stride=2, normalize=True, dropout=0 ):
		super( myconv, self ).__init__()
		K = int( in_dim - stride*( out_dim-1 ) )

		layers = [ nn.Conv2d( in_ch, out_ch, K, stride ) ]

		if normalize:
			layers.append( nn.BatchNorm2d( out_ch ) )
		layers.append( nn.LeakyReLU(  ) )

		if dropout :
			layers.append( nn.Dropout(dropout) )
		self.model = nn.Sequential( *layers )
	def forward( self, x ):

		## debug messange
		if DEBUG:
			print( '##########' )
			print( x.shape )

		x = self.model( x )

		## debug messange
		if DEBUG:
			print( x.shape )
			print( '##########' )
		return x

class mytransconv( nn.Module ):
	"""
	only need to input in_dim and out_dim with channels,auto handle kernel
	stride default to be 1
	normalize default to be true
	dropout default to be 0
	"""
	def __init__( self, in_ch, out_ch, in_dim, out_dim, stride=2, normalize=True, dropout=0 ):
		super( mytransconv, self ).__init__()
		K = int( out_dim - (in_dim-1)*stride )

		layers = [ nn.ConvTranspose2d( in_ch, out_ch, K, stride ) ]

		if normalize:
			layers.append( nn.BatchNorm2d( out_ch ) )
		layers.append( nn.LeakyReLU(  ) )

		if dropout:
			layers.append( nn.Dropout(dropout) )
		self.model = nn.Sequential( *layers )
	def forward( self, x ):

		## debug messange
		if DEBUG:
			print( '##########' )
			print( x.size() )

		x = self.model( x )

		## debug messange
		if DEBUG:
			print( x.size() )
			print( '##########' )
		return x


##############################
#        generator
##############################

class generator( nn.Module ):
	""" 
	10,1 -> 2028,4 -> 1024,8 -> 512,16 -> 
	256,32 -> 128,32 -> 64,64 -> 64, 128 -> 3, 360
	"""
	def __init__( self ):
		super( generator, self ).__init__()
		self.In = nn.Sequential( 
			nn.Linear(10, 256),
			nn.ReLU( ),
			nn.Linear(256,1024),
			nn.ReLU( ),
			nn.Linear(1024,4*1024) )

		## reshape to _,1024,2,2
		self.layers = nn.Sequential(
			mytransconv( 1024, 1024 ,2 ,8 ),
			mytransconv( 1024, 512, 8, 16 ),
			mytransconv( 512, 256, 16, 32 ),
			mytransconv( 256, 128, 32, 64 ),
			mytransconv( 128, 64, 64, 128 ),
			mytransconv( 64, 3, 128, 360 ),
			nn.Tanh()
		)
	def forward( self, x ):

		if DEBUG:
			print( '##########raw' )
			print( x.size() )
			print( '##########in' )

		x = self.In( x )

		x = x.view( -1, 1024, 2, 2 )
		if DEBUG:
			print( '##########in' )
			print( x.size() )
			print( '##########layers' )
		x = self.layers( x )

		return x

##############################
#        discriminator
##############################

class discriminator(nn.Module):
	"""docstring for discriminator"""
	def __init__(self):
		super(discriminator, self).__init__()
		self.layers = nn.Sequential(
			myconv( 3, 64, 360, 128 ),
			myconv( 64, 128, 128, 64 ),
			myconv( 128, 256, 64, 32 ),
			myconv( 256, 512, 32, 16 ),
			myconv( 512, 1024, 16, 4 ),
			myconv( 1024,1024, 4,  1 , normalize = False)
		)
		## reshape
		self.final = nn.Sequential(
			nn.Linear( 1024, 512 ),
			nn.ReLU( ),
			nn.Linear( 512, 256),
			nn.ReLU( ),
			nn.Linear( 256, 128 ),
			nn.ReLU( ),
			nn.Linear( 128, 1 ),
			nn.Sigmoid()
			)
	def forward( self,  x ):
		x = self.layers( x )
		
		x = x.view( -1, 1024 )

		x = self.final( x )

		return x








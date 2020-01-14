import numpy as np
import torch
import torch.utils.data as tdata 
import torchvision.transforms as transforms
from torch.autograd import Variable
import time
import os
import datetime
import sys
import random
import model as model
import psutil
from tensorboardX import SummaryWriter 
import matplotlib as mlt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import shutil
import re

DEBUG = True

def now_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# torch.cuda.set_device(4)
comment = 'one_data-GAN{}/'.format(now_time())

if DEBUG:
	comment = 'debug_GAN_one_input_g:1e-5,git:10'
	if os.path.exists( 'pokemon/.runs/'+comment ):
		shutil.rmtree('pokemon/.runs/'+comment )
torch.cuda.empty_cache()


_batch_size = 10
epoch = 100000
# lamda_pix + lamda_tag < 0.5
lamda_pix = 0.2
lamda_tag = 0.2

train_num = 100
test_num =100
if not os.path.exists( 'pokemon/.runs/' ):
	os.mkdir( 'pokemon/.runs/' ) 
if not os.path.exists( 'pokemon/.runs/'+comment ):
	os.mkdir( 'pokemon/.runs/'+comment ) 
if not os.path.exists(  "pokemon/log/" ):
	os.mkdir( "pokemon/log/" )

Summary_Path = 'pokemon/.runs/'+comment
concole_log_file = "pokemon/log/"+comment+"_concole.log"
loss_log_file = "pokemon/log/"+comment+"_loss.log"
train_data_path = 'pokemon/pic' 
test_data_path  = 'pokemon/pic/test.txt'

file = open( loss_log_file , 'w' )
file.write(now_time()+'\n' )
file.close()

file = open( concole_log_file , 'w' )
file.write( now_time()+'\n' )
file.close()


def main():

	writer = SummaryWriter( Summary_Path )
	Gen = model.generator()
	Dis = model.discriminator()

	criterion_GAN = torch.nn.BCELoss()
	criterion_VF = torch.nn.SmoothL1Loss()
	criterion_pixelwise = torch.nn.BCELoss()
	optimizer_G = torch.optim.Adam(Gen.parameters(), lr=1e-5)
	optimizer_D = torch.optim.Adam(Dis.parameters(), lr=1e-5)
	d_iter = 1
	g_iter = 10
	
	cuda = True if torch.cuda.is_available() else False

	if cuda:
		Gen = Gen.cuda()
		Dis = Dis.cuda()
		criterion_GAN.cuda()
		criterion_pixelwise.cuda()
		criterion_VF.cuda()
		torch.set_default_tensor_type('torch.cuda.FloatTensor')
	Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

	
	## reading file
	test_list = np.loadtxt( train_data_path+'/test.txt' )
	test_data = []
	train_data = []
	for filename in os.listdir( train_data_path ):
		if filename == 'test.txt':
			continue
		img = plt.imread( train_data_path +'/'+ filename )
		new_img = np.zeros( (3,360,360) )
		new_img[0,:,:] = img[:,:,0]
		new_img[1,:,:] = img[:,:,1]
		new_img[2,:,:] = img[:,:,2]
		new_img = np.array( new_img )/255.0
		if int( filename[ :filename.find('-') ] ) in test_list:
			test_data.append( new_img )
		else:
			train_data.append( new_img )
		if DEBUG:
			if np.shape( train_data )[0] >= 10 :
				break
	if DEBUG:
		train_data = train_data[0:10]
		test_data = train_data
		# print( train_data[0] )
	print( np.shape( train_data ) )
	training_set = tdata.DataLoader( 
			np.array(train_data)/255.0,
			batch_size=_batch_size,
			shuffle = True)

	test_set = tdata.DataLoader( 
			np.array(test_data)/255.0,
			batch_size= _batch_size,
			shuffle = False)
	train_num = np.shape( train_data )[0]
	test_num = np.shape( test_data )[0]
	print( np.random.randint( test_num, size= 1 )[0] )
	print( np.shape( test_data[ np.random.randint( test_num, size= 1 )[0] ] ) )
	writer.add_image( 'true_figure/ground1' , test_data[ np.random.randint( test_num, size= 1 )[0] ] )
	writer.add_image( 'true_figure/ground2' , test_data[ np.random.randint( test_num, size= 1 )[0] ] )
	writer.add_image( 'true_figure/ground3' , test_data[ np.random.randint( test_num, size= 1 )[0] ] )
	writer.add_image( 'true_figure/ground4' , test_data[ np.random.randint( test_num, size= 1 )[0] ] )


	print( 'End of loading Data' )
	file = open( concole_log_file , 'a+' )
	file.write( now_time() + 'End of loading Data'+'\n' )
	file.close()
    
	G_training_loss = []
	D_training_loss = []
	G_testing_loss = []
	D_testing_loss = []
    
	for epoch_num in range( epoch ):
		G_training_loss.append([])
		D_training_loss.append([])
		for i, ( PIC ) in enumerate( training_set ):
			Gen.train()
			Dis.train()

			# print( type(PIC) )

			for j in range( g_iter ):

				PIC = Variable( PIC.float(), requires_grad=False )
				Noise = Tensor( np.random.normal( 0, 1, 10*PIC.size(0)) ).view(-1,10)
				valid = Variable(torch.ones( (PIC.size(0),1) , device=torch.device( 'cuda' if cuda else 'cpu') ), requires_grad=False)
				fake = Variable(torch.zeros( (PIC.size(0),1) , device=torch.device( 'cuda' if cuda else 'cpu') ), requires_grad=False)

				#--------------
				#   train gen
				#--------------

				file = open( concole_log_file , 'a+' )
				file.write( now_time() + 'train gen' +'\n' )
				file.close()
	            
				print( '#######train gen' )

				Gen.zero_grad()

				### tracing loss
				fake_PIC = Gen( Noise )

				pred_fake = Dis( torch.div(torch.add(fake_PIC.detach(),1),2) )
				
				# discriminator loss
				loss_GAN =  criterion_GAN( pred_fake, valid )
				
				#Total
				loss_G = loss_GAN
				loss_G.backward()
				optimizer_G.step()


			#--------------
			#   train dis
			#--------------
			for j in range(d_iter):
				file = open( concole_log_file , 'a+' )
				file.write( now_time() + 'train dis' +'\n' )
				file.close()
	            
				print( '#######train dis' )
				Dis.zero_grad()

				## traing loss ##

				#true pic
				pred_real = Dis( PIC )
				loss_real = criterion_GAN( pred_real, valid )
				
				# fakevd pic
				pred_fake = Dis( torch.div(torch.add(fake_PIC.detach(),1), 2) )
				loss_fake = criterion_GAN( pred_fake, fake )

				loss_D = (loss_real+loss_fake)*0.5
				loss_D.backward()
				optimizer_D.step()



			# --------------
			#  Log Progress
			# --------------

			# Print log
			print(
				"\r[Epoch {0}/{1}] [Batch {2}/{3}] [D loss: {4:.2f}] [G loss: {5:.2f}] ".format(
					epoch_num+1,
					epoch,
					i+1,
					len(training_set),
					loss_D.item(),
					loss_G.item()
				)
			)
			file = open( concole_log_file , 'a+' )
			file.write( now_time() +  "\n[Epoch {0}/{1}] [Batch {2}/{3}] [D loss: {4:.2f}] [G loss: {5:.2f}] ".format(
					epoch_num+1,
					epoch,
					i+1,
					len(training_set),
					loss_D.item(),
					loss_G.item()
				)+'\n')
			file.close()
			G_training_loss[-1].append(loss_G)
			D_training_loss[-1].append(loss_D)
			if i%2 == 0 :
				writer.add_scalar( 'Train_Loss/Gen' , loss_G.item(), epoch_num*train_num+i )
				writer.add_scalar( 'Train_Loss/Dis' , loss_D.item(), epoch_num*train_num+i )
				

				
		# ----------
		# testing
		# ----------


		Gen.eval()
		Dis.eval()
		lossg = 0
		lossd = 0
		for i, ( PIC ) in enumerate( test_set ):
			# IN = Variable( torch.ones( IN.size(),requires_grad=True ,device= torch.device('cuda') ) )
			# SE = Variable( torch.ones( SE.size(),requires_grad=True ,device= torch.device('cuda') ) )
			PIC = PIC.float()

			valid = Variable(torch.ones( (PIC.size(0),1) , device=torch.device( 'cuda' if cuda else 'cpu') ), requires_grad=False)
			fake = Variable(torch.zeros( (PIC.size(0),1) , device=torch.device( 'cuda' if cuda else 'cpu') ), requires_grad=False)
			

			#Total loss
			Noise = Tensor( np.random.normal( 0, 1, 10*PIC.size(0)) ).view(-1,10)
			fake_PIC = Gen( Noise )


			if (i == 0) and ((epoch_num+1)%10 == 0) :
				writer.add_image( 'figure/pred1' , torch.div(torch.add(fake_PIC[ np.random.randint(int(PIC.size(0))) ].detach().view(3,360,360), 1), 2), epoch_num+1)
				writer.add_image( 'figure/pred2' , torch.div(torch.add(fake_PIC[ np.random.randint(int(PIC.size(0))) ].detach().view(3,360,360), 1), 2), epoch_num+1)
				writer.add_image( 'figure/pred3' , torch.div(torch.add(fake_PIC[ np.random.randint(int(PIC.size(0))) ].detach().view(3,360,360), 1), 2), epoch_num+1)
				writer.add_image( 'figure/pred4' , torch.div(torch.add(fake_PIC[ np.random.randint(int(PIC.size(0))) ].detach().view(3,360,360), 1), 2), epoch_num+1)
				print( '################\nadding figure\n################' )
			
			## loss gen 
			pred_fake = Dis( fake_PIC )

			loss_GAN =  criterion_GAN( pred_fake, valid )

			loss_G = loss_GAN
			## loss gen 

			## loss ids
			pred_real = Dis( PIC )
			loss_real = criterion_GAN( pred_real, valid )
			pred_fake = Dis( fake_PIC.detach() )
			loss_fake = criterion_GAN( pred_fake, fake )

			loss_D =  (loss_real+loss_fake)*0.5

			## loss ids

			
			lossg = lossg + loss_G.item()
			lossd = lossd + loss_D.item()

		print( "testing : [Epoch {0}/{1}] [D loss: {2:.2f}] [G loss: {3:.2f}] ".format( 
			epoch_num+1,epoch,lossd ,lossg ) )
        
		file = open( concole_log_file , 'a+' )
		file.write( now_time() + "testing : [Epoch {0}/{1}] [D loss: {2:.2f}] [G loss: {3:.2f}]".format( 
			epoch_num+1,epoch,lossd ,lossg )+'\n' )
		file.close()
        
		G_testing_loss.append( lossg )
		D_testing_loss.append( lossd )
		print( '################\nwriting testing loss\n################' )
		writer.add_scalar( 'Test_Loss/Gen' , lossg, epoch_num+1 )
		writer.add_scalar( 'Test_Loss/Dis' , lossd, epoch_num+1 )


		if (epoch_num+1) % 5 == 0:
			if not os.path.exists( 'Dir_/model/'+comment+'/' ):
				os.mkdir( 'Dir_/model/'+comment+'/' )
			if not DEBUG:
				torch.save( Gen.state_dict() , 'Dir_/model/'+comment+'/generator'+str(epoch_num + 1) )
				torch.save( Gen.state_dict() , 'Dir_/model/'+comment+'/discriminator'+str(epoch_num + 1) )
			print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  )
			file = open( loss_log_file , 'a+' )
			file.write( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'\n'  )
			file.write( 'G training : \n' )
			file.write( str(G_training_loss)  )
			file.write( '\n' )
			file.write( 'D training : \n' )
			file.write( str(D_training_loss)  )
			file.write( '\n' )
			file.write( 'G testing : \n' )
			file.write( str(G_testing_loss ) )
			file.write( '\n' )
			file.write( 'D testing : \n' )
			file.write( str(D_testing_loss ) )
			file.write( '\n' )
			file.close()
			D_training_loss = []
			G_training_loss = []
			G_testing_loss = []
			D_testing_loss = []
	writer.close()


















if __name__ == '__main__':
	main()
	print('done')
	torch.cuda.empty_cache()



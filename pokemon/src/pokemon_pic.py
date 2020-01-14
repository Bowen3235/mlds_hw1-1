import urllib
import requests
import os
import numpy as np
from bs4 import BeautifulSoup
import cv2

user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

class myWeb(object):
	"""docstring for myWeb"""
	def __init__(self, url):
		super(myWeb, self).__init__()
		self.url = url
		self.page =  requests.get( self.url , headers={'user-agent':user_agent} )
		self.soup =  BeautifulSoup( self.page.text, 'html5lib' )

def main():
	_listing_page = myWeb('https://pokemondb.net/pokedex/all')
	if not os.path.exists('pic'):
		os.mkdir( 'pic' )
	All_name   = []
	All_type   = []
	All_Weight = []
	All_Height = []
	All_Hp  = []
	All_Att = []
	All_Def = []
	All_Spa = []
	All_Spd = []
	All_Spe = []
	mutant = 0
	# print( len(_listing_page.soup.find_all( 'tr' )) )
	
	for num ,item in  enumerate(_listing_page.soup.find_all( 'tr' )):
		if num == 0:
			continue
		if num < 0:
			break
		#num = 405
		#item = _listing_page.soup.find_all( 'tr' )[num]
		# mutant = 0
		# in for loop
		# print( type(item.find('a') ))
		# print( item.find('a')['href'] )
		# print( item.find('small') )
		# print( item.find('a').text )



		############
		##getting name
		############
		name = item.find('a').text
		if item.find( 'small' ) :
			name = item.find( 'small' ).text
			mutant = mutant+1
		else :
			mutant = 0

		############
		##saving pic
		############
		href = item.find('a')['href']
		_child_page = myWeb( 'https://pokemondb.net'+href )
		for img_src in _child_page.soup.find_all('img' ) :
			# print( 'in loop : ' ,img_src['data-title'] )
			# print( 'in loop : ' ,name )
			# print( img_src['data-title'].find( name ) )
			if img_src['alt'].find( name ) != -1:
				img_src = img_src['src']
				break
		img_type = img_src[-4:]
		# print( img_type )
		urllib.request.urlretrieve( img_src, 'pic/'+str(num)+'-'+name+img_type )
		img = cv2.imread( 'pic/'+str(num)+'-'+name+img_type ,cv2.IMREAD_UNCHANGED )
		img = cv2.resize( img , (360,360) )
		if img_type!= '.jpg' :
			for j in range(3):
				img[ :,:,j ] = np.where( img[:,:,3]==0 , 255 , img[ :,:,j ] )
			# print( np.shape(img) ) 
			os.remove( 'pic/'+str(num)+'-'+name+img_type )

		cv2.imwrite( ('pic/'+str(num)+'-'+name+'.jpg'), img )
		
		############
		##getting height/weight
		############
		count = 0
		for i , table in enumerate( _child_page.soup.find_all( 'table' ) ):
			if table.find( 'th' , text = 'Height' ):
				if count == mutant:
					# print( table.find( 'th' , text = 'Height' ).next_sibling.next_element.text )
					Height = table.find( 'th' , text = 'Height' ).next_sibling.next_element.text
					Height = float(Height[0: Height.find('m')-1 ])
					# print( float(height) )
					Weight = table.find( 'th' , text = 'Weight' ).next_sibling.next_element.text
					Weight = float(Weight[0: Weight.find('kg')-1 ])
					# print( float(weight) )
				count +=1 

		############
		## Type
		############

		Thetype = item.find_all('td')[2].text.split()

		if len( Thetype )==1:
			Thetype = [ Thetype[0].strip() , Thetype[0].strip() ]
	 
		# for i in Thetype:
		#	print( '\''+i+'\'' )
		# print( item.find_all('td')[2].text )

		############
		## other value
		############
		HP = int(item.find_all('td')[4].text.strip())
		Att = int(item.find_all('td')[5].text.strip())
		Def = int(item.find_all('td')[6].text.strip())
		Spa = int(item.find_all('td')[7].text.strip())
		Spd = int(item.find_all('td')[8].text.strip())
		Spe = int(item.find_all('td')[9].text.strip())
		# print( HP, Att, Def, Spa, Spd, Spe )

		############
		## saving Values
		###########
		All_name.append(name)
		All_type.append( Thetype )
		All_Weight.append(Weight)
		All_Height.append(Height)
		All_Hp .append(HP)
		All_Att.append(Att)
		All_Def.append(Def)
		All_Spa.append(Spa)
		All_Spd.append(Spd)
		All_Spe.append(Spe)
		print( num , name )
		if name == 'Melmetal' :
			break
		#break
	if not os.path.exists('Data'):
		os.mkdir( 'Data' )
	np.savez( 'Data/data.npz',
	Name = All_name,
	Type = All_type,
	Weight = All_Weight,
	Height = All_Height,
	Hp =All_Hp,
	Att = All_Att,
	Def = All_Def,
	Spa = All_Spa,
	Spd = All_Spd,
	Spe = All_Spe )



if __name__ == '__main__':
	main()
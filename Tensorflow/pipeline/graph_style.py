from matplotlib import rc


rc('font',**{'family':'sans-serif','sans-serif':['Lato'], 'weight':'light', 'size':20})
rc('text', usetex=False)
rc('ytick', labelsize = 15)
rc('xtick', labelsize = 15)
rc('figure', **{'titlesize':25, 'titleweight':'light', 'figsize':(15,8)})
rc('legend', **{'fontsize':20, 'frameon': False})
rc('axes', **{'labelsize':22, 'labelweight':'light', 'spines.top': 'False', 'spines.right':'False',
			'prop_cycle': "cycler('color',['2476AB', 'EF9C34', 'D22C2C', 'E5D81D', '8E44AD','1EB22C', '38C6BE','C638A2'])",
			'titlesize': 25, 'titleweight': 'light'
	})

colour_dict = {
	"default":[
			    '#2476AB', # 0 Blue
			    '#EF9C34', # 1 Orange
			    '#D22C2C', # 2 Red
			    '#8E44AD', # 3 Purple  
			    '#E5D81D', # 4 Yellow
			    '#1EB22C', # 5 Green
			    '#C638A2', # 6 Pink
			    '#38C6BE', # 7 Turquoise
   				],
   	"datascience_4": [
				'#513A8A', # Dark Purple
				'#1786BD', # Sky Blue
				'#4B4D98', # Purple
				'#2A72AD', # Blue		
						],
	"datascience_8": [
				'#513A8A', # Dark Purple
				'#108ECF', # Sky Blue
				'#4D4C9B', # Purple
				'#1285BD', # Pale Blue
				'#4657A7', # Indigo
				'#3C5DA1', # Dark Blue
				'#197AB2', # Turquoise
				'#2F66A2', # Blue			
				]
}


def set_colour_cycle(colour_choice = colour_dict["default"]):
	rc('axes', **{'prop_cycle': "cycler('color',{})".format(colour_choice)})

\* Whiskas *\
Minimize
OBJ: 0.008 _ing_beef + 0.013 _ing_chicken
Subject To
fat_req: 0.2 _ing_beef + 0.08 _ing_chicken >= 6
fibre_req: 0.005 _ing_beef + 0.001 _ing_chicken <= 2
protein_req: 0.2 _ing_beef + 0.1 _ing_chicken >= 8
salt_req: 0.005 _ing_beef + 0.002 _ing_chicken <= 0.4
total_req: _ing_beef + _ing_chicken = 100
Bounds
_ing_beef free
_ing_chicken free
End

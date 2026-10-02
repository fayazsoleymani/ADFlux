import casadi as ca

class Rxn():

    def __init__(self, rxn_id, reactants, products, stoich_coefs, atom_mapping, rev= 0, flux= None):
        self.rxn_id= rxn_id
        self.reactants= reactants
        self.products= products
        self.stoich_coefs= stoich_coefs
        self.rev = rev
        self.atom_mapping= atom_mapping
        self.flux= flux
        # self.v= ca.MX.sym(self.rxn_id)


    def __str__(self):
        #TODO
        return self.rxn_id
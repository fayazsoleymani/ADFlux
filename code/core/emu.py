import numpy as np
import casadi as ca

class EMU():

    def __init__(self, emu_string):
        self.met= "_".join(emu_string.split('_')[0:-1])
        self.pos= [int(digit)-1 for digit in emu_string.split('_')[-1]]
        self.MID= np.zeros(1 + len(self.pos))
        self.size= len(self.pos)
        self.solved = False
        self.inst_fun= None


    def EMU2EMU(self, rxn):
        # retrive the EMUs, which are producing the EMU in the specific rxn

        # finding the atoms of the EMU in the products side
        side_index = rxn.products.index(self.met)
        atoms= "".join([rxn.atom_mapping[1][side_index][i] for i in self.pos])

        # finding the corresponding atoms based on the atom transitions

        # creating dictionary which saves the metabolites as the keys and their corresponding atom index if it matches
        # the target EMU atom
        index2loc = {k: [] for k in rxn.reactants}
        mapping2reactant= dict()
        mapping2rep= dict()
        counter = 0
        for l, reactant in enumerate(rxn.reactants):
            n= max(1, -rxn.stoich_coefs[l])
            for rep in range(int(n)):
                index2loc[reactant].append([])
                mapping2reactant[counter]= l
                mapping2rep[counter]= rep
                counter += 1

        for atom in atoms:
            for mapping_index, mol in enumerate(rxn.atom_mapping[0]):
                if atom in mol:
                    reactant_index= mapping2reactant[mapping_index]
                    index2loc[rxn.reactants[reactant_index]][mapping2rep[mapping_index]].append(mol.index(atom))

        # combining metabolites with their atom indexes and creating retrieved EMUs
        emus = []
        for reactant, indices_list in index2loc.items():
            for indices in indices_list:
                if indices:
                    indices_str = [str(index + 1) for index in indices]
                    emus.append(reactant + '_' + ''.join(sorted(indices_str)))
        return emus


    def inverse_EMU(self, emu_size):
        complete= list(reversed(range(emu_size)))
        inversed_pos= sorted([complete[pos] for pos in self.pos])
        return self.met + '_' + ''.join(str(digit + 1) for digit in inversed_pos)


    def __str__(self):
        return self.met + '_' + ''.join(str(digit + 1) for digit in self.pos)

    def __eq__(self, other):
        if isinstance(other, EMU):
            return self.met == other.met and self.pos == other.pos
        else:
            return False

    def __lt__(self, other):
        return self.size < other.size

    def __hash__(self):
        return hash(str(self))


class JointEMU():

    def __init__(self, emus):
        self.emus= emus
        self.size= sum(emu.size for emu in emus)
        self.solved= True

    def inst_fun(self, t):
        first_mid = self.emus[0].inst_fun(t)
        second_mid = self.emus[1].inst_fun(t)
        return np.convolve(first_mid, second_mid)

    @property
    def MID(self):
        return np.convolve(self.emus[0].MID, self.emus[1].MID)

    @property
    def x(self):
        conv = ca.MX.zeros(1, self.size + 1)

        for i in range(self.emus[0].size + 1):
            for j in range(self.emus[1].size + 1):
                conv[0, i + j] += (self.emus[0].x[0, i] * self.emus[1].x[0, j])
        return conv

    def __str__(self):
        return '*'.join([str(emu) for emu in self.emus])

    def __eq__(self, other):
        if isinstance(other, JointEMU):
            return self.emus[0] == other.emus[0] and self.emus[1] == other.emus[1]
        else:
            return False

    def __hash__(self):
        return hash(str(self))

class EMURxn():

    def __init__(self, start, rxn, end, rxn_coef= 1.0):
        self.start = start
        self.rxn = rxn
        self.rxn_coef= rxn_coef
        self.end = end
        assert start.size == end.size
        self.size= len(end.pos)


    def __str__(self):
        return str(self.rxn_coef) + self.rxn.rxn_id + ' : ' + str(self.start) + '->' + str(self.end)

    def __eq__(self, other):
        return str(self) == str(other)


class EMUNetwork():

    def __init__(self):
        self.V= set()
        self.E= []
        self.stack= []
        self.visited= dict()
        self.created_count= 0
        self.rotational_emus2emu= dict()
        self.rotational_emu2binary= dict()

    def add_EMU_reaction(self, emu_rxn):

        if isinstance(emu_rxn.start, EMU):
            self.V.add(emu_rxn.start)
        elif isinstance(emu_rxn.start, JointEMU):
            for emu in emu_rxn.start.emus:
                self.V.add(emu)

        self.V.add(emu_rxn.end)

        if emu_rxn in self.E:
            index= self.E.index(emu_rxn)
            self.E[index].rxn_coef += emu_rxn.rxn_coef
        else:
            self.E.append(emu_rxn)



    def __str__(self):
        temp_str= ''
        temp_str += "Number of Nodes:\t" + str(len(self.V))
        temp_str += "\nNumber of Edges:\t" + str(len(self.E))
        return temp_str


class EMUSubNetwork(EMUNetwork):

    def __init__(self):
        super().__init__()

    def add_EMU_reaction(self, emu_rxn):
        self.V.add(emu_rxn.start)
        self.V.add(emu_rxn.end)
        self.E.append(emu_rxn)

    def __bool__(self):
        return len(self.V) > 0


def generate_EMU_network(model):
    # traversing the EMU graph based on dfs
    emu_net = EMUNetwork()

    emu_net.stack= [emu for emu in model.target_emus]
    emu_net.created_count= len(emu_net.stack)


    while len(emu_net.stack) > 0:

        # visiting one node from the stack
        emu_net.stack.sort()
        emu = emu_net.stack.pop()
        emu_net.visited[str(emu)]= emu

        if emu.met in model.uptake_mets:
            continue


        decomposition_dict, emu_net= decompose_EMU(emu, model, emu_net)

        for rxn, mapped_emu in decomposition_dict.items():

            coef= 1
            if len(mapped_emu) > 1:
                coef *= 0.5
            if emu.met in model.rotational_mols and emu_net.rotational_emu2binary.get(str(emu), False):
                coef *= 0.5

            # adding the edge
            for decomposed_emu in mapped_emu:
                emu_net.add_EMU_reaction(EMURxn(decomposed_emu, rxn, emu_net.rotational_emus2emu.get(str(emu), emu), coef))

    # assert created_count == len(visited)
    emu_net.V= sorted(emu_net.V, key=lambda x: (x.size, str(x)))
    return emu_net


def decompose_EMU(emu, model, emu_net):

    rxn2decomposedEMUs= dict()
    if emu.met in model.mets:
        producing_rxns_indices = np.where(model.S[model.mets.index(emu.met)] > 0)[0]
        producing_rxns = [model.rxns[i] for i in producing_rxns_indices if model.rxns[i] not in model.biomass]
    else:
        producing_rxns= model.export_rxns[emu.met]
    for rxn in producing_rxns:

        # obtaining the lhs of the EMU reaction based on atom mapping of the rxn
        mapped_emu_list = emu.EMU2EMU(rxn)

        if len(mapped_emu_list) == 1:
            mapped_emu_str = mapped_emu_list[0]
            mapped_emu, emu_net, temp = check_new_emu(mapped_emu_str, emu_net, model.rotational_mols)

            # check if the mapped_emu has only single influx
            if mapped_emu.met in model.single_influx:

                if isinstance(mapped_emu, EMU) and emu_net.rotational_emu2binary.get(str(mapped_emu), False):
                    if emu_net.stack and temp == emu_net.stack[-1]:
                        alt_emu= emu_net.stack.pop()
                        # alt_emu= emu.net.stack[-1]
                        emu_net.visited[str(alt_emu)] = alt_emu
                    else:
                        alt_emu= temp

                    if emu_net.stack and mapped_emu == emu_net.stack[-1]:
                        main_emu= emu_net.stack.pop()
                        # main_emu= emu.net.stack[-1]
                        emu_net.visited[str(main_emu)] = main_emu # it should be checked again
                    else:
                        main_emu= mapped_emu

                    alt_rxn2decomposed, emu_net = decompose_EMU(alt_emu, model, emu_net)
                    alt_decomposed= next(iter(alt_rxn2decomposed.values()))[0]

                    main_rxn2decomposed, emu_net = decompose_EMU(main_emu, model, emu_net)
                    main_decomposed= next(iter(main_rxn2decomposed.values()))[0]
                    rxn2decomposedEMUs[rxn]= [main_decomposed, alt_decomposed]

                else:
                    if emu_net.stack and mapped_emu == emu_net.stack[-1]:
                        emu_net.stack.pop()
                        emu_net.visited[mapped_emu_str] = mapped_emu
                    new_rxn2decomposed, emu_net = decompose_EMU(mapped_emu, model, emu_net)
                    new_decomposed = next(iter(new_rxn2decomposed.values()))[0]
                    rxn2decomposedEMUs[rxn]= [new_decomposed]

            else:
                rxn2decomposedEMUs[rxn]= [mapped_emu]

        else:
            mapped_emus = []
            for mapped_emu_str in mapped_emu_list:
                created_emu, emu_net, temp = check_new_emu(mapped_emu_str, emu_net, model.rotational_mols)

                if created_emu.met in model.single_influx:
                    if emu_net.stack and created_emu == emu_net.stack[-1]:
                        emu_net.stack.pop()
                        emu_net.visited[mapped_emu_str] = created_emu
                    new_rxn2decomposed, emu_net = decompose_EMU(created_emu, model, emu_net)
                    new_decomposed = next(iter(new_rxn2decomposed.values()))[0]

                    mapped_emus.append(new_decomposed)
                else:
                    mapped_emus.append(created_emu)

            mapped_emu = JointEMU(mapped_emus)
            rxn2decomposedEMUs[rxn] = [mapped_emu]

    return rxn2decomposedEMUs, emu_net




def check_new_emu(emu_str, emu_net, rotational_emus):

    rotational_flag= False
    alt_emu= None
    if emu_str.split('_')[0] in rotational_emus:

        met, pos = emu_str.split('_')
        n = rotational_emus[met]
        rev_map = {i: n - 1 - i for i in range(n)}
        pos_rev = ''.join(sorted(str(rev_map[int(p) - 1] + 1) for p in pos))
        alt_str= f"{met}_{max(pos, pos_rev)}"
        emu_str= f"{met}_{min(pos, pos_rev)}"
        if alt_str != emu_str:
            rotational_flag= True
            emu_net.rotational_emu2binary[emu_str] = True
            emu_net.rotational_emu2binary[alt_str] = True
        else:
            rotational_flag= False
            emu_net.rotational_emu2binary[emu_str] = False

    if emu_str in emu_net.visited:
        mapped_emu = emu_net.visited[emu_str]
    elif emu_str in [str(temp_emu) for temp_emu in emu_net.stack]:
        mapped_emu = emu_net.stack[[str(temp_emu) for temp_emu in emu_net.stack].index(emu_str)]
    else:
        mapped_emu = EMU(emu_str)
        emu_net.stack.append(mapped_emu)
        emu_net.created_count += 1

    if rotational_flag:
        if alt_str in emu_net.visited:
            alt_emu= emu_net.visited[alt_str]
        elif alt_str in [str(temp_emu) for temp_emu in emu_net.stack]:
            alt_emu= emu_net.stack[[str(temp_emu) for temp_emu in emu_net.stack].index(alt_str)]
        else:
            alt_emu= EMU(alt_str)
            emu_net.stack.append(alt_emu)
            emu_net.created_count += 1
            emu_net.rotational_emus2emu[emu_str] = mapped_emu
            emu_net.rotational_emus2emu[alt_str] = mapped_emu

    return mapped_emu, emu_net, alt_emu


